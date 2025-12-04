import copy
import json
import os
import warnings
from absl import app, flags
from tqdm import trange

import torch
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.utils import make_grid, save_image
from torchvision import transforms

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model.model import UNet
from utils.augmentation import *
from dataset import ImbalanceCIFAR100, ImbalanceCIFAR10
from score.both import get_inception_and_fid_score
from utils.augmentation import KarrasAugmentationPipeline

from loss_tracker import LossTracker


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load model.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
flags.DEFINE_bool('improve', False, help='use improved diffusion network implemented by OpenAI')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help='gradient norm clipping')
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help='ema decay rate')
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_bool('conditional', False, help='conditional generation')
flags.DEFINE_bool('weight', False, help='reweight')
flags.DEFINE_bool('cotrain', False, help='cotrain with an adjusted classifier or not')
flags.DEFINE_bool('logit', False, help='use logit adjustment or not')
flags.DEFINE_bool('augm', False, help='whether to use ADA augmentation')
flags.DEFINE_bool('cfg', True, help='whether to train unconditional generation with with 10%  probability')
# Dataset
flags.DEFINE_string('root', './', help='path of dataset')
flags.DEFINE_string('data_type', 'cifar100', help='data type, must be in [cifar10, cifar100, cifar10lt, cifar100lt]')
flags.DEFINE_float('imb_factor', 0.01, help='imb_factor for long tail dataset')
flags.DEFINE_float('num_class', 0, help='number of class of the pretrained model')
flags.DEFINE_bool('download', True, help='Download the dataset if not already present')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/', help='log directory')
flags.DEFINE_integer('sample_size', 64, 'sampling size of images')
flags.DEFINE_integer('sample_step', 10000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 100000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_integer('private_num_images', 0, help='the number of private images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_string('sample_name', 'saved', help='name for a set of samples to be saved or to be evaluated')
flags.DEFINE_bool('sampled', False, help='evaluate sampled images')
flags.DEFINE_string('sample_method', 'cfg', help='sampling method, must be in [cfg, cond, uncond]')
flags.DEFINE_float('omega', 0.0, help='guidance strength for cfg sampling method')
flags.DEFINE_bool('prd', False, help='evaluate precision and recall (F_beta), only evaluated with 50k samples')
flags.DEFINE_bool('improved_prd', False, help='evaluate improved precision and recall, only evaluated with 50k samples')
# CBDM hyperparameters
flags.DEFINE_bool('cb', False, help='train with class-balancing(adjustment) loss')
flags.DEFINE_float('tau', 1.0, help='weight for the class-balancing(adjustment) loss')
# CBDM finetuning mechanism
flags.DEFINE_bool('finetune', False, help='finetuned based on a pretrained model')
flags.DEFINE_string('finetuned_logdir', '', help='logdir for the new model, where FLAGS.logdir will be the folder for \
                     the pretrained model')
flags.DEFINE_integer('ckpt_step', 0, help='step to reload the pretained checkpoint')
# SupCon loss parameters
flags.DEFINE_bool('supcon', False, help='use supervised contrastive loss')
flags.DEFINE_float('supcon_weight', 0.5, help='weight for supcon loss')
flags.DEFINE_float('temperature_scaling', 0.3, help='temperature for loss scaling')
flags.DEFINE_float('supcon_temp', 0.1, help='temperature for SupConLoss')
# Visualization flags
flags.DEFINE_bool('visualize_bottleneck', False, help='visualize bottleneck features (before projection)')
flags.DEFINE_bool('visualize_projection', False, help='visualize projection features (means)')

# AMP parameters
flags.DEFINE_bool('amp', False, help='Use Automatic Mixed Precision for training')

device = torch.device('cuda')


def uniform_sampling(n, N, k):
    return np.stack([np.random.randint(int(N/n)*i, int(N/n)*(i+1), k) for i in range(n)])


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def evaluate(sampler, model, sampled):
    if not sampled:
        model.eval()
        with torch.no_grad():
            images = []
            labels = []
            desc = 'generating images'
            for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
                batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
                x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
                batch_images, batch_labels = sampler(x_T.to(device),
                                                     omega=FLAGS.omega,
                                                     method=FLAGS.sample_method)
                images.append((batch_images.cpu() + 1) / 2)
                if FLAGS.sample_method!='uncond' and batch_labels is not None:
                    labels.append(batch_labels.cpu())
            images = torch.cat(images, dim=0).numpy()
        np.save(os.path.join(FLAGS.logdir, '{}_{}_samples_ema_{}.npy'.format(
                                            FLAGS.sample_method, FLAGS.omega,
                                            FLAGS.sample_name)), images)
        if FLAGS.sample_method != 'uncond':
            labels = torch.cat(labels, dim=0).numpy()
            np.save(os.path.join(FLAGS.logdir, '{}_{}_labels_ema_{}.npy'.format(
                                            FLAGS.sample_method, FLAGS.omega,
                                            FLAGS.sample_name)), labels)
        model.train()
    else:
        labels = None
        images = np.load(os.path.join(FLAGS.logdir, '{}_{}_samples_ema_{}.npy'.format(
                                            FLAGS.sample_method, FLAGS.omega,
                                            FLAGS.sample_name)))

        if FLAGS.sample_method != 'uncond':
            labels = np.load(os.path.join(FLAGS.logdir, '{}_{}_labels_ema_{}.npy'.format(
                                                FLAGS.sample_method, FLAGS.omega,
                                                FLAGS.sample_name)))
    save_image(
        torch.tensor(images[:256]),
        os.path.join(FLAGS.logdir, 'visual_ema_{}_{}_{}.png'.format(
                                    FLAGS.sample_method, FLAGS.omega, FLAGS.sample_name)),
        nrow=16)

    (IS, IS_std), FID, prd_score, ipr = get_inception_and_fid_score(
        images, labels, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, FLAGS=FLAGS)

    return (IS, IS_std), FID, prd_score, ipr


def train():
    if FLAGS.augm:
        tran_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([FLAGS.img_size, FLAGS.img_size]),
            transforms.ToPILImage(),
            KarrasAugmentationPipeline(0.12),
        ])
    else:
        tran_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize([FLAGS.img_size, FLAGS.img_size])
        ])

    if FLAGS.data_type == 'cifar10':
        dataset = CIFAR10(
                root=FLAGS.root,
                train=True,
                download=True,
                transform=tran_transform)
    elif FLAGS.data_type == 'cifar100':
        dataset = CIFAR100(
                root=FLAGS.root,
                train=True,
                download=True,
                transform=tran_transform)
    elif FLAGS.data_type == 'cifar10lt':
        dataset = ImbalanceCIFAR10(
                root=FLAGS.root,
                imb_type='exp',
                imb_factor=FLAGS.imb_factor,
                rand_number=0,
                train=True,
                transform=tran_transform,
                target_transform=None,
                download=True)
    elif FLAGS.data_type == 'cifar100lt':
        dataset = ImbalanceCIFAR100(
                root=FLAGS.root,
                imb_type='exp',
                imb_factor=FLAGS.imb_factor,
                rand_number=0,
                train=True,
                transform=tran_transform,
                target_transform=None,
                download=True)
    else:
        print('Please enter a data type included in [cifar10, cifar100, cifar10lt, cifar100lt]')

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size,
        shuffle=True, num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)
    print('Dataset {} contains {} images with {} classes'.format(
        FLAGS.data_type, len(dataset.targets), len(np.unique(dataset.targets))))

    # get class weights for the current dataset
    def class_counter(all_labels):
        all_classes_count = torch.Tensor(np.unique(all_labels, return_counts=True)[1])
        return all_classes_count / all_classes_count.sum()
    weight = class_counter(dataset.targets)

    # Print SupCon loss info if enabled
    if FLAGS.supcon:
        print(f"SupCon loss enabled with temperature={FLAGS.supcon_temp}, weight={FLAGS.supcon_weight}")

    # model setup
    FLAGS.num_class = 100 if 'cifar100' in FLAGS.data_type else 10
    
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        cond=FLAGS.conditional, augm=FLAGS.augm, num_class=FLAGS.num_class)
    ema_model = copy.deepcopy(net_model)

    # training setup
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    
    # Initialize GradScaler for AMP
    scaler = None
    if FLAGS.amp:
        scaler = GradScaler()
        print("Using Automatic Mixed Precision (AMP) for training")
    
    # Load checkpoint if needed
    if FLAGS.ckpt_step != 0:
        ckpt = torch.load(os.path.join(FLAGS.logdir if not FLAGS.finetune else FLAGS.finetuned_logdir,
                                      'ckpt_{}.pt'.format(FLAGS.ckpt_step)), map_location='cpu')
        net_model.load_state_dict(ckpt['net_model'])
        ema_model.load_state_dict(ckpt['ema_model'])
        optim.load_state_dict(ckpt['optim'])
        sched.load_state_dict(ckpt['sched'])
        
        # Also load scaler state if it exists
        if FLAGS.amp and scaler is not None and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
            print(f"Resumed AMP GradScaler state from checkpoint at step {FLAGS.ckpt_step}")
    
    # Initialize trainer with SupCon parameters
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, dataset,
        FLAGS.num_class, FLAGS.cfg, FLAGS.cb, FLAGS.tau, weight, FLAGS.finetune,
        FLAGS.supcon, FLAGS.supcon_weight, FLAGS.temperature_scaling,
        FLAGS.supcon_temp).to(device)
    
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.num_class, FLAGS.img_size, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.num_class, FLAGS.img_size, FLAGS.var_type).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    if not os.path.exists(os.path.join(FLAGS.logdir, 'sample')):
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    else:
        print('LOGDIR already exists.')
    writer = SummaryWriter(FLAGS.logdir)
    writer.flush()
    
    # fix seeds for generation to keep generated images comparable
    fixed_x_T = torch.randn(min(FLAGS.sample_size, 100), 3, FLAGS.img_size, FLAGS.img_size)
    fixed_x_T = fixed_x_T.to(device)

    # backup all arguments
    with open(os.path.join(FLAGS.logdir, 'flagfile.txt'), 'w') as f:
        f.write(FLAGS.flags_into_string())

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    loss_tracker = LossTracker(FLAGS.logdir, save_interval=10)
    with trange(FLAGS.ckpt_step, FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0, y_0 = next(datalooper)

            # when using ADA, the augmentation parameters will also be returned by the dataloader
            augm = None
            if type(x_0) == list:
                x_0, augm = x_0
                augm = augm.to(device)

            x_0 = x_0.to(device)
            y_0 = y_0.to(device)

            # Modify the forward pass to use AMP autocast
            if FLAGS.amp:
                # Forward pass with autocast
                with torch.cuda.amp.autocast():
                    loss_ddpm, loss_cb, loss_supcon = trainer(x_0, y_0, augm)
                    
                    # Combine losses
                    loss = loss_ddpm
                    if FLAGS.cb:
                        loss = loss + loss_cb
                    if FLAGS.supcon:
                        loss = loss + loss_supcon
                
                # gradient rescaling for the supcon component
                temperature = FLAGS.supcon_temp  
                
                if FLAGS.supcon:
                    # Create a rescaled loss where only the supcon component is rescaled
                    rescaled_loss = loss_ddpm
                    if FLAGS.cb:
                        rescaled_loss = rescaled_loss + loss_cb
                    rescaled_loss = rescaled_loss + (loss_supcon * temperature)
                else:
                    rescaled_loss = loss
                
                # Proceed with AMP workflow
                scaler.scale(rescaled_loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
                success = scaler.step(optim)
                if not success:
                    print(f"Warning: Optimizer step skipped due to NaN/Inf gradients at step {step}")
                scaler.update()
            else:
                # non-AMP path
                loss_ddpm, loss_cb, loss_supcon = trainer(x_0, y_0, augm)
                
                # Start with DDPM loss
                loss = loss_ddpm
                
                # Add CB loss only if enabled
                if FLAGS.cb:
                    loss = loss + loss_cb
                    
                # Add supcon loss only if enabled
                temperature = FLAGS.supcon_temp
                if FLAGS.supcon:
                    loss = loss_ddpm + (loss_supcon * temperature)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
                optim.step()
            
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # logs - only log what's actually being used
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('loss_ddpm', loss_ddpm, step)
            if FLAGS.cb:
                writer.add_scalar('loss_cb', loss_cb, step)
            if FLAGS.supcon:
                writer.add_scalar('loss_supcon', loss_supcon, step)
            
            pbar.set_postfix(loss='%.5f' % loss)

            # sample
            if step != FLAGS.ckpt_step and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0, _  = ema_sampler(fixed_x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'fixed_x_T': fixed_x_T,
                }
                
                # Save scaler state if using AMP
                if FLAGS.amp and scaler is not None:
                    ckpt['scaler'] = scaler.state_dict()
                
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(step)))

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                # net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID, prd_score, ipr = evaluate(ema_sampler, ema_model, False)
                metrics = {
                    'IS': ema_IS[0],
                    'IS_std': ema_IS[1],
                    'FID': ema_FID,
                    'PRD_Precision': prd_score[0],
                    'PRD_Recall': prd_score[1],
                    'IPR_Precision': ipr[0],
                    'IPR_Recall': ipr[1]
                }
                print(step, metrics)
                pbar.write(
                    '%d/%d ' % (step, FLAGS.total_steps) +
                    ', '.join('%s:%.5f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + '\n')
    writer.close()


def eval():
    FLAGS.num_class = 100 if 'cifar100' in FLAGS.data_type else 10
    
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        cond=FLAGS.conditional, augm=FLAGS.augm, num_class=FLAGS.num_class)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.num_class, FLAGS.img_size, FLAGS.var_type).to(device)

    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)
    FLAGS.sample_name = '{}_N{}_STEP{}'.format(FLAGS.sample_name, FLAGS.num_images, FLAGS.ckpt_step)

    # load ema model (almost always better than the model) and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(FLAGS.ckpt_step)), map_location='cpu')

    # Update FID cache path
    if 'cifar100' in FLAGS.data_type:
        FLAGS.fid_cache = './stats/cifar100.train.npz'
    else:
        FLAGS.fid_cache = './stats/cifar10.train.npz'

    if not FLAGS.sampled:
        model.load_state_dict(ckpt['ema_model'])
    else:
        model = None

    (IS, IS_std), FID, prd_score, ipr = evaluate(sampler, model, FLAGS.sampled)

    print('logdir', FLAGS.logdir)
    print("Model(EMA): IS:%6.5f(%.5f), FID:%7.5f \n" % (IS, IS_std, FID))
    print("Improved PRD:%6.5f, RECALL:%7.5f \n" % (ipr[0], ipr[1]))
    print("PRD PRECISION:%6.5f, RECALL:%7.5f \n" % (prd_score[0], prd_score[1]))

    with open(os.path.join(FLAGS.logdir,  'res_ema_{}.txt'.format(FLAGS.sample_name)), 'a+') as f:
        f.write("Settings: NUM:{} EPOCH:{}, OMEGA:{}, METHOD:{} \n" .format(FLAGS.num_images, FLAGS.ckpt_step, FLAGS.omega,FLAGS.sample_method))
        f.write("Model(EMA): IS:%6.5f(%.5f), FID:%7.5f \n" % (IS, IS_std, FID))
        f.write("Improved PRD:%6.5f, RECALL:%7.5f \n" % (ipr[0], ipr[1]))
        f.write("PRD PRECISION:%6.5f, RECALL:%7.5f \n" % (prd_score[0], prd_score[1]))
    f.close()

def visualize_latent_space():
    """
    Visualize the latent space of a trained model using t-SNE or UMAP.
    This function loads a trained model, encodes validation data, and creates
    visualizations of the latent space with different coloring based on classes.
    """
    FLAGS.num_class = 100 if 'cifar100' in FLAGS.data_type else 10
    
    # Load model
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        cond=FLAGS.conditional, augm=FLAGS.augm, num_class=FLAGS.num_class)
    
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.num_class, FLAGS.img_size, FLAGS.var_type).to(device)
    
    # Load checkpoint
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(FLAGS.ckpt_step)), map_location='cpu')
    model.load_state_dict(ckpt['ema_model'])
    model.to(device)
    model.eval()
    
    # Prepare dataset for visualization - ensure consistent image size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize([FLAGS.img_size, FLAGS.img_size])
    ])
    
    # Prepare appropriate dataset
    if FLAGS.data_type == 'cifar10':
        dataset = CIFAR10(
                root=FLAGS.root,
                train=False,  # Use test set for visualization
                download=True,
                transform=transform)
    elif FLAGS.data_type == 'cifar100':
        dataset = CIFAR100(
                root=FLAGS.root,
                train=False,
                download=True,
                transform=transform)
    elif FLAGS.data_type == 'cifar10lt':
        dataset = ImbalanceCIFAR10(
                root=FLAGS.root,
                imb_type='exp',
                imb_factor=FLAGS.imb_factor,
                rand_number=0,
                train=True,
                transform=transform,
                target_transform=None,
                download=True)
    elif FLAGS.data_type == 'cifar100lt':
        dataset = ImbalanceCIFAR100(
                root=FLAGS.root,
                imb_type='exp',
                imb_factor=FLAGS.imb_factor,
                rand_number=0,
                train=True,
                transform=transform,
                target_transform=None,
                download=True)
    else:
        print('Please enter a data type included in [cifar10, cifar100, cifar10lt, cifar100lt]')
        return
    
    # Create dataloader with smaller batch for visualization
    vis_batch_size = 128
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=vis_batch_size, shuffle=False, num_workers=FLAGS.num_workers)
    
    # Create folder for visualizations
    vis_dir = os.path.join(FLAGS.logdir, 'latent_vis')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Determine which visualizations to create
    visualize_bottleneck = hasattr(FLAGS, 'visualize_bottleneck') and FLAGS.visualize_bottleneck
    visualize_projection = hasattr(FLAGS, 'visualize_projection') and FLAGS.visualize_projection
    
    # If none specified, set default based on model type
    if not visualize_bottleneck and not visualize_projection:
        if FLAGS.supcon:
            visualize_projection = True
        else:
            visualize_bottleneck = True
    
    try:
        if visualize_bottleneck:
            print(f"Starting bottleneck feature extraction for {len(dataset)} samples...")
            
            # Extract bottleneck features
            bottleneck_encodings, bottleneck_labels = sampler.get_encodings(
                dataloader, device, timestep=0.1, get_bottleneck=True)
            
            # Save bottleneck features
            torch.save({
                'encodings': bottleneck_encodings.cpu(),
                'labels': bottleneck_labels.cpu()
            }, os.path.join(vis_dir, f'bottleneck_encodings_step_{FLAGS.ckpt_step}.pt'))
            
            print(f"Saved bottleneck encodings to {os.path.join(vis_dir, f'bottleneck_encodings_step_{FLAGS.ckpt_step}.pt')}")
            
            # Visualize bottleneck features
            visualize_features(bottleneck_encodings, bottleneck_labels, vis_dir, 
                               f'bottleneck_step_{FLAGS.ckpt_step}', prefix='bottleneck')
        
        if visualize_projection and hasattr(model, 'mean_proj'):
            print(f"Starting projection feature extraction for {len(dataset)} samples...")
            
            # Extract projection features (means)
            projection_encodings, projection_labels = sampler.get_encodings(
                dataloader, device, timestep=0.1, get_bottleneck=False)
            
            # Save projection features
            torch.save({
                'encodings': projection_encodings.cpu(),
                'labels': projection_labels.cpu()
            }, os.path.join(vis_dir, f'projection_encodings_step_{FLAGS.ckpt_step}.pt'))
            
            print(f"Saved projection encodings to {os.path.join(vis_dir, f'projection_encodings_step_{FLAGS.ckpt_step}.pt')}")
            
            # Visualize projection features
            visualize_features(projection_encodings, projection_labels, vis_dir, 
                               f'projection_step_{FLAGS.ckpt_step}', prefix='projection')
            
        print(f"Latent space visualizations completed and saved to {vis_dir}")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def visualize_features(encodings, labels, vis_dir, filename_base, prefix=''):
    """
    Visualize features using t-SNE and UMAP.
    
    Args:
        encodings: Feature encodings tensor
        labels: Labels tensor
        vis_dir: Directory to save visualizations
        filename_base: Base filename for saved visualizations
        prefix: Prefix for console output
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        
        # Subsample for t-SNE to keep visualization manageable
        max_points = 20000
        if len(encodings) > max_points:
            indices = np.random.choice(len(encodings), max_points, replace=False)
            encodings_sample = encodings[indices].cpu().numpy()
            labels_sample = labels[indices].cpu().numpy()
        else:
            encodings_sample = encodings.cpu().numpy()
            labels_sample = labels.cpu().numpy()
        
        print(f"Applying t-SNE on {len(encodings_sample)} samples for {prefix} features...")
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        encodings_2d = tsne.fit_transform(encodings_sample)
        
        # Plot with class coloring
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(encodings_2d[:, 0], encodings_2d[:, 1], 
                            c=labels_sample, cmap='inferno', alpha=0.2, s=4)
        plt.colorbar(scatter, label='Class')
        plt.title(f'T-SNE visualization of {prefix} space (step {FLAGS.ckpt_step})')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'tsne_{filename_base}.png'), dpi=300)
        
        print(f"Saved t-SNE visualization to {os.path.join(vis_dir, f'tsne_{filename_base}.png')}")
        
        # Also try to generate UMAP if available
        try:
            import umap
            
            print(f"Applying UMAP visualization for {prefix} features...")
            reducer = umap.UMAP(random_state=42)
            encodings_umap = reducer.fit_transform(encodings_sample)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(encodings_umap[:, 0], encodings_umap[:, 1], 
                                c=labels_sample, cmap='inferno', alpha=0.2, s=4)
            plt.colorbar(scatter, label='Class')
            plt.title(f'UMAP visualization of {prefix} space (step {FLAGS.ckpt_step})')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'umap_{filename_base}.png'), dpi=300)
            
            print(f"Saved UMAP visualization to {os.path.join(vis_dir, f'umap_{filename_base}.png')}")
            
        except ImportError:
            print(f"UMAP package not available for visualization of {prefix} features")
            
    except ImportError:
        print(f"Visualization packages (matplotlib, sklearn) not available for {prefix} features")


def main(argv):
    # suppress inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()
    
    # Check if we should run visualization
    if FLAGS.visualize_bottleneck or FLAGS.visualize_projection or (FLAGS.supcon and not FLAGS.train and not FLAGS.eval):
        visualize_latent_space()
    
    # If no operation was specified, print help message
    if not (FLAGS.train or FLAGS.eval or FLAGS.visualize_bottleneck or FLAGS.visualize_projection or 
           (FLAGS.supcon and not FLAGS.train and not FLAGS.eval)):
        print('Add --train, --eval, --visualize_bottleneck, or --visualize_projection to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
