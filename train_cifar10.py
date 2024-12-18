# -*- coding: utf-8 -*-
'''
Train CIFAR10 with PyTorch and Vision Transformers!
This code now includes multiplicative regularization similar to the provided CNN code snippet.
It also includes feature normalization for the regularization terms.
'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import math
import random
import csv
import time
import pickle
import argparse
import os

import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer

# Functions to compute regularization terms
def compute_l2_sum(model, features_normalization):
    # Computes the sum of (||W||^2 / f_out) over linear/conv layers if features_normalization is True
    norms = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            denom = m.weight.size(0) if features_normalization else 1
            norms.append((torch.norm(m.weight, p=2) ** 2) / denom)
    if len(norms) == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.sum(torch.stack(norms))

def compute_l2_mul(model, features_normalization):
    # Computes the product of (||W||^2 / f_out) over linear/conv layers if features_normalization is True
    norms = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            denom = m.weight.size(0) if features_normalization else 1
            norms.append((torch.norm(m.weight, p=2) ** 2) / denom)
    if len(norms) == 0:
        return torch.tensor(1.0, device=next(model.parameters()).device) # product neutral element
    return torch.prod(torch.stack(norms))


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

parser.add_argument('--use_lambda_scheduler', action='store_true', help='use cyclic cosine lambda scheduling for weight_decay with AdamW')
parser.add_argument('--initial-lambda', type=float, default=0.001, help='initial lambda (weight_decay) for AdamW optimizer')
parser.add_argument('--use_sqrt_lambda_scheduler', action='store_true', help='use sqrt lambda decay scheduler')

# New arguments for multiplicative and sum regularization
parser.add_argument('--lambda_sum', type=float, default=0.0, help='L2 sum regularization coefficient (λ_sum)')
parser.add_argument('--lambda_mul', type=float, default=0.0, help='L2 multiplicative regularization coefficient (λ_mul)')
parser.add_argument('--features_normalization', action='store_true', help='Normalize by number of output features for regularization')


args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

# Set seed
set_seed(7)

initial_lambda = args.initial_lambda

use_cyclic_lambda_scheduler = args.use_lambda_scheduler and args.opt == "adamW"
use_sqrt_lambda_scheduler = args.use_sqrt_lambda_scheduler and args.opt == "adamW"

# Determine experiment type for logging
if use_cyclic_lambda_scheduler:
    experiment_type = "cosine_cyclic_decay"
elif use_sqrt_lambda_scheduler:
    experiment_type = "sqrt_decay"
else:
    experiment_type = "none"

if args.lambda_sum != 0.0 or args.lambda_mul != 0.0:
    experiment_type = "multiplicative_reg_" + experiment_type

usewandb = not args.nowandb
if usewandb:
    import wandb
    if use_cyclic_lambda_scheduler:
        watermark = "{}_wd{}_lambda_scheduler_cyclic".format(args.net, initial_lambda)
    elif use_sqrt_lambda_scheduler:
        watermark = "{}_wd{}_lambda_scheduler_sqrt".format(args.net, initial_lambda)
    else:
        watermark = "{}_wd{}_no_scheduler".format(args.net, initial_lambda)
    if args.lambda_sum != 0.0 or args.lambda_mul != 0.0:
        watermark += "_multiplicative_reg"

    wandb.init(project="cifar10-challange",
               name=watermark)
    wandb.config.update(args)


bs = int(args.bs)
imsize = int(args.size)
use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if aug:
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Building model..')
if args.net=='res18':
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    net = ResNet50()
elif args.net=='res101':
    net = ResNet101()
elif args.net=="convmixer":
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
elif args.net=="mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
        image_size = 32,
        channels = 3,
        patch_size = args.patch,
        dim = 512,
        depth = 6,
        num_classes = 10
    )
elif args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 4,
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1
    )
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512
    )
elif args.net=="vit":
    net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
elif args.net=="cait":
    from models.cait import CaiT
    net = CaiT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        cls_depth=2,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05
    )
elif args.net=="cait_small":
    from models.cait import CaiT
    net = CaiT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        cls_depth=2,
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05
    )
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                 num_classes=10,
                 downscaling_factors=(2,2,2,1))

if 'cuda' in device:
    print(device)
    if args.dp:
        print("using data parallel")
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['model'])
    best_acc = 0 # we don't have acc in the saved state in this snippet
    start_epoch = 0

criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
elif args.opt == "adamW":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=initial_lambda)
else:
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Schedulers
def cyclic_cosine_lambda(epoch, initial_lambda, lambda_min, T_cycle, decay_factor=0.8):
    cycle = epoch // T_cycle
    max_lambda = initial_lambda * (decay_factor ** cycle)
    return lambda_min + 0.5 * (max_lambda - lambda_min) * (
        1 + math.cos(math.pi * (epoch % T_cycle) / T_cycle)
    )

def sqrt_lambda_decay(epoch, initial_lambda, total_epochs):
    return initial_lambda / math.sqrt(epoch+1)

lambda_min = 0.0
T_cycle = 100
decay_factor = 0.3

use_amp = not args.noamp
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    grad_norm = 0.0
    l2_sum_val = 0.0
    l2_mul_val = 0.0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # Compute regularization terms
            l2_sum_loss = torch.tensor(0.0, device=device)
            l2_mul_loss = torch.tensor(0.0, device=device)

            if args.lambda_sum != 0.0:
                l2_sum_loss = compute_l2_sum(net, args.features_normalization)
            if args.lambda_mul != 0.0:
                l2_mul_loss = compute_l2_mul(net, args.features_normalization)

            total_loss = loss + args.lambda_sum * l2_sum_loss + args.lambda_mul * l2_mul_loss

        scaler.scale(total_loss).backward()

        if batch_idx == len(trainloader)-1:
            with torch.no_grad():
                total_grad_norm = 0.0
                for p in net.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                grad_norm = total_grad_norm**0.5

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += total_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # For logging purposes
        if args.lambda_sum != 0.0:
            l2_sum_val = l2_sum_loss.item()
        if args.lambda_mul != 0.0:
            l2_mul_val = l2_mul_loss.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Compute parameter norm
    with torch.no_grad():
        total_param_norm = 0.0
        for p in net.parameters():
            param_norm = p.data.norm(2)
            total_param_norm += param_norm.item() ** 2
        param_norm = total_param_norm**0.5

    return train_loss/(batch_idx+1), grad_norm, param_norm, l2_sum_val, l2_mul_val

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

if usewandb:
    wandb.watch(net)

net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss, grad_norm, param_norm, l2_sum_val, l2_mul_val = train(epoch)
    val_loss, acc = test(epoch)

    # Update lambda if scheduler is used
    if use_cyclic_lambda_scheduler:
        current_lambda = cyclic_cosine_lambda(epoch, initial_lambda, lambda_min, T_cycle, decay_factor)
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = current_lambda
    elif use_sqrt_lambda_scheduler:
        current_lambda = sqrt_lambda_decay(epoch, initial_lambda, args.n_epochs)
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = current_lambda
    else:
        current_lambda = initial_lambda

    list_loss.append(val_loss)
    list_acc.append(acc)
    
    if usewandb:
        log_dict = {
            'epoch': epoch, 
            'train_loss': trainloss, 
            'val_loss': val_loss, 
            'val_acc': acc, 
            'lr': optimizer.param_groups[0]["lr"], 
            'weight_decay': optimizer.param_groups[0]["weight_decay"],
            'grad_norm': grad_norm,
            'param_norm': param_norm,
            'epoch_time': time.time()-start,
            'experiment_type': experiment_type
        }
        if args.lambda_sum != 0.0:
            log_dict['l2_sum_loss'] = l2_sum_val
        if args.lambda_mul != 0.0:
            log_dict['l2_mul_loss'] = l2_mul_val
        wandb.log(log_dict)

    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)

if usewandb:
    wandb.save("wandb_{}.h5".format(args.net))


###################################
# After training, calculate margins and save results
###################################
net.eval()
margins = []
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        for i in range(outputs.size(0)):
            correct_class = targets[i].item()
            correct_logit = outputs[i, correct_class].item()
            mask = torch.ones(outputs.size(1), dtype=torch.bool, device=device)
            mask[correct_class] = False
            other_max = outputs[i][mask].max().item()
            margin = correct_logit - other_max
            margins.append(margin)

os.makedirs("results", exist_ok=True)

results_dict = {
    "arguments": vars(args),
    "final_val_losses": list_loss,
    "final_val_accuracies": list_acc,
    "final_margins": margins,
    "initial_lambda": initial_lambda
}

random_number = random.randint(1000, 9999)
pickle_filename = f"results/results_{args.net}_patch{args.patch}_{random_number}.pkl"

with open(pickle_filename, "wb") as f:
    pickle.dump(results_dict, f)
