"""
CIFAR-10 trainer — based on akamaster/pytorch_resnet_cifar10 trainer.py (BSD-2-Clause, (c) 2018 Yerlan Idelbayev).

[study] changes vs. the original trainer.py
  1. --plain           : train the plain counterpart (resnet.py residual=False)
  2. LR schedule       : per-ITERATION schedule as in the paper (He et al. 2016, Sec 4.2):
                         lr 0.1, /10 at 50% and 75% of training, stop at --iters (default 64k = paper)
                         (original: MultiStepLR [100,150] hard-coded to epochs, breaks if --epochs changes)
  3. gradient norms    : weight-gradient L2 norm of every conv layer (stem + each block conv1/conv2),
                         averaged over each epoch, plus an "epoch 0" row measured at initialization
  4. logging / resume  : log.csv + grads.csv per run, full checkpoint (model+optimizer+scheduler+RNG)
                         every epoch -> safe to re-run after a Colab disconnect (original --resume was broken)
  Also: fixed seed, no DataParallel (single GPU), optional --amp, ResNet-110/1202 warm-up now actually ends.
Everything else (data augmentation, normalization, optimizer, wd, batch size, model code) is unchanged.
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='ResNet vs plain on CIFAR-10 (study fork of akamaster trainer)')
parser.add_argument('--arch', '-a', default='resnet20', choices=model_names)
parser.add_argument('--plain', action='store_true', help='[study] remove shortcuts (plain net)')
parser.add_argument('--iters', default=64000, type=int,
                    help='[study] total training iterations (paper: 64k). lr /10 at 50%% and 75%%')
parser.add_argument('-j', '--workers', default=2, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
parser.add_argument('--seed', default=0, type=int, help='[study] random seed')
parser.add_argument('--amp', action='store_true', help='[study] mixed precision (faster, not in paper)')
parser.add_argument('--grad-every', default=20, type=int,
                    help='[study] measure conv gradient norms every N iterations')
parser.add_argument('--data-dir', default='./data', type=str)
parser.add_argument('--save-dir', default='runs', type=str,
                    help='parent dir; this run goes to <save-dir>/<arch>_<residual|plain>_s<seed>')
parser.add_argument('--fake-data', action='store_true', help=argparse.SUPPRESS)  # smoke test only


def run_name(a):
    return f"{a.arch}_{'plain' if a.plain else 'residual'}_s{a.seed}"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def conv_layers(model):
    """(name, module) of every 3x3 conv in network order: stem, then layerX.Y.conv1/conv2."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]


def get_loaders(args):
    # unchanged from akamaster
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
                                   transforms.ToTensor(), normalize])
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])
    if args.fake_data:
        train_set = datasets.FakeData(512, (3, 32, 32), 10, transforms.Compose([transforms.ToTensor(), normalize]))
        test_set = datasets.FakeData(256, (3, 32, 32), 10, transforms.Compose([transforms.ToTensor(), normalize]))
    else:
        train_set = datasets.CIFAR10(root=args.data_dir, train=True, transform=train_tf, download=True)
        test_set = datasets.CIFAR10(root=args.data_dir, train=False, transform=test_tf, download=True)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=False, generator=g, persistent_workers=args.workers > 0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss_sum += criterion(out, y).item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return loss_sum / n, 100.0 * (1 - correct / n)


def grad_norms(convs):
    return [m.weight.grad.detach().norm().item() for _, m in convs]


def init_grad_norms(model, loader, criterion, device, convs, batches=10):
    """[study] gradient norms at initialization (no parameter update), averaged over a few batches."""
    model.train()
    acc = np.zeros(len(convs))
    it = iter(loader)
    batches = min(batches, len(loader))
    for _ in range(batches):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        criterion(model(x), y).backward()
        acc += np.array(grad_norms(convs))
    model.zero_grad(set_to_none=True)
    return acc / batches


def atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_dir = os.path.join(args.save_dir, run_name(args))
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, 'DONE')):
        print(f'[{run_name(args)}] already finished -> skip')
        return

    seed_everything(args.seed)
    cudnn.benchmark = True
    model = resnet.__dict__[args.arch](residual=not args.plain).to(device)
    convs = conv_layers(model)
    n_params = sum(p.numel() for p in model.parameters())

    train_loader, test_loader = get_loaders(args)
    iters_per_epoch = len(train_loader)
    total_iters = args.iters
    n_epochs = math.ceil(total_iters / iters_per_epoch)
    milestones = [int(0.5 * total_iters), int(0.75 * total_iters)]   # paper: 32k, 48k of 64k

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # stepped every ITERATION (original: every epoch with fixed [100,150])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device == 'cuda')

    warmup = args.arch in ['resnet110', 'resnet1202'] and not args.plain  # paper: lr 0.01 until train err < 80%
    start_epoch, global_iter = 1, 0
    ckpt_path = os.path.join(run_dir, 'checkpoint.th')
    log_path, grad_path = os.path.join(run_dir, 'log.csv'), os.path.join(run_dir, 'grads.csv')

    if os.path.exists(ckpt_path):                       # [study] auto-resume
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['state_dict'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler'])
        random.setstate(ck['rng']['python']); np.random.set_state(ck['rng']['numpy'])
        torch.set_rng_state(ck['rng']['torch'])
        if device == 'cuda' and ck['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        train_loader.generator.set_state(ck['rng']['loader'])
        start_epoch, global_iter, warmup = ck['epoch'] + 1, ck['global_iter'], ck['warmup']
        print(f'[{run_name(args)}] resumed from epoch {ck["epoch"]} (iter {global_iter})')
    else:
        json.dump({**vars(args), 'run': run_name(args), 'params': n_params, 'iters_per_epoch': iters_per_epoch,
                   'epochs': n_epochs, 'milestones_iter': milestones, 'device': device,
                   'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                   'torch': torch.__version__, 'conv_layers': [n for n, _ in convs]},
                  open(os.path.join(run_dir, 'config.json'), 'w'), indent=1)
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'iter', 'lr', 'train_loss', 'train_err', 'test_loss',
                                    'test_err', 'epoch_time_s'])
        with open(grad_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'layer_idx', 'layer', 'grad_norm'])
        # epoch 0 = initialization
        te_loss, te_err = evaluate(model, test_loader, criterion, device)
        g0 = init_grad_norms(model, train_loader, criterion, device, convs)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([0, 0, args.lr, '', '', f'{te_loss:.5f}', f'{te_err:.3f}', 0])
        with open(grad_path, 'a', newline='') as f:
            w = csv.writer(f)
            for i, ((name, _), v) in enumerate(zip(convs, g0)):
                w.writerow([0, i, name, f'{v:.6g}'])
        seed_everything(args.seed)  # so the init measurement does not change the training stream
        train_loader.generator.manual_seed(args.seed)

    print(f'[{run_name(args)}] params={n_params} iters={total_iters} ({n_epochs} epochs x {iters_per_epoch}) '
          f'lr drops at iter {milestones} device={device} amp={args.amp}')

    if warmup and global_iter == 0:
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr * 0.1

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, correct, n = 0.0, 0, 0
        gsum, gcnt = np.zeros(len(convs)), 0
        for x, y in train_loader:
            if global_iter >= total_iters:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=scaler.is_enabled()):
                out = model(x)
                loss = criterion(out, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if global_iter % args.grad_every == 0:
                scaler.unscale_(optimizer)          # true (unscaled) gradients
                gsum += np.array(grad_norms(convs)); gcnt += 1
            scaler.step(optimizer)
            scaler.update()
            if not warmup:
                scheduler.step()
            global_iter += 1
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)

        tr_loss, tr_err = loss_sum / n, 100.0 * (1 - correct / n)
        if warmup and tr_err < 80.0:               # end warm-up (original code never switched back)
            warmup = False
            for pg in optimizer.param_groups:
                pg['lr'] = scheduler.get_last_lr()[0]
        te_loss, te_err = evaluate(model, test_loader, criterion, device)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, global_iter, f'{lr_now:.5g}', f'{tr_loss:.5f}', f'{tr_err:.3f}',
                                    f'{te_loss:.5f}', f'{te_err:.3f}', f'{dt:.1f}'])
        with open(grad_path, 'a', newline='') as f:
            w = csv.writer(f)
            for i, ((name, _), v) in enumerate(zip(convs, gsum / max(gcnt, 1))):
                w.writerow([epoch, i, name, f'{v:.6g}'])
        atomic_save({'epoch': epoch, 'global_iter': global_iter, 'warmup': warmup,
                     'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                     'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                             'torch': torch.get_rng_state(),
                             'cuda': torch.cuda.get_rng_state_all() if device == 'cuda' else None,
                             'loader': train_loader.generator.get_state()}}, ckpt_path)
        eta = dt * (n_epochs - epoch) / 60
        print(f'ep {epoch:3d}/{n_epochs} it {global_iter:5d} lr {lr_now:.4f} | train {tr_err:6.2f}% '
              f'| test {te_err:6.2f}% | {dt:5.1f}s/ep | ETA {eta:5.1f} min', flush=True)

    torch.save({'state_dict': model.state_dict(), 'final_test_err': te_err}, os.path.join(run_dir, 'final.th'))
    open(os.path.join(run_dir, 'DONE'), 'w').write(f'{te_err:.3f}\n')
    print(f'[{run_name(args)}] done. final test error {te_err:.2f}%')


if __name__ == '__main__':
    main()
