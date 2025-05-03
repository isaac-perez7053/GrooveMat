import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.optim as optim
from sklearn import metrics
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import matplotlib.pyplot as plt

from disco.data import CIFData
from disco.data import collate_pool, get_train_val_test_loader
from disco.model import ConvergenceRegressor
from disco.matgl_loss import MatGLLoss
from disco.utils.normalizer import Normalizer

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')

parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')

parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')

parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')

parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

train_group = parser.add_mutually_exclusive_group()

train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')

train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')

valid_group = parser.add_mutually_exclusive_group()

valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')

valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')

test_group = parser.add_mutually_exclusive_group()

test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')

test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')

parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')

parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')

parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')

parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

parser.add_argument('--perturb-std',  type=float, default=0.01,
                    help='standard deviation for random displacement (Å)')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

best_m3g_error = 1e10

def main():
    global args, best_m3g_error

    # Load data
    dataset = CIFData(*args.data_options)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)

    if len(dataset) < 500:
        warnings.warn('Dataset has less than 500 data points. '
                        'Lower accuracy is expected. ')
        sample_data_list = [dataset[i] for i in range(len(dataset))]

    else:
        sample_data_list = [dataset[i] for i in
                            sample(range(len(dataset)), 500)]
        
    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

    # Build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    
    # after you compute orig_atom_fea_len, nbr_fea_len:
    device = torch.device("cuda" if args.cuda else "cpu")

    model = ConvergenceRegressor(
        orig_atom_fea_len, nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h
    ).to(device)

    if args.cuda:
        model.cuda()

    # Grab the loss function
    loss_fn = MatGLLoss(model_name="M3GNet-MP-2021.2.8-PES")
    criterion = nn.MSELoss()

    # Choose optimizer
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
        
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)

    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # Optionally resume from checkpoint 
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_m3g_error = checkpoint['best_m3g_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Initialize scheduler, in charge of updating the learning rate
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)
    
    train_losses = []
    val_losses = []
    train_maes = []
    val_mae_losses = []

    for epoch in range(args.start_epoch, args.epochs):

        # Train for one epoch
        train_loss, train_mae = train(
            normalizer,    # your Normalizer
            train_loader,  # the DataLoader
            model,         # your model (already on device)
            loss_fn,       # the MatGLLoss instance
            criterion,     # the pointwise loss (MSELoss)
            optimizer,     # the optimizer
            device,        # the torch.device
            epoch          # current epoch index
        )

        # evaluate on validation set
        m3g_error, mae_val_error = validate(val_loader, model, 
                             criterion,
                             normalizer, 
                             device)
        
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        val_losses.append(m3g_error)
        val_mae_losses.append(mae_val_error)

        if m3g_error != m3g_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # Remember the best m3g_eror and save checkpoint
        is_best = m3g_error < best_m3g_error
        best_m3g_error = min(m3g_error, best_m3g_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_m3g_error': best_m3g_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)


    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer, device, test=True)
        # Test best model 
    plt.figure()
    plt.plot(range(1, args.epochs+1), train_maes,      label='Train MAE (Å)')
    plt.plot(range(1, args.epochs+1), val_mae_losses,  label='Val MAE   (Å)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (Å)')
    plt.title('Training & Validation MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig('mae_vs_epoch.png', dpi=300)


def train(normalizer: Normalizer,
          train_loader,
          model: nn.Module,
          loss_fn: MatGLLoss,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          epoch: int):
    model.train()

    train_meter = AverageMeter()
    mae_meter = AverageMeter()

    for i, (inputs, dr_true, batch_struct) in enumerate(train_loader):
        atom_fea, nbr_fea, nbr_idx, crystal_atom_idx = inputs
        atom_fea = atom_fea.to(device)
        nbr_fea  = nbr_fea.to(device)
        nbr_idx  = nbr_idx.to(device)

        # forward over the full batch → (N_total, 3)
        pred_dr_n = model(atom_fea, nbr_fea, nbr_idx)

        # accumulate MatGL loss per crystal
        total_loss = 0.0
        for idx_map, struct_relaxed in zip(crystal_atom_idx, batch_struct):
            # extract this crystal’s atom predictions
            pred_i = pred_dr_n[idx_map]            # (n_i, 3)
            x_flat = pred_i.view(-1)               # (3*n_i,)

            # use the perturbed Structure returned by collate_pool
            total_loss += loss_fn(x_flat, struct_relaxed, classifier=criterion)

        # average over the batch
        loss = total_loss / len(batch_struct)

        train_meter.update(loss.item(), len(batch_struct))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()




        with torch.no_grad():
            # denormalize the whole‐batch predictions
            pred_all = normalizer.denorm(pred_dr_n)   # (N_total, 3)
            dr_all   = dr_true                         # (N_total, 3)

            # split into per‐crystal chunks
            sizes    = [len(idx_map) for idx_map in crystal_atom_idx]
            pred_list = pred_all.split(sizes, dim=0)   # list of (n_i,3)
            dr_list   = dr_all.split(sizes, dim=0)     # list of (n_i,3)

            # compute each crystal’s ⟨|change in r_pred – change in r_true|⟩
            errs = [(p - d).abs().mean() for p, d in zip(pred_list, dr_list)]
            mean_err = torch.stack(errs).mean().item()
            batch_mae = (pred_all - dr_true).abs().mean().item()
        
        mae_meter.update(batch_mae, dr_true.size(0))


        if i % args.print_freq == 0:
            print(f"Epoch {epoch} | Iter {i}: "
                  f"train loss={loss.item():.4f}, "
                  f"⟨|Δr|⟩={mean_err:.3f} Å")
            
        return train_meter.avg, mae_meter.avg



def validate(val_loader, model, criterion, normalizer: Normalizer, device, test=False):
    batch_time = AverageMeter()
    m3g_errors = AverageMeter()
    mae_errors = AverageMeter()

    # If we're in a “test” run, collect predictions/ids for csv
    if test:
        test_preds = []
        test_targets = []
        test_cif_ids = []

    model.eval()
    start = time.time()

    with torch.no_grad():
        for i, (inputs, dr_true, batch_cif_ids) in enumerate(val_loader):
            # unpack your graph inputs
            atom_fea, nbr_fea, nbr_idx, _ = inputs

            # move everything onto device
            atom_fea = atom_fea.to(device)
            nbr_fea  = nbr_fea.to(device)
            nbr_idx  = nbr_idx.to(device)
            dr_true  = dr_true.to(device)

            # normalize target displacement
            dr_true_n = normalizer.norm(dr_true)

            # forward + MSE loss
            pred_n = model(atom_fea, nbr_fea, nbr_idx)
            loss   = criterion(pred_n, dr_true_n)

            # record MSE loss
            m3g_errors.update(loss.item(), dr_true.size(0))

            # denormalize for MAE
            pred = normalizer.denorm(pred_n)
            m3g  = (pred - dr_true).abs().mean().item()
            # record MAE per atom, not MatGLLoss instance
            mae_errors.update(m3g, dr_true.size(0))

            # if in test mode, stash for CSV
            if test:
                test_preds   += pred.view(-1).cpu().tolist()
                test_targets += dr_true.view(-1).cpu().tolist()
                test_cif_ids += batch_cif_ids

            # timing
            batch_time.update(time.time() - start)
            start = time.time()

            # progress print
            if i % args.print_freq == 0:
                print(f'Val: [{i}/{len(val_loader)}]  '
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      f'Loss {m3g_errors.val:.4f} ({m3g_errors.avg:.4f})  '
                      f'M3g Error {m3g_errors.val:.3f} ({m3g_errors.avg:.3f})')

    # if test, dump CSV
    if test:
        import csv
        with open('test_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cif_id', 'true_Δr', 'pred_Δr'])
            for cid, t, p in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cid, t, p))

    # Final Summary
    star = '**' if test else '*'
    print(f' {star} M3g Error Avg {m3g_errors.avg:.3f}')
    print(f'{star}  Val MAE Avg = {mae_errors.avg:.4f} Å')
    return m3g_errors.avg, mae_errors.avg




def class_eval(prediction, target: torch.Tensor):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == '__main__':
    main()



