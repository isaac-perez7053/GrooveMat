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

from groovemat.data import CIFData
from groovemat.data import collate_pool, get_train_val_test_loader
from groovemat.model import ConvergenceRegressor
from groovemat.matgl_loss import MatGLLoss

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

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

best_m3g_error = 1e10

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


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
    model = ConvergenceRegressor(
        orig_atom_fea_len, nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h)

    if args.cuda:
        model.cuda()

    # Grab the loss function
    criterion = MatGLLoss()

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

    for epoch in range(args.start_epoch, args.epochs):

        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer) 

        # evaluate on validation set
        m3g_error = validate(val_loader, model, criterion, normalizer)

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

    # Test best model 
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer, test=True)


def train(normalizer: Normalizer, train_loader, model, criterion: MatGLLoss, optimizer: torch.optim, epoch: int):
    model.train()
    for i, (inputs, dr_true) in enumerate(train_loader):
        atom_fea , nbr_fea, nbr_idx, _ = inputs
        atom_fea: torch.Tensor = atom_fea

        # move to device
        # atom_fea = atom_fea.to(device)
        # nbr_fea  = nbr_fea.to(device)
        # nbr_idx  = nbr_idx.to(device)
        # dr_true  = dr_true.to(device)

        # normalize the ground-truth displacements
        dr_true_n = normalizer.norm(dr_true)

        # forward + loss (in normalized space)
        pred_dr_n = model(atom_fea, nbr_fea, nbr_idx)

        # Use the mat_gl loss function
        loss = criterion(pred_dr_n, dr_true_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for logging, denormalize and measure error in angstroms
        with torch.no_grad():
            pred_dr = normalizer.denorm(pred_dr_n)
            mean_err = (pred_dr - dr_true).abs().mean().item()

        if i % args.print_freq == 0:
            print(f"Epoch {epoch} | Iter {i}: "
                f"train loss={loss.item():.4f}, "
                f"⟨|Δr|⟩={mean_err:.3f} Å")


def validate(val_loader, model, criterion, normalizer: Normalizer, device, test=False):
    batch_time = AverageMeter()
    losses     = AverageMeter()
    m3g_errors = AverageMeter()

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

            # forward + loss
            pred_n = model(atom_fea, nbr_fea, nbr_idx)
            loss   = criterion(pred_n, dr_true_n)

            # record loss
            losses.update(loss.item(), dr_true.size(0))

            # denormalize for m3g loss
            pred = normalizer.denorm(pred_n)
            m3g  = (pred - dr_true).abs().mean().item()
            m3g_errors.update(MatGLLoss(), dr_true.size(0)) # TODO: Hope this works

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
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
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
    return m3g_errors.avg



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