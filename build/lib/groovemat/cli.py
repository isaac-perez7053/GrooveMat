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
from pymatgen.core import Structure
import click

from groovemat.data import CIFData
from groovemat.data import collate_pool, get_train_val_test_loader
from groovemat.model import ConvergenceRegressor
from groovemat.matgl_loss import MatGLLoss
from groovemat.utils.normalizer import Normalizer

BANNER = r"""
+----------------------------------------------------------+
|      ____                          __  __       _        |
|     / ___|_ __ ___   _____   _____|  \/  | __ _| |_      |
|    | |  _| '__/ _ \ / _ \ \ / / _ \ |\/| |/ _` | __|     |
|    | |_| | | | (_) | (_) \ V /  __/ |  | | (_| | |_      |
|     \____|_|  \___/ \___/ \_/ \___|_|  |_|\__,_|\__|     |
|                                                          |
|                Get those atoms moving!                   |
+----------------------------------------------------------+
"""
args = None


class CustomGroup(click.Group):
    def get_help(self, ctx):
        base_help = super().get_help(ctx)
        return BANNER + "\n\n" + base_help


@click.group(cls=CustomGroup, invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """
    GrooveMat command‑line interface.
    """
    if ctx.invoked_subcommand is None:
        click.echo(cli.get_help(ctx))


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.argument("input_cif", type=click.Path(exists=True))
@click.argument("output_cif", type=click.Path())
@click.option("--cuda/--no-cuda", default=False, help="Enable CUDA if available")
def predict(checkpoint, input_cif, output_cif, cuda):
    """
    Take a trained model at CHECKPOINT, relax INPUT_CIF,
    and write the relaxed structure to OUTPUT_CIF.
    """
    ckpt = torch.load(checkpoint, map_location="cpu")
    saved_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    model = ConvergenceRegressor(
        atom_fea_len=saved_args.atom_fea_len,
        n_conv=saved_args.n_conv,
        h_fea_len=saved_args.h_fea_len,
        n_h=saved_args.n_h,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    normalizer = Normalizer()

    struct = Structure.from_file(input_cif)

    ds = CIFData(root_dir=None, file_list=[input_cif], perturb_std=0.0)
    loader = get_train_val_test_loader(
        dataset=ds,
        batch_size=1,
        train_ratio=None,
        val_ratio=None,
        test_ratio=None,
        return_test=True,
        num_workers=0,
    )[
        -1
    ]  # test_loader

    with torch.no_grad():
        for inputs, _dr_true, batch_struct in loader:
            atom_fea, nbr_fea, nbr_idx, idx_map = inputs
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_idx = nbr_idx.to(device)

            pred_dr_n = model(atom_fea, nbr_fea, nbr_idx)
            pred_dr_n = normalizer.denorm(pred_dr_n)
            pred_dr_n = pred_dr_n.cpu().numpy().reshape(-1, 3)

    orig_coords = np.array(struct.cart_coords)
    new_coords = orig_coords + pred_dr_n

    relaxed = Structure(
        lattice=struct.lattice,
        species=struct.species,
        coords=new_coords,
        coords_are_cartesian=True,
    )

    relaxed.to(filename=output_cif, fmt="cif")
    click.echo(f"Wrote relaxed structure ➔ {output_cif}")


@cli.command(context_settings={"ignore_unknown_options": True})
@click.argument("data_options", nargs=-1, type=click.STRING)
@click.option("--disable-cuda", is_flag=True, help="Disable CUDA")
@click.option(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 0)",
)
@click.option(
    "--epochs",
    default=30,
    type=int,
    metavar="N",
    help="number of total epochs to run (default: 30)",
)
@click.option(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
@click.option(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini‑batch size (default: 256)",
)
@click.option(
    "--lr",
    "--learning-rate",
    "lr",
    default=0.01,
    type=float,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
@click.option(
    "--lr-milestones",
    default=[100],
    multiple=True,
    type=int,
    metavar="N",
    help="milestones for scheduler (default: [100])",
)
@click.option("--momentum", default=0.9, type=float, metavar="M", help="momentum")
@click.option(
    "--weight-decay",
    "--wd",
    "weight_decay",
    default=0.0,
    type=float,
    metavar="W",
    help="weight decay (default: 0)",
)
@click.option(
    "-p",
    "--print-freq",
    "print_freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
@click.option(
    "--resume",
    default="",
    type=click.Path(),
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
@click.option(
    "--train-ratio",
    type=float,
    default=None,
    help="number of training data to be loaded (mutually exclusive with --train-size)",
)
@click.option(
    "--train-size",
    type=int,
    default=None,
    help="number of training data to be loaded (mutually exclusive with --train-ratio)",
)
@click.option(
    "--val-ratio",
    type=float,
    default=0.1,
    help="percentage of validation data to be loaded (default: 0.1)",
)
@click.option(
    "--val-size",
    type=int,
    default=None,
    help="number of validation data to be loaded (mutually exclusive with --val-ratio)",
)
@click.option(
    "--test-ratio",
    type=float,
    default=0.1,
    help="percentage of test data to be loaded (default: 0.1)",
)
@click.option(
    "--test-size",
    type=int,
    default=None,
    help="number of test data to be loaded (mutually exclusive with --test-ratio)",
)
@click.option(
    "--optim",
    default="SGD",
    type=click.Choice(["SGD", "Adam"]),
    metavar="SGD",
    help="choose an optimizer, SGD or Adam (default: SGD)",
)
@click.option(
    "--atom-fea-len",
    default=64,
    type=int,
    metavar="N",
    help="number of hidden atom features in conv layers",
)
@click.option(
    "--h-fea-len",
    default=128,
    type=int,
    metavar="N",
    help="number of hidden features after pooling",
)
@click.option(
    "--n-conv", default=3, type=int, metavar="N", help="number of conv layers"
)
@click.option(
    "--n-h",
    default=1,
    type=int,
    metavar="N",
    help="number of hidden layers after pooling",
)
@click.option(
    "--perturb-std",
    default=0.01,
    type=float,
    help="standard deviation for random displacement (Å)",
)
def train(
    data_options,
    disable_cuda,
    workers,
    epochs,
    start_epoch,
    batch_size,
    lr,
    lr_milestones,
    momentum,
    weight_decay,
    print_freq,
    resume,
    train_ratio,
    train_size,
    val_ratio,
    val_size,
    test_ratio,
    test_size,
    optim,
    atom_fea_len,
    h_fea_len,
    n_conv,
    n_h,
    perturb_std,
):
    """
    Train the GrooveMat model
    """

    class Args:
        pass

    global args
    args = Args()
    args.data_options = data_options
    args.disable_cuda = disable_cuda
    args.workers = workers
    args.epochs = epochs
    args.start_epoch = start_epoch
    args.batch_size = batch_size
    args.lr = lr
    args.lr_milestones = list(lr_milestones)
    args.momentum = momentum
    args.weight_decay = weight_decay
    args.print_freq = print_freq
    args.resume = resume
    args.train_ratio = train_ratio
    args.train_size = train_size
    args.val_ratio = val_ratio
    args.val_size = val_size
    args.test_ratio = test_ratio
    args.test_size = test_size
    args.optim = optim
    args.atom_fea_len = atom_fea_len
    args.h_fea_len = h_fea_len
    args.n_conv = n_conv
    args.n_h = n_h
    args.perturb_std = perturb_std

    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    # Load data
    dataset = CIFData(*args.data_options)
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        return_test=True
    )

    if len(dataset) < 500:
        warnings.warn(
            "Dataset has less than 500 data points. " "Lower accuracy is expected. "
        )
        sample_data_list = [dataset[i] for i in range(len(dataset))]

    else:
        sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]

    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

    # Build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    # after you compute orig_atom_fea_len, nbr_fea_len:
    device = torch.device("cuda" if args.cuda else "cpu")

    model = ConvergenceRegressor(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
    ).to(device)

    if args.cuda:
        model.cuda()

    # Grab the loss function
    loss_fn = MatGLLoss(model_name="M3GNet-MP-2021.2.8-PES")
    criterion = nn.MSELoss()

    # Choose optimizer
    if args.optim == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )

    else:
        raise NameError("Only SGD or Adam is allowed as --optim")

    # Optionally resume from checkpoint
    best_m3g_error = 1000
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_m3g_error = checkpoint["best_m3g_error"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            normalizer.load_state_dict(checkpoint["normalizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Initialize scheduler, in charge of updating the learning rate
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    train_losses = []
    val_losses = []
    train_maes = []
    val_mae_losses = []

    for epoch in range(args.start_epoch, args.epochs):

        # Train for one epoch
        train_loss, train_mae = train_helper(
            normalizer,  # your Normalizer
            train_loader,  # the DataLoader
            model,  # your model (already on device)
            loss_fn,  # the MatGLLoss instance
            criterion,  # the pointwise loss (MSELoss)
            optimizer,  # the optimizer
            device,  # the torch.device
            epoch,  # current epoch index
        )

        # evaluate on validation set
        m3g_error, mae_val_error = validate(
            val_loader, model, criterion, normalizer, device
        )

        train_losses.append(train_loss)
        train_maes.append(train_mae)
        val_losses.append(m3g_error)
        val_mae_losses.append(mae_val_error)

        if m3g_error != m3g_error:
            print("Exit due to NaN")
            sys.exit(1)

        scheduler.step()

        # Remember the best m3g_eror and save checkpoint
        best_m3g_error = min(m3g_error, best_m3g_error)
        is_best = m3g_error < best_m3g_error
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_m3g_error": best_m3g_error,
                "optimizer": optimizer.state_dict(),
                "normalizer": normalizer.state_dict(),
                "args": vars(args),
            },
            is_best,
        )

    print("---------Evaluate Model on Test Set---------------")
    best_checkpoint = torch.load("model_best.pth.tar")
    model.load_state_dict(best_checkpoint["state_dict"])
    validate(test_loader, model, criterion, normalizer, device, test=True)
    # Test best model
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_maes, label="Train MAE (Å)")
    plt.plot(range(1, args.epochs + 1), val_mae_losses, label="Val MAE   (Å)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (Å)")
    plt.title("Training & Validation MAE")
    plt.legend()
    plt.grid(True)
    plt.savefig("mae_vs_epoch.png", dpi=300)


def train_helper(
    normalizer: Normalizer,
    train_loader,
    model: nn.Module,
    loss_fn: MatGLLoss,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
):
    model.train()

    train_meter = AverageMeter()
    mae_meter = AverageMeter()

    for i, (inputs, dr_true, batch_struct) in enumerate(train_loader):
        atom_fea, nbr_fea, nbr_idx, crystal_atom_idx = inputs
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_idx = nbr_idx.to(device)

        # forward over the full batch → (N_total, 3)
        pred_dr_n = model(atom_fea, nbr_fea, nbr_idx)

        # split the true Δr into per‑crystal chunks
        sizes = [len(idx_map) for idx_map in crystal_atom_idx]
        dr_list = dr_true.split(sizes, dim=0)  # list of (n_i,3)

        # accumulate MatGL loss per crystal
        total_loss = 0.0
        for idx_map, struct_relaxed, dr_i in zip(
            crystal_atom_idx, batch_struct, dr_list
        ):
            # extract this crystal’s atom predictions
            pred_i = pred_dr_n[idx_map]  # (n_i, 3)
            x_flat = pred_i.view(-1)  # (3*n_i,)

            # reconstruct the initial (perturbed) coords:
            #   r_initial = r_relaxed - Δr_true
            coords_relaxed = np.array(struct_relaxed.cart_coords)
            delta_np = dr_i.cpu().numpy()  # (n_i,3)
            coords_initial = coords_relaxed - delta_np

            # build a new Structure at the perturbed geometry
            struct_initial = Structure(
                struct_relaxed.lattice,
                struct_relaxed.species,
                coords_initial,
                coords_are_cartesian=True,
            )

            # use the perturbed Structure for the loss
            total_loss += loss_fn(x_flat, struct_initial, classifier=criterion)

        # average over the batch
        loss = total_loss / len(batch_struct)

        train_meter.update(loss.item(), len(batch_struct))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            # denormalize the whole‑batch predictions
            pred_all = normalizer.denorm(pred_dr_n)  # (N_total, 3)
            dr_all = dr_true  # (N_total, 3)

            # split into per‑crystal chunks
            sizes = [len(idx_map) for idx_map in crystal_atom_idx]
            pred_list = pred_all.split(sizes, dim=0)  # list of (n_i,3)
            dr_list = dr_all.split(sizes, dim=0)  # list of (n_i,3)

            # compute each crystal’s ⟨|change in r_pred – change in r_true|⟩
            errs = [(p - d).abs().mean() for p, d in zip(pred_list, dr_list)]
            mean_err = torch.stack(errs).mean().item()
            batch_mae = (pred_all - dr_true).abs().mean().item()

        mae_meter.update(batch_mae, dr_true.size(0))

        if i % args.print_freq == 0:
            print(
                f"Epoch {epoch} | Iter {i}: "
                f"train loss={loss.item():.4f}, "
                f"⟨|Δr|⟩={mean_err:.3f} Å"
            )

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
            nbr_fea = nbr_fea.to(device)
            nbr_idx = nbr_idx.to(device)
            dr_true = dr_true.to(device)

            # normalize target displacement
            dr_true_n = normalizer.norm(dr_true)

            # forward + MSE loss
            pred_n = model(atom_fea, nbr_fea, nbr_idx)
            loss = criterion(pred_n, dr_true_n)

            # record MSE loss
            m3g_errors.update(loss.item(), dr_true.size(0))

            # denormalize for MAE
            pred = normalizer.denorm(pred_n)
            m3g = (pred - dr_true).abs().mean().item()
            # record MAE per atom, not MatGLLoss instance
            mae_errors.update(m3g, dr_true.size(0))

            # if in test mode, stash for CSV
            if test:
                test_preds += pred.view(-1).cpu().tolist()
                test_targets += dr_true.view(-1).cpu().tolist()
                test_cif_ids += batch_cif_ids

            # timing
            batch_time.update(time.time() - start)
            start = time.time()

            # progress print
            if i % args.print_freq == 0:
                print(
                    f"Val: [{i}/{len(val_loader)}]  "
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    f"Loss {m3g_errors.val:.4f} ({m3g_errors.avg:.4f})  "
                    f"M3g Error {m3g_errors.val:.3f} ({m3g_errors.avg:.3f})"
                )

    # if test, dump CSV
    if test:
        import csv

        with open("test_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cif_id", "true_Δr", "pred_Δr"])
            for cid, t, p in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cid, t, p))

    # Final Summary
    star = "**" if test else "*"
    print(f" {star} M3g Error Avg {m3g_errors.avg:.3f}")
    print(f"{star}  Val MAE Avg = {mae_errors.avg:.4f} Å")
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
            target_label, pred_label, average="binary"
        )
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


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    main()
