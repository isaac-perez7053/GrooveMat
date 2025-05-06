#!/usr/bin/env python
import argparse
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from groovemat.data import CIFData, collate_pool
from groovemat.model import ConvergenceRegressor
from groovemat.utils.normalizer import Normalizer


def load_model(checkpoint_path, device, sample_item):
    # sample_item is ((atom_fea, nbr_fea, nbr_idx), dr_true, struct_relaxed)
    (atom_fea, nbr_fea, nbr_idx), _, _ = sample_item
    orig_atom_fea_len = atom_fea.shape[-1]
    nbr_fea_len = nbr_fea.shape[-1]
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]
    model = ConvergenceRegressor(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=args["atom_fea_len"],
        n_conv=args["n_conv"],
        h_fea_len=args["h_fea_len"],
        n_h=args["n_h"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    _, sample_target, _ = sample_item
    normalizer = Normalizer(sample_target)
    normalizer.load_state_dict(ckpt["normalizer"])
    return model, normalizer


def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu"
    )

    # build dataset & loader
    dataset = CIFData(
        root_dir=args.data_dir,
        max_num_nbr=args.max_num_nbr,
        radius=args.radius,
        dmin=args.dmin,
        step=args.step,
        random_seed=args.random_seed,
        perturb_std=args.perturb_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_pool,
        shuffle=False,
        num_workers=0,
    )

    # load model from checkpoint
    sample_item = dataset[0]
    model, normalizer = load_model(args.checkpoint, device, sample_item)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cif_id",
                "atom_index",
                "true_dx",
                "true_dy",
                "true_dz",
                "pred_dx",
                "pred_dy",
                "pred_dz",
                "error_mag",
            ]
        )

        with torch.no_grad():
            for (
                (atom_fea, nbr_fea, nbr_idx, crystal_atom_idx),
                dr_true,
                struct_relaxed,
            ) in loader:

                atom_fea = atom_fea.to(device)
                nbr_fea = nbr_fea.to(device)
                nbr_idx = nbr_idx.to(device)

                # predict per-atom displacement
                pred = model(atom_fea, nbr_fea, nbr_idx)  # (N_total, 3)
                pred = normalizer.denorm(pred).cpu().numpy()
                true = dr_true.cpu().numpy()  # (N_total, 3)

                # split predictions and truths per crystal
                start = 0
                for struct, idx_map in zip(struct_relaxed, crystal_atom_idx):
                    n_i = len(idx_map)
                    true_i = true[start : start + n_i]
                    pred_i = pred[start : start + n_i]
                    errors = np.linalg.norm(pred_i - true_i, axis=1)
                    # use formula or filename as ID
                    try:
                        cif_id = os.path.basename(struct.filename)
                    except:
                        cif_id = struct.formula

                    for atom_j in range(n_i):
                        t0, t1, t2 = true_i[atom_j]
                        p0, p1, p2 = pred_i[atom_j]
                        e = errors[atom_j]
                        writer.writerow([cif_id, atom_j, t0, t1, t2, p0, p1, p2, e])
                    start += n_i


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-atom error evaluator")
    parser.add_argument("data_dir", help="Path to directory of CIF files")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model_best.pth.tar checkpoint"
    )
    parser.add_argument(
        "--output-csv",
        default="test_atom_errors.csv",
        help="Where to write per-atom errors",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--perturb-std", type=float, default=0.01)
    parser.add_argument("--max-num-nbr", type=int, default=12)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--dmin", type=float, default=0.0)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--disable-cuda", action="store_true")
    args = parser.parse_args()
    evaluate(args)

# Use:
# python evaluate.py data/structures \
#     --checkpoint model_best.pth.tar \
#     --output-csv test_atom_errors.csv \
#     --batch-size 16 \
#     --perturb-std 0.02
