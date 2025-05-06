#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from groovemat.data import CIFData, collate_pool
from groovemat.model import ConvergenceRegressor
from groovemat.utils.normalizer import Normalizer

def load_model(checkpoint_path, device, sample_item):
    # sample_item is ((atom_fea, nbr_fea, nbr_idx), dr_true, struct_relaxed)
    (atom_fea, nbr_fea, nbr_idx), _, _ = sample_item
    orig_atom_fea_len = atom_fea.shape[-1]
    nbr_fea_len = nbr_fea.shape[-1]
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt['args']
    model = ConvergenceRegressor(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=args['atom_fea_len'],
        n_conv=args['n_conv'],
        h_fea_len=args['h_fea_len'],
        n_h=args['n_h'],
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    _, sample_target, _ = sample_item
    normalizer = Normalizer(sample_target)
    normalizer.load_state_dict(ckpt['normalizer'])
    return model, normalizer


def main():
    parser = argparse.ArgumentParser(
        description="Test a trained Disco model on a folder of CIFs"
    )
    parser.add_argument("cif_folder",
                        help="Directory containing your test CIF files")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model_best.pth.tar")
    parser.add_argument("--perturb_std", type=float, default=0.005,
                        help="Std dev for initial perturbation (Å)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="How many crystals per batch")
    parser.add_argument("--output_csv", default="test_results.csv",
                        help="Where to write per-crystal errors")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a CIFData that will (randomly) perturb each input by perturb_std
    dataset = CIFData(
        root_dir=args.cif_folder,
        perturb_std=args.perturb_std
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pool,
        num_workers=0
    )

    # Grab a sample to infer atom_fea_len & nbr_fea_len
    sample_item = dataset[0]
    model, normalizer = load_model(args.checkpoint, device, sample_item)

    # Collect all per-atom errors
    all_atom_errors = []

    # Open CSV and write header
    with open(args.output_csv, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "cif_id",
            "mean_error(Å)",
            "max_error(Å)"
        ])

        with torch.no_grad():
            for (inputs, dr_true, struct_relaxed) in loader:
                atom_fea, nbr_fea, nbr_idx, crystal_idx = inputs
                atom_fea = atom_fea.to(device)
                nbr_fea = nbr_fea.to(device)
                nbr_idx = nbr_idx.to(device)

                # predict per-atom displacement
                pred_n = model(atom_fea, nbr_fea, nbr_idx)  # (N_total,3)
                pred = normalizer.denorm(pred_n).cpu().numpy()
                true = dr_true.numpy()  # (N_total,3)

                # split into crystals & compute errors
                start = 0
                for struct, idx_map in zip(struct_relaxed, crystal_idx):
                    n_i = len(idx_map)
                    true_i = true[start:start + n_i]
                    pred_i = pred[start:start + n_i]
                    errors = np.linalg.norm(pred_i - true_i, axis=1)
                    all_atom_errors.extend(errors.tolist())
                    mean_e = errors.mean()
                    max_e = errors.max()
                    # get a human-readable ID
                    cif_id = getattr(struct, "filename",
                                     getattr(struct, "formula", "unknown"))
                    writer.writerow([cif_id, f"{mean_e:.4f}", f"{max_e:.4f}"])
                    start += n_i

    print(f"Done! Wrote per-crystal errors to {args.output_csv}")

    # Plot histogram of all per-atom errors
    errors = np.array(all_atom_errors)
    plt.figure(figsize=(6, 4))
    # use a nicer binning (e.g. 20 bins or ‘auto’), add black edges & slight transparency
    n, bins, patches = plt.hist(
        errors,
        bins=20,
        edgecolor='black',
        alpha=0.7
    )

    # labels & title with a bit more breathing room
    plt.xlabel('Per‑atom error (Å)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Histogram of per‑atom prediction errors', fontsize=14)

    # horizontal grid lines on y‑axis only
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # tighten margins so nothing gets cut off
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
    