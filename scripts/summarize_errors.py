import pandas as pd

# Load the per‑atom errors
df = pd.read_csv('test_atom_errors.csv')

# Overall error‐mag summary
print("Overall error_mag statistics:")
print(df['error_mag'].describe(), "\n")

# And if you want the median & 90th percentile explicitly:
print(f"Median error: {df['error_mag'].median():.4f} Å")
print(f"90th percentile error: {df['error_mag'].quantile(0.9):.4f} Å\n")

# You can also see which atom was worst:
worst = df.loc[df['error_mag'].idxmax()]
print("Worst atom:")
print(f"  cif_id    : {worst['cif_id']}")
print(f"  atom_idx  : {int(worst['atom_index'])}")
print(f"  true Δr   : ({worst['true_dx']:.4f}, {worst['true_dy']:.4f}, {worst['true_dz']:.4f})")
print(f"  pred Δr   : ({worst['pred_dx']:.4f}, {worst['pred_dy']:.4f}, {worst['pred_dz']:.4f})")
print(f"  error_mag : {worst['error_mag']:.4f} Å")
