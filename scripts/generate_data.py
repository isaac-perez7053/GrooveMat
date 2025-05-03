import os
import csv
import argparse
from mp_api.client import MPRester

parser = argparse.ArgumentParser(description="Generate dataset from Materials Project")
parser.add_argument("--n", type=int, help="Number of structures to fetch")
args = parser.parse_args()

# Configuration
API_KEY = "jqiPpuSEWUeahM0ZyBKtX7jT7HYKxSEs" # Materials Project API key
OUTPUT_DIR = "data"
STRUCTURE_DIR = os.path.join(OUTPUT_DIR, "structures")
CSV_PATH = os.path.join(OUTPUT_DIR, "id_prop.csv")

if args.n is None:
    print("No limit set. Fetching all results...")

num_samples = args.n  # Limit the number of results to fetch

search_kwargs = dict(
    formula="ABO3",
    crystal_system="Cubic",
    fields=["material_id", "formula_pretty", "structure", "energy_per_atom"],
)

if num_samples:
    search_kwargs.update({
        "chunk_size": num_samples,
        "num_chunks": 1,
    })

# Make directories
os.makedirs(STRUCTURE_DIR, exist_ok=True)

# Fetch data
id_prop_rows = []

with MPRester(API_KEY) as mpr:
    results = mpr.materials.summary.search(**search_kwargs)

    for entry in results:
        try:
            mp_id = entry.material_id
            structure = entry.structure
            energy = entry.energy_per_atom

            # structure = mpr.get_structure_by_material_id(mp_id)
            # entry = mpr.get_entry_by_material_id(mp_id)
            # energy = entry.energy_per_atom

            cif_path = os.path.join(STRUCTURE_DIR, f"{mp_id}.cif")
            structure.to(fmt="cif", filename=cif_path)

            id_prop_rows.append((mp_id, energy))
            print(f"Saved {mp_id}.cif with energy {energy:.4f}")

        except Exception as e:
            print(f"Failed to fetch {mp_id}: {e}")

# Write CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "property"])
    writer.writerows(id_prop_rows)

print(f"\n Done! CIFs in: {STRUCTURE_DIR}")
print(f"CSV saved at: {CSV_PATH}")
