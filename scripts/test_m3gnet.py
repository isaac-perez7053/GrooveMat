import warnings
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
from matgl.ext.ase import Relaxer, PESCalculator

# suppress TF warnings for clarity
for cat in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=cat, module="tensorflow")

# 1) Build your Mo structure (stretched from DFT a≈3.168 → we use 3.3 Å)
mo = Structure(
    Lattice.cubic(3.3),
    ["Mo", "Mo"],
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

# 2) Load the MatGL potential you want.
#    Here we use the universal M3GNet interatomic potential.
#    You can list all available PES models via:
#      matgl.get_available_pretrained_models()
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")  # :contentReference[oaicite:0]{index=0}

# 3) Relaxation via ASE interface
relaxer = Relaxer(potential=pot)
relax_results = relaxer.relax(mo, fmax=0.01, verbose=True)  # :contentReference[oaicite:1]{index=1}

final_struct = relax_results["final_structure"]
E_traj = relax_results["trajectory"].energies  # list of total energies along the path
E_relaxed = E_traj[-1] / len(final_struct)     # per‑atom

print(f"Relaxed lattice parameter: {final_struct.lattice.abc[0]:.3f} Å")
print(f"Final energy: {E_relaxed:.3f} eV/atom")

# 4) Single-point energy + forces (via ASE → torch)
#    Convert pymatgen Structure → ASE Atoms
ase_adaptor = AseAtomsAdaptor()
atoms = ase_adaptor.get_atoms(mo)

#    Attach the MatGL PESCalculator
atoms.set_calculator(PESCalculator(potential=pot))

#    Query ASE
E_ase = atoms.get_potential_energy()  # total energy, eV
F_ase = atoms.get_forces()            # (N,3) array, eV/Å

print(f"ASE single‑point energy: {E_ase:.3f} eV")
print(f"ASE forces shape: {F_ase.shape}")

#    If you really need Torch tensors (e.g. to backprop through them):
pos0 = torch.tensor(mo.cart_coords, dtype=torch.float32, requires_grad=True).view(-1)
energy_t = torch.tensor(E_ase, dtype=torch.float32, device=pos0.device)
forces_t = torch.tensor(F_ase, dtype=torch.float32, device=pos0.device)

print("Torch energy tensor:", energy_t)
print("Torch forces tensor:", forces_t.shape)


