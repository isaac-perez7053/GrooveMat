import warnings
import torch
from m3gnet.models import Relaxer, Potential, M3GNet
from pymatgen.core import Lattice, Structure

for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="tensorflow")

# Init a Mo structure with stretched lattice (DFT lattice constant ~ 3.168)
mo = Structure(Lattice.cubic(3.3), ["Mo", "Mo"], [[0., 0., 0.], [0.5, 0.5, 0.5]])
model = M3GNet.load()
pot = Potential(model)

relaxer = Relaxer()  # This loads the default pre-trained model

relax_results = relaxer.relax(mo, verbose=True)

final_structure = relax_results['final_structure']
final_energy_per_atom = float(relax_results['trajectory'].energies[-1] / len(mo))

print(f"Relaxed lattice parameter is {final_structure.lattice.abc[0]:.3f} Å")
print(f"Final energy is {final_energy_per_atom:.3f} eV/atom")

pos0 = torch.tensor(mo.cart_coords, dtype=torch.float32, requires_grad=True) # (N, 3)
x0 = pos0.view(-1) 
E, _ = pot.get_ef(mo)
E_np = E.numpy()
print(type(E))
energy_t = torch.tensor(E_np, dtype=torch.float32)
print(torch.as_tensor(energy_t, dtype=torch.float32, device=x0.device))

