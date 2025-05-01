import torch; import torch.nn as nn 
from m3gnet.models import Potential, M3GNet
from pymatgen.core import Structure

class M3gnetLoss(nn.Module):
    """
    Uses the m3gnet nn to calculate the loss associated with particular displacements
    """ 
    def __init__(self):
        super().__init__()
        self.pot = Potential(M3GNet.load())

    def predict_force_energy(self, x_flat: torch.Tensor, structure: Structure):
        """
        Parameters
        ----------
        structure: Structure
          Pymatgen structure used to predict energy
        
        Returns
        -------
        energy: torch.Tensor
          Returns the energy represented as a torch.Tensor
        """
        E, F = self.pot.get_ef(structure)
        E_np  = E.numpy()
        F_np = F.numpy()
        return  torch.as_tensor(F_np, dtype=torch.float32, device=x_flat.device), \
                torch.as_tensor(E_np, dtype=torch.float32, device=x_flat.device)
    
    def forward(self, input: torch.Tensor, target: Structure, classifier) -> torch.Tensor:
        # 1) build a flat “reference” coordinate vector x0 with grad enabled
        pos0 = torch.tensor(
            target.cart_coords,
            dtype=torch.float32,
            device=input.device,
            requires_grad=True,
        )                   # (N,3)
        x0 = pos0.view(-1)  # (3*N,)

        # 2) get forces & energy at x0
        f0, e0 = self.predict_force_energy(x0, target)
        #    — predict_force_energy returns (forces, energy)

        # 3) define an energy‐only function for Hessian
        energy_fn = lambda y: self.predict_force_energy(y, target)[1]

        # 4) compute Hessian d²E/dy² at x0
        H = torch.autograd.functional.hessian(energy_fn, x0)

        # 5) invert (with a tiny ridge for stability)
        ridge = 1e-6 * torch.eye(H.size(0), device=H.device)
        H_inv = torch.linalg.inv(H + ridge)

        # 6) actual displacement by Δx = −H⁻¹ ⋅ f
        delta_x = -H_inv.matmul(f0)     # (3*N,)

        # 7) reshape to match whatever shape `input` has
        actual_disp = delta_x.view_as(input)

        print("Step norm:", actual_disp.norm().item())

        # 8) return your loss between predicted vs. “actual”
        return classifier(input, actual_disp)




# from ase.phonons import Phonons
# from pymatgen.io.ase import AseAtomsAdaptor

# from m3gnet.models import M3GNet, M3GNetCalculator, Potential

# mp_129 = mpr.get_structure_by_material_id(mp_id := "mp-129")

# # Setup crystal and EMT calculator
# atoms = AseAtomsAdaptor().get_atoms(mp_129)

# potential = Potential(M3GNet.load())

# calculator = M3GNetCalculator(potential=potential, stress_weight=0.01)


# usage
# criterion = M3gnetClassifier()
# loss = criterion(predict_r, structure, classifier) 