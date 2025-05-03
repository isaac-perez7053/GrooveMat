import torch
import torch.nn as nn
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
from matgl.ext.ase import PESCalculator

class MatGLLoss(nn.Module):
    """
    Uses MatGL’s pre‑trained PES model to compute forces & energy for a displaced cell.
    """
    def __init__(self, model_name: str = "M3GNet-MP-2021.2.8-PES"):
        super().__init__()
        # load the MatGL PES model
        self.pot = matgl.load_model(model_name)
        # adaptor to convert pymatgen to ASE
        self.adaptor = AseAtomsAdaptor()

    def predict_force_energy(self, x_flat: torch.Tensor, structure: Structure):
        """
        Predict forces & energy on a displaced unit cell using MatGL’s PESCalculator.

        Parameters
        ----------
        x_flat: torch.Tensor
          Flattened displacement vector of length 3*N
        structure: Structure
          The reference (relaxed) pymatgen structure

        Returns
        -------
        F_torch: torch.Tensor, shape (3*N,)
        E_torch: torch.Tensor, scalar
        """
        # reshape & detach
        disp = x_flat.detach().cpu().numpy().reshape(-1, 3)  # (N,3)
        new_coords = structure.cart_coords + disp

        # build shifted pymatgen structure
        shifted = Structure(
            structure.lattice,
            structure.species,
            new_coords,
            coords_are_cartesian=True
        )

        # convert to ASE Atoms and attach MatGL PESCalculator
        atoms = self.adaptor.get_atoms(shifted)
        atoms.set_calculator(PESCalculator(potential=self.pot))

        # single‑point evaluation
        E = atoms.get_potential_energy()  # eV
        F = atoms.get_forces().reshape(-1) # flattened to (3*N,)

        # back to torch
        device = x_flat.device
        E_torch = torch.tensor(E, dtype=torch.float32, device=device)
        F_torch = torch.tensor(F, dtype=torch.float32, device=device)

        return F_torch, E_torch

    def forward(self, input: torch.Tensor, target: Structure, classifier) -> torch.Tensor:
        """
        Compute loss between predicted displacements and the “true” displacement

        Parameters
        ----------
        input: torch.Tensor, shape (3*N,)
          The predicted displacement vector
        target: Structure
          The relaxed pymatgen structure
        classifier: Callable
          A loss function, e.g. nn.MSELoss()

        Returns
        -------
        loss: torch.Tensor
        """
        # build reference coords with grad
        pos0 = torch.tensor(
            target.cart_coords,
            dtype=torch.float32,
            device=input.device,
            requires_grad=True
        ).view(-1)  # (3*N,)

        # get forces & energy at pos0
        f0, e0 = self.predict_force_energy(pos0, target)

        # energy-only function for Hessian
        energy_fn = lambda y: self.predict_force_energy(y, target)[1]

        # Hessian at reference
        H: torch.Tensor = torch.autograd.functional.hessian(energy_fn, pos0)

        # invert with ridge
        ridge = 1e-6 * torch.eye(H.size(0), device=H.device)
        H_inv: torch.Tensor = torch.linalg.inv(H + ridge)

        # r = −H^{-1} * f
        delta_x = -H_inv.matmul(f0)
        actual_disp = delta_x.view_as(input)

        print("Step norm:", actual_disp.norm().item())
        return classifier(input, actual_disp)


# usage
# criterion = MatGLLoss()
# loss = criterion(predict_r, structure) 