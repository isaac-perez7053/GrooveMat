import torch
import torch.nn as nn
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np

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

    def predict_force_energy(self, pos_flat: torch.Tensor, structure: Structure):
        """
        Given a flat tensor of displacements (shape 3N) and a pymatgen Structure,
        build an ASE Atoms at coords = structure.cart_coords + displacement,
        attach the PESCalculator, and return forces (flat 3N) and energy.
        """
        # 1) reshape pos_flat → (N, 3) numpy array
        disp = pos_flat.detach().cpu().numpy().reshape(-1, 3)

        # 2) get the base (perturbed) coords from the Structure
        base_coords = np.array(structure.cart_coords)  # shape (N,3)

        # 3) compute the new absolute coords
        coords = base_coords + disp  # shape (N,3)

        # 4) build ASE Atoms from the pymatgen Structure
        atoms = AseAtomsAdaptor.get_atoms(structure)

        # 5) update the ASE Atoms’ positions to your new coords
        atoms.positions = coords

        # 6) attach the calculator correctly
        calc = PESCalculator(potential=self.pot)
        atoms.calc = calc

        # 7) evaluate forces & energy
        forces = atoms.get_forces().astype(np.float32).reshape(-1)
        energy = float(atoms.get_potential_energy())

        # 8) convert back to torch
        f_tensor = torch.as_tensor(forces, dtype=torch.float32, device=pos_flat.device)
        e_tensor = torch.as_tensor(energy, dtype=torch.float32, device=pos_flat.device)
        return f_tensor, e_tensor

    def compute_fd_hessian(self, force_fn, disp0, eps=1e-3):
        """
        Placeholder while we figure out how to use torch's native hessian function
        """
        n = disp0.numel()
        H = torch.zeros(n, n, device=disp0.device)
        for j in range(n):
            e = torch.zeros_like(disp0)
            e[j] = eps
            f_plus, _ = force_fn(disp0 + e)
            f_minus, _ = force_fn(disp0 - e)
            H[:, j] = (f_plus - f_minus) / (2 * eps)
        return H

    def forward(
        self, input: torch.Tensor, structure: Structure, classifier
    ) -> torch.Tensor:
        """
        Compute loss between predicted displacements and the “true” displacement

        Parameters
        ----------
        input: torch.Tensor, shape (3*N,)
          The predicted displacement vector
        structure: Structure
          The perturbed pymatgen structure used to compute forces and Hessian
        classifier: Callable
          A loss function, e.g. nn.MSELoss()

        Returns
        -------
        loss: torch.Tensor
        """
        # build zero‐displacement vector with grad
        disp0 = torch.zeros_like(
            input, device=input.device, requires_grad=True
        )  # (3*N,)

        # get forces & energy at the perturbed geometry (disp0 = 0 → new_coords = structure.cart_coords)
        f0, e0 = self.predict_force_energy(disp0, structure)

        # Hessian in displacement space
        H = self.compute_fd_hessian(
            lambda d: self.predict_force_energy(d, structure), disp0
        )

        # Ideally we are able to run this
        # H  = torch.autograd.functional.hessian(lambda d: self.predict_force_energy(d, structure), disp0)

        # regularize & solve for δx: (H + λI) δx = –F₀
        diag_mean = H.diagonal().abs().mean()
        ridge_val = 1e-1 * diag_mean
        ridge = ridge_val * torch.eye(H.size(0), device=H.device)
        H_reg = H + ridge

        try:
            delta_x = torch.linalg.solve(H_reg, -f0)
        except RuntimeError:
            # fallback to least‑squares if still singular
            sol = torch.linalg.lstsq(H_reg, (-f0).unsqueeze(-1))
            delta_x = sol.solution.squeeze(-1)

        # reshape to match network’s output
        actual_disp = delta_x.view_as(input)

        print("Step norm:", actual_disp.norm().item())
        return classifier(input, actual_disp)
