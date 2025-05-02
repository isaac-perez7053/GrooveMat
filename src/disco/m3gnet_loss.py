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
        Predict the forces and energy of the displaced unit cell using m3gnet

        Parameters
        ----------
        x_flat: torch.Tensor
          The predicted displacement array
        structure: Structure
          The relaxed structure

        Returns
        -------
        E_torch: torch.Tensor
          The energy represented as a torch tensor
        F_torch: torch.Tensor
          The force represented as a torch tensor
        """
        # Find the displacement of the cell 
        disp = x_flat.detach().cpu().numpy().reshape(-1, 3)

        orig_coords = structure.cart_coords     
        new_coords  = orig_coords + disp        

        # Initialize the displaced structure object
        shifted = Structure(
            structure.lattice,
            structure.species,
            new_coords,
            coords_are_cartesian=True
        )

        # Predict the forces and energy of the shifted model
        E_tf, F_tf = self.pot.get_ef(shifted)
        E_np = E_tf.numpy()
        F_np = F_tf.numpy()

        # Turn numpy tensors into torch tensors 
        E_torch = torch.as_tensor(E_np, dtype=torch.float32, device=x_flat.device)
        F_torch = torch.as_tensor(F_np, dtype=torch.float32, device=x_flat.device)

        return F_torch, E_torch

    
    def forward(self, input: torch.Tensor, target: Structure, classifier) -> torch.Tensor:
        """
        The implementation of the loss function

        Parameters
        ----------
        input: torch.Tensor
          The predicted displacement array
        target: Structure
          The relaxed structure
        classifier
          The loss function used to predict the difference between the predicted
          and actual displacement arrays

        Returns
        -------
        loss: torch.Tensor
          Returns the loss as a torch tensor
        """
        # Build a flat “reference” coordinate vector x0 with grad enabled
        pos0 = torch.tensor(
            target.cart_coords,
            dtype=torch.float32,
            device=input.device,
            requires_grad=True,
        )                   # (N,3)
        x0 = pos0.view(-1)  # (3*N,)

        # Get forces & energy at x0
        f0, e0 = self.predict_force_energy(x0, target)
        #    — predict_force_energy returns (forces, energy)

        # Define an energy‐only function for Hessian
        energy_fn = lambda y: self.predict_force_energy(y, target)[1]

        # Compute Hessian d^2E/dy^2 at x0
        H = torch.autograd.functional.hessian(energy_fn, x0)

        # Invert (with a tiny ridge for stability)
        ridge = 1e-6 * torch.eye(H.size(0), device=H.device)
        H_inv = torch.linalg.inv(H + ridge)

        # Actual displacement by Δx = −(inv Hessian) ⋅ f
        delta_x = -H_inv.matmul(f0)     # (3*N,)

        # Reshape to match whatever shape `input` has
        actual_disp = delta_x.view_as(input)

        print("Step norm:", actual_disp.norm().item())

        # Return your loss between predicted vs. “actual”
        return classifier(input, actual_disp)

# usage
# criterion = M3gnetClassifier()
# loss = criterion(predict_r, structure, classifier) 