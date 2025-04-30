import torch; import torch.nn as nn 
from m3gnet.models import Potential
from pymatgen.core import Structure

class M3gnetClassifier(nn.Module):
    """
    Uses the m3gnet nn to calculate the loss associated with particular displacements
    """ 
    def __init__(self):
        super().__init__()

    def predict_energy(self, x_flat: torch.Tensor, structure: Structure):
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
        return torch.as_tensor(Potential.get_energies(structure), dtype=torch.float32, device=x_flat)
    
    def predict_forces(self, x_flat: torch.Tensor, structure: Structure):
        """
        Parameters
        ----------
        structure: Structure
          Pymatgen structure used to predict energy
        
        Returns
        -------
        energy: torch.Tensor
          Returns the forces represented as a torch.Tensor
        """
        return torch.as_tensor(Potential.get_forces(structure), dtype=torch.float32, device=x_flat)

    def forward(self, input: torch.Tensor, target: Structure, classifier) -> torch.Tensor:
        """
        Calculates the loss associated with predicted displacement and actual displacement using 
        a chosen loss function

        Parameters
        ----------
        input: torch.Tensor
          Predicted displacement of the lattice

        target: Structure
          target (relaxed) structure
        
        classifier: 
          pytorch function that calculates the loss between input and target

        """
        pos0 = torch.tensor(target.cart_coords, dtype=torch.float32, requires_grad=True) # (N, 3)
        x0 = pos0.view(-1)   # (3*N, )

        # Compute forces and energy using prediction
        F = self.predict_forces(x0, target)

        # Calculate Hessian (force constants)
        H: torch.Tensor = torch.autograd.functional.hessian(lambda y: self.predict_energy(y, target), x0)

        # add small ridge for stability in the Hessian 
        ridge = 1e-6 * torch.eye(H.shape[0], device=H.device)
        H_inv: torch.Tensor = torch.linalg.inv(H + ridge)

        # Calculate the actual displacement vector by inverse Hooke's law
        delta_x = -H_inv.matmul(F)

        # Reshape back into (N, 3)
        actual_r = delta_x.view(-1, 3)
        print("Step norm:", actual_r.norm().item())

        return classifier(input, actual_r)
    



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