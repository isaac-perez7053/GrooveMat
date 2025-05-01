import torch; import torch.nn as nn

class ConvLayer(nn.Module):
    """
    Convolution operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer. Message-level-embedder that sends message
        along the edges

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        # Create embedding for bond-atom-bond group
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len, 
                                 2*self.atom_fea_len)
        # Create sigmoid for gate
        self.sigmoid = nn.Sigmoid()

        # Create softplus for core
        self.softplus1 = nn.Softplus()

        # Create normalization functions for the in-between steps
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # Pull out each of the atom's M neighbor-features, yeilding an (N, M, A) tensor
        N, M = nbr_fea_idx.shape

        # Grab the neighboring atom's features
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]

        # Concatenate central, neighbor, and bond features (atom-atom-edge)
        total_nbr_fea = torch.cat([
        atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),  # central feats repeated (N, M, A)
        atom_nbr_fea,                                              # neighbor feats       (N, M, A)
        nbr_fea                                                    # bond feats           (N, M, B)
        ], dim=2)                                                  # → (N, M, 2A + B)
        
        # Pass through Linear nn
        total_gated_fea = self.fc_full(total_nbr_fea) # (N, M, 2A)

        # Normalize results. bn expects a 2D tensor. Flatten then reshape
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        
        # Split into filter and core
        nbr_filter, nbr_core = total_gated_fea.chuck(2, dim=2)
        
        # Pass gate and core through sigmoid and softplus respectively
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        # Weight and sum over neighbors
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1) # (N, A)

        # Apply normalization
        nbr_sumed = self.bn2(nbr_sumed)

        # Residual add + activation 
        out = self.softplus2(atom_in_fea + nbr_sumed) # (N, A)

        return out
        

class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting
    predicting material properties (in this case, convergence)
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        
        # node-level embedder that brings raw inputs into same dimensionality for
        # convolution layer
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # Initialize convolutional stack. Each layer implements a round of message-passing
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                              for _ in range(n_conv)])
        
        # Map from conv_space to FC-space
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        # add a smooth nonlinearity
        self.conv_to_fc_softplus = nn.Softplus()

        # Optional hidden layers. If you want more hidden layers after pooling, 
        # created here

        if n_h > 1: 
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h - 1)])
            
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h - 1)])
            
        # Finish processing the input with classification or regression
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """

        # Embed atom feature vector
        atom_fea = self.embedding(atom_fea)

        # Pass atom embedding through convolution layer (message passing)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)

        # Pool-to-crystal level features. For each crystal, gather its subset of N
        # atom features and sums them
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        # Project conv-space to hidden MLP-space
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        # Optional dropout for classication only
        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        out = self.fc_out(crys_fea)

        if self.classification:
            out = self.logsoftmax(out)

        return out
    
    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]

        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        
        return torch.cat(summed_fea, dim=0)


class ConvergenceRegressor(CrystalGraphConvNet):
    def __init__(self,
                orig_atom_fea_len,
                nbr_fea_len,
                atom_fea_len, n_conv, 
                h_fea_len,
                n_h, 
                classification=False):
    
        # Build as a regressor 
        super().__init__(orig_atom_fea_len, 
                         nbr_fea_len,
                         atom_fea_len,
                         n_conv,
                         h_fea_len,
                         n_h,
                         classification)
        
        # Override the output to be 3 dim per atom. Pooling step unnecessary
        self.fc_out = nn.Linear(h_fea_len, 3)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx=None):

        # Embed atoms
        x = self.embedding(atom_fea) # (N, F)

        # Message pass through convolution layers
        for conv in self.convs:
            x = conv(x, nbr_fea, nbr_fea_idx) # still (N, F)

        # Project each atom into the 3-vector space
        x = self.conv_to_fc_softplus(self.conv_to_fc(x)) # (N, H)

        # If extra layers are created, run through 
        if hasattr(self, 'fcs'):
            for fc, act in zip(self.fcs, self.softpluses):
                # applies linear layer to x and applies soft max. Repeats
                x = act(fc(x))

        # Serves as our per-atom displacement vectors
        out = self.fc_out(x)

        return out


# Training: 

# model = ConvergenceRegressor(...)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.MSELoss()

# for batch in dataloader:
#     atom_fea, nbr_fea, nbr_idx, true_deltas = batch
#     pred_deltas = model(atom_fea, nbr_fea, nbr_idx)
#     loss = criterion(pred_deltas, true_deltas)  # both are (N,3)
#     loss.backward(); optimizer.step(); optimizer.zero_grad()


