import csv
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


def get_train_val_test_loader(
    dataset,
    batch_size=64,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    return_test=False,
    num_workers=1,
    pin_memory=False,
):
    """
    Utility function for dividing a dataset to train, val, test datasets.
    """
    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        warnings.warn(
            f"train_ratio is None, using 1 - val_ratio - test_ratio = {train_ratio} as training data."
        )
    else:
        assert train_ratio + val_ratio + test_ratio <= 1

    indices = list(range(total_size))
    train_size = int(train_ratio * total_size)
    test_size = int(test_ratio * total_size)
    valid_size = int(val_ratio * total_size)

    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[train_size : train_size + valid_size])
    if return_test:
        test_sampler = SubsetRandomSampler(
            indices[train_size + valid_size : train_size + valid_size + test_size]
        )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_pool,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_pool,
        pin_memory=pin_memory,
    )
    if return_test:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=collate_pool,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader
    return train_loader, val_loader, None


def collate_pool(dataset_list):
    """
    Collate a list of data points for per-atom displacement training:
      ((atom_fea, nbr_fea, nbr_idx), dr_true, struct_relaxed)
    Returns:
      (atom_batch, nbr_batch, idx_batch, crystal_atom_idx), dr_batch, batch_struct
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_idx = [], [], []
    crystal_atom_idx, batch_dr, batch_struct = [], [], []
    base_idx = 0
    for (atom_fea, nbr_fea, nbr_idx), dr_true, struct_relaxed in dataset_list:
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        # shift neighbor indices by base_idx for batching
        batch_nbr_idx.append(nbr_idx + base_idx)
        # record this crystal's atom indices
        crystal_atom_idx.append(
            torch.arange(base_idx, base_idx + n_i, dtype=torch.long)
        )
        batch_dr.append(dr_true)
        batch_struct.append(struct_relaxed)
        base_idx += n_i

    atom_batch = torch.cat(batch_atom_fea, dim=0)
    nbr_batch = torch.cat(batch_nbr_fea, dim=0)
    idx_batch = torch.cat(batch_nbr_idx, dim=0)
    dr_batch = torch.cat(batch_dr, dim=0)

    return (atom_batch, nbr_batch, idx_batch, crystal_atom_idx), dr_batch, batch_struct


class GaussianDistance(object):
    """Gaussian smearing for bond distances"""

    def __init__(self, dmin: float, dmax: float, step: float, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var if var is not None else step

    def expand(self, distances: np.ndarray):
        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2)


class AtomInitializer(object):
    """Base class for atom type embeddings"""

    def __init__(self, atom_types):
        self.atom_types: set = set(atom_types)
        self._embedding: dict = {}

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {v: k for k, v in self._embedding.items()}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, "_decodedict"):
            self._decodedict = {v: k for k, v in self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """Load element embeddings from JSON"""

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding: dict = json.load(f)
        elem_embedding = {int(k): v for k, v in elem_embedding.items()}
        super().__init__(set(elem_embedding.keys()))
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    CIFData with on-the-fly "relaxed" structures via random perturbation.
    Returns original features, per-atom displacement targets, and perturbed Structures.
    """

    def __init__(
        self,
        root_dir,
        max_num_nbr=12,
        radius=8.0,
        dmin=0.0,
        step=0.2,
        random_seed=123,
        perturb_std=0.01,
    ):
        self.root_dir = root_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.dmin = dmin
        self.step = step
        self.perturb_std = perturb_std
        random.seed(random_seed)

        # read id_prop.csv
        id_prop_file = os.path.join(root_dir, "id_prop.csv")
        assert os.path.exists(id_prop_file), "id_prop.csv missing"
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = list(reader)
        random.shuffle(self.id_prop_data)

        # load embeddings and distance filter
        atom_init_file = os.path.join(root_dir, "atom_init.json")
        assert os.path.exists(atom_init_file), "atom_init.json missing"
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        cif_id, _, _ = self.id_prop_data[idx]
        # load original crystal
        crystal = Structure.from_file(os.path.join(self.root_dir, f"{cif_id}.cif"))

        # original cartesian positions
        orig_pos = np.array(crystal.cart_coords)
        # generate small Gaussian perturbation
        delta = np.random.normal(scale=self.perturb_std, size=orig_pos.shape)
        new_coords = orig_pos + delta

        # build perturbed structure
        struct_relaxed = Structure(
            crystal.lattice, crystal.species, new_coords, coords_are_cartesian=True
        )

        # per-atom displacement target
        dr_true = torch.tensor(delta, dtype=torch.float32)

        # featurize original crystal
        atom_fea, nbr_fea, nbr_idx = self._featurize(crystal)

        return (atom_fea, nbr_fea, nbr_idx), dr_true, struct_relaxed

    def _featurize(self, crystal):
        # atom features
        atom_feats = [self.ari.get_atom_fea(site.specie.number) for site in crystal]
        atom_fea = torch.tensor(np.vstack(atom_feats), dtype=torch.float32)

        # build neighbor list
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_idx_list, nbr_dist_list = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn("Neighbors < max_num_nbr, padding with zeros.")
                idxs = [x[2] for x in nbr] + [0] * (self.max_num_nbr - len(nbr))
                dists = [x[1] for x in nbr] + [self.radius + 1] * (
                    self.max_num_nbr - len(nbr)
                )
            else:
                idxs = [x[2] for x in nbr[: self.max_num_nbr]]
                dists = [x[1] for x in nbr[: self.max_num_nbr]]
            nbr_idx_list.append(idxs)
            nbr_dist_list.append(dists)

        nbr_idx_arr = np.array(nbr_idx_list)
        nbr_dist_arr = np.array(nbr_dist_list)
        nbr_fea = self.gdf.expand(nbr_dist_arr)

        nbr_idx = torch.tensor(nbr_idx_arr, dtype=torch.long)
        nbr_fea = torch.tensor(nbr_fea, dtype=torch.float32)

        return atom_fea, nbr_fea, nbr_idx
