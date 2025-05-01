

# data/distorted/  BaTiO3.cif, SrTiO3.cif, …
# data/relaxed/    BaTiO3.cif, SrTiO3.cif, …


# Launch

# python src/disco/main.py \
#     data/distorted data/relaxed \
#     --task regression \
#     --batch-size  16 \
#     --epochs      50 \
#     --lr          1e-3 \
#     --atom-fea-len 64 \
#     --n-conv      3 \
#     --h-fea-len   128 \
#     --n-h         1







# import numpy as np

# def teacher_forces(r_flat):
#     """Given r_flat shape (3N,), returns forces shape (3N,) as numpy."""
#     struct = struct_template.copy()
#     struct.replace(r_flat.reshape(-1,3))
#     _, forces = teacher.predict_structure(struct)  
#     return forces.flatten()  # (3N,)

# def compute_hessian_fd(r_flat, eps=1e-3):
#     N3 = r_flat.size
#     H = np.zeros((N3, N3), dtype=float)
#     F0 = teacher_forces(r_flat)
#     for j in range(N3):
#         # unit vector in direction j
#         e = np.zeros_like(r_flat); e[j] = eps
#         Fp = teacher_forces(r_flat + e)
#         Fm = teacher_forces(r_flat - e)
#         # ∂F/∂r ≈ (Fp - Fm) / (2 eps)
#         J_col = (Fp - Fm) / (2 * eps)
#         # Hessian H_ij = - ∂F_i/∂r_j
#         H[:, j] = -J_col
#     return H

