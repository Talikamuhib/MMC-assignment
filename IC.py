import numpy as np

def svd_small_matrix_example():
    # Step 1: Construct the "image" matrix (3x2)
    I = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ], dtype=float)
    print("Original matrix I:\n", I, "\n")

    # Step 2: Compute the SVD (thin form)
    U, S, VT = np.linalg.svd(I, full_matrices=False)
    # U is 3x2, S is length-2, VT is 2x2

    print("U:\n", U)
    print("S (singular values):\n", S)
    print("VT:\n", VT, "\n")

    # Step 3: Reconstruct I exactly (rank-2) from U, S, VT
    # Build Sigma as diag(S), then multiply: U * Sigma * VT
    Sigma = np.diag(S)   # 2x2
    I_reconstructed = U @ Sigma @ VT
    print("Full SVD reconstruction (rank=2):\n", I_reconstructed, "\n")

    # Step 4: Form a rank-1 approximation
    # Keep only the largest singular value (S[0])
    # and the corresponding first column of U, first row of VT.
    # U_1 is 3x1, Sigma_1 is 1x1, VT_1 is 1x2
    U_1 = U[:, [0]]        # shape (3,1)
    Sigma_1 = np.array([[S[0]]])  # shape (1,1)
    VT_1 = VT[[0], :]      # shape (1,2)
    
    I_rank1 = U_1 @ Sigma_1 @ VT_1
    print("Rank-1 approximation:\n", I_rank1, "\n")

    # Step 5: Compare rank-1 approximation with the original
    # (Compute the difference or the Frobenius norm of the difference)
    diff = I - I_rank1
    frob_error = np.linalg.norm(diff, 'fro')
    print("Difference (I - I_rank1):\n", diff)
    print(f"Frobenius norm of difference = {frob_error:.4f}")

if __name__ == "__main__":
    svd_small_matrix_example()
