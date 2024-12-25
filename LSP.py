import numpy as np

def svd_least_squares():
    # Step 1: Construct A and b
    # ---------------------------------------
    # Data points: (x, y) = (1,2), (2,3), (3,5), (4,4), (5,6)
    # We want y = a*x + b, so A has columns [x, 1], and b is the vector of y-values.

    A = np.array([
        [1, 1],
        [2, 1],
        [3, 1],
        [4, 1],
        [5, 1]
    ], dtype=float)

    b = np.array([2, 3, 5, 4, 6], dtype=float)

    # Step 2: Compute the SVD of A
    # ---------------------------------------
    # full_matrices=False produces the "thin" SVD for a 5x2 matrix
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # U is 5x2, S is a length-2 vector of singular values, Vt is 2x2

    # Step 3: Form the pseudoinverse A^+ = V Σ^+ U^T
    # ---------------------------------------
    # Since S is returned as a 1D array [σ₁, σ₂], we can invert it (for nonzero σᵢ).
    # S+ = diag(1/σᵢ).
    S_inv = np.diag(1.0 / S)  # 2x2 diagonal matrix with reciprocals of S on the diagonal

    # Vt is the transpose of V, so V = Vt.T
    V = Vt.T  # 2x2
    # U^T is the transpose of U, i.e. U.T

    # Now compute the pseudoinverse
    A_plus = V @ S_inv @ U.T  # This will be a 2x5 matrix

    # Step 4: Solve for x = A^+ b
    # ---------------------------------------
    x = A_plus @ b  # x will be [a, b]

    a, intercept = x

    print("---- SVD-based Least-Squares Solution ----")
    print(f"a = {a:.4f}, b = {intercept:.4f}")
    print(f"The best-fit line is y = {a:.3f}*x + {intercept:.3f}")
    print()

    # Step 5 (Optional): Validate with np.linalg.lstsq
    # ------------------------------------------------
    # np.linalg.lstsq(A, b, rcond=None) solves min ||A*x - b|| in the least-squares sense
    x_lstsq, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)
    a_ls, b_ls = x_lstsq

    print("---- np.linalg.lstsq Built-in Check ----")
    print(f"a (lstsq) = {a_ls:.4f}, b (lstsq) = {b_ls:.4f}")
    print(f"Residuals: {residuals}")
    print(f"Rank of A: {rank}")
    print(f"Singular values of A: {svals}")


if __name__ == "__main__":
    svd_least_squares()
