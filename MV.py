import numpy as np
from numpy.linalg import svd

# Define the original matrix M
M = np.array([
    [2, 1, 0],
    [0, 2, 1],
    [1, 0, 2]
])

# Perform Singular Value Decomposition
U, S, Vt = svd(M)

# Rank-2 approximation: Set the smallest singular value to 0
S_rank2 = np.copy(S)
S_rank2[2] = 0  # Set the third singular value to zero

# Create the rank-2 diagonal matrix
S_matrix_rank2 = np.zeros_like(M, dtype=float)
np.fill_diagonal(S_matrix_rank2, S_rank2)

# Reconstruct the matrix using the top 2 singular values
M_rank2 = U @ S_matrix_rank2 @ Vt

# Reconstruct the rank-2 approximated matrix explicitly
M_rank2_reconstructed = U @ S_matrix_rank2 @ Vt

# Replace missing values (zeros in the original matrix) with approximated values
M_filled = np.where(M == 0, M_rank2_reconstructed, M)

# Print results
print("Original Matrix:")
print(M)
print("\nLeft Singular Vectors (U):")
print(U)
print("\nSingular Values (S):")
print(S)
print("\nRight Singular Vectors Transposed (Vt):")
print(Vt)
print("\nMatrix with Missing Values Filled:")
print(M_filled)
