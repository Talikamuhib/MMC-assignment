import numpy as np

# Step 1: Define the Matrix M
M = np.array([
    [2, 1, 0],
    [0, 2, 1],
    [1, 0, 2]
], dtype=float)

print("Original Matrix (M):")
print(M)
print("\n")

# Step 2: Compute M^T M
M_transpose = M.T
N = np.dot(M_transpose, M)

print("Transpose of M (M^T):")
print(M_transpose)
print("\n")

print("Matrix N = M^T * M:")
print(N)
print("\n")

# Step 3: Find Eigenvalues and Eigenvectors of N
# Since N is symmetric, use eigh which is optimized for symmetric matrices
eigenvalues, eigenvectors = np.linalg.eigh(N)

print("Eigenvalues of N (sorted in ascending order):")
print(eigenvalues)
print("\n")

print("Corresponding Eigenvectors of N:")
print(eigenvectors)
print("\n")

# Step 4: Sort Eigenvalues and Eigenvectors in Descending Order
# Argsort in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

print("Sorted Eigenvalues (Descending):")
print(eigenvalues_sorted)
print("\n")

print("Corresponding Sorted Eigenvectors:")
print(eigenvectors_sorted)
print("\n")

# Step 5: Compute Singular Values Sigma as sqrt of Eigenvalues
singular_values = np.sqrt(eigenvalues_sorted)
print("Singular Values (Sigma):")
print(singular_values)
print("\n")

# Step 6: Form the Right Singular Vectors Matrix V
V = eigenvectors_sorted
print("Right Singular Vectors (V):")
print(V)
print("\n")

# Step 7: Compute the Left Singular Vectors Matrix U
# U = (1 / sigma_i) * M * V_i for each singular value and corresponding V_i
U = np.zeros_like(M)

for i in range(len(singular_values)):
    if singular_values[i] > 1e-10:  # Avoid division by zero
        U[:, i] = np.dot(M, V[:, i]) / singular_values[i]
    else:
        U[:, i] = 0  # If singular value is zero, set the column to zero

print("Left Singular Vectors (U):")
print(U)
print("\n")

# Step 8: Assemble the Diagonal Matrix Sigma
Sigma = np.zeros_like(N, dtype=float)
np.fill_diagonal(Sigma, singular_values)

print("Diagonal Matrix Sigma:")
print(Sigma)
print("\n")

# Step 9: Verify that M ≈ U * Sigma * V^T
M_reconstructed = np.dot(U, np.dot(Sigma, V.T))
print("Reconstructed Matrix (U * Sigma * V^T):")
print(M_reconstructed)
print("\n")

# Calculate the Frobenius norm of the difference
difference = np.linalg.norm(M - M_reconstructed)
print(f"Frobenius Norm of (M - U*Sigma*V^T): {difference:.6f}")
print("\n")

# Optional: Display U, Sigma, V^T neatly
print("Final SVD Components:")
print("U Matrix:")
print(U)
print("\nSigma Matrix:")
print(Sigma)
print("\nV^T Matrix:")
print(V.T)
print("\n")
