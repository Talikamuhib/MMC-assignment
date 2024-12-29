import numpy as np

# Given matrices and test vector
M_A = np.array([[1, 0, 1],
                [0, 1, 1],
                [1, 1, 0]])

M_B = np.array([[2, 1, 0],
                [1, 0, 1],
                [0, 1, 2]])

q = np.array([1, 1, 1])

# Function to compute SVD and residuals
def compute_residual(M, q):
    # Perform Singular Value Decomposition (SVD)
    U, _, _ = np.linalg.svd(M)

    # Compute projection of q onto the subspace spanned by U
    projection = U @ U.T @ q

    # Compute the residual
    residual = np.linalg.norm(q - projection)
    return residual

# Compute residuals for Class A and Class B
residual_A = compute_residual(M_A, q)
residual_B = compute_residual(M_B, q)

# Classification based on residuals
if residual_A < residual_B:
    classification = "Class A"
elif residual_A > residual_B:
    classification = "Class B"
else:
    classification = "Equidistant from both classes"

# Print the results
print(f"Residual for Class A: {residual_A:.4f}")
print(f"Residual for Class B: {residual_B:.4f}")
print(f"Classification: {classification}")
