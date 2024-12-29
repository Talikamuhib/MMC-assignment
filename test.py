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
    U, S, Vt = np.linalg.svd(M)

    # Print SVD components
    print("Matrix:")
    print(M)
    print("U:")
    print(U)
    print("Singular values:")
    print(S)
    print("Vt:")
    print(Vt)

    # Compute projection of q onto the subspace spanned by U
    projection = U @ U.T @ q

    # Compute the residual
    residual = np.linalg.norm(q - projection)
    return projection, residual

# Compute residuals for Class A and Class B
print("SVD for Class A:")
projection_A, residual_A = compute_residual(M_A, q)
print("\nSVD for Class B:")
projection_B, residual_B = compute_residual(M_B, q)

# Classification based on residuals
if residual_A < residual_B:
    classification = "Class A"
elif residual_A > residual_B:
    classification = "Class B"
else:
    classification = "Equidistant from both classes"

# Print the results
print("\nMatrix M_A:")
print(M_A)
print("\nMatrix M_B:")
print(M_B)
print("\nTest Vector q:")
print(q)
print("\nResidual for Class A: {:.4f}".format(residual_A))
print("Residual for Class B: {:.4f}".format(residual_B))
print("Classification: {}".format(classification))
