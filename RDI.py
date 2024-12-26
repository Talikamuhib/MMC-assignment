import numpy as np

# Step 1: Define the Ratings Matrix M
# Rows correspond to users: Alice, Bob, Carol
# Columns correspond to movies: Inception (Sci-Fi), Titanic (Romance), Jaws (Thriller)
# A rating of 0 indicates a missing (unrated) entry.

M = np.array([
    [5, 0, 3],  # Alice
    [4, 3, 0],  # Bob
    [0, 5, 4]   # Carol
], dtype=float)

print("Original Ratings Matrix (M):")
print(M)
print("\n")

# Step 2: Perform Singular Value Decomposition (SVD) on M
# SVD decomposes M into U, Sigma, and Vt such that M = U * Sigma * Vt
U, singular_values, Vt = np.linalg.svd(M, full_matrices=True)

print("Left Singular Vectors (U):")
print(U)
print("\n")

print("Singular Values (Sigma):")
print(singular_values)
print("\n")

print("Right Singular Vectors Transposed (Vt):")
print(Vt)
print("\n")

# Step 3: Reconstruct the Matrix M from its SVD components
# To reconstruct M, we need to form the Sigma matrix with appropriate dimensions
Sigma = np.zeros_like(M, dtype=float)
np.fill_diagonal(Sigma, singular_values)

M_reconstructed = np.dot(U, np.dot(Sigma, Vt))

print("Reconstructed Ratings Matrix (U * Sigma * Vt):")
print(M_reconstructed)
print("\n")

# Step 4: Predict Missing Ratings by Filling in the Zeros
# We'll create a copy of the original matrix and fill in the missing values
M_predicted = M.copy()

# Identify indices where the rating is 0 (missing)
missing_indices = np.where(M == 0)

print("Missing Indices (User, Movie):")
for user_idx, movie_idx in zip(*missing_indices):
    print(f"User {user_idx + 1}, Movie {movie_idx + 1}")

print("\n")

# Fill in the missing ratings from the reconstructed matrix
for user_idx, movie_idx in zip(*missing_indices):
    predicted_rating = M_reconstructed[user_idx, movie_idx]
    M_predicted[user_idx, movie_idx] = predicted_rating
    print(f"Predicted rating for User {user_idx + 1}, Movie {movie_idx + 1}: {predicted_rating:.2f}")

print("\n")

print("Ratings Matrix with Predicted Values:")
print(M_predicted)
print("\n")

# Step 5: Make Recommendations Based on Predicted Ratings
# For each user, identify the movie with the highest predicted rating among the missing ones

# Define user and movie names for clarity
users = ["Alice", "Bob", "Carol"]
movies = ["Inception (Sci-Fi)", "Titanic (Romance)", "Jaws (Thriller)"]

print("Recommendations:")

for user_idx, user in enumerate(users):
    # Find movies not rated by the user
    unrated_movies_indices = np.where(M[user_idx] == 0)[0]
    if len(unrated_movies_indices) == 0:
        print(f"{user} has rated all movies.")
        continue
    
    # Predict ratings for these movies
    predicted_ratings = M_predicted[user_idx, unrated_movies_indices]
    
    # Find the movie with the highest predicted rating
    best_movie_idx = unrated_movies_indices[np.argmax(predicted_ratings)]
    best_rating = predicted_ratings.max()
    
    print(f"- {user} has not rated {movies[unrated_movies_indices][0]}.")
    print(f"  Predicted rating: {best_rating:.2f}")
    # Optionally, recommend if the rating is above a certain threshold
    # For this example, we'll recommend the movie if predicted rating >= 3
    if best_rating >= 3:
        print(f"  Recommendation: Recommend '{movies[best_movie_idx]}' to {user}.")
    else:
        print(f"  Recommendation: No strong recommendation for {user} based on predicted ratings.")

print("\n")

# Optional: Rank k approximation (e.g., k=2) to see if predictions improve
# This is often done to reduce noise and improve generalization

k = 2
U_k = U[:, :k]
Sigma_k = np.diag(singular_values[:k])
Vt_k = Vt[:k, :]

M_approx_k = np.dot(U_k, np.dot(Sigma_k, Vt_k))

print(f"Rank-{k} Approximation of M:")
print(M_approx_k)
print("\n")

# Predict missing ratings using the rank-k approximation
M_predicted_k = M.copy()

for user_idx, movie_idx in zip(*missing_indices):
    predicted_rating_k = M_approx_k[user_idx, movie_idx]
    M_predicted_k[user_idx, movie_idx] = predicted_rating_k
    print(f"Predicted rating (Rank-{k}) for User {user_idx + 1}, Movie {movie_idx + 1}: {predicted_rating_k:.2f}")

print("\n")

print(f"Ratings Matrix with Predicted Values (Rank-{k} Approximation):")
print(M_predicted_k)
print("\n")

# Make Recommendations Based on Rank-k Predicted Ratings
print(f"Recommendations based on Rank-{k} Approximation:")

for user_idx, user in enumerate(users):
    # Find movies not rated by the user
    unrated_movies_indices = np.where(M[user_idx] == 0)[0]
    if len(unrated_movies_indices) == 0:
        print(f"{user} has rated all movies.")
        continue
    
    # Predict ratings for these movies
    predicted_ratings_k = M_predicted_k[user_idx, unrated_movies_indices]
    
    # Find the movie with the highest predicted rating
    best_movie_idx_k = unrated_movies_indices[np.argmax(predicted_ratings_k)]
    best_rating_k = predicted_ratings_k.max()
    
    print(f"- {user} has not rated {movies[unrated_movies_indices][0]}.")
    print(f"  Predicted rating (Rank-{k}): {best_rating_k:.2f}")
    if best_rating_k >= 3:
        print(f"  Recommendation: Recommend '{movies[best_movie_idx_k]}' to {user}.")
    else:
        print(f"  Recommendation: No strong recommendation for {user} based on predicted ratings.")

