import numpy as np

import matplotlib.pyplot as plt
from skimage import io, color, transform, img_as_float

def CompressImage(filename):
    """
    Compresses an image using Singular Value Decomposition (SVD) and displays
    the original and compressed images for various ranks.

    Parameters:
    ----------
    filename : str
        Path to the image file to be compressed.

    Returns:
    -------
    None
        Displays the original and compressed images using matplotlib.
    """
    # Step 1: Read the Image and Convert to Grayscale
    try:
        # Read the image from the given filename
        image_rgb = io.imread("filename")
    except FileNotFoundError:
        print(f"Error: The file was not found.")
        return
    except Exception as e:
        print(f"Error: An error occurred while reading the file: {e}")
        return

    # Convert to grayscale if the image is colored
    if image_rgb.ndim == 3:
        image_gray = color.rgb2gray(image_rgb)
    else:
        image_gray = image_rgb.astype(float)

    # Step 2: Resize the Image to 50% of its Original Size
    # Using anti-aliasing to reduce artifacts
    image_resized = transform.resize(image_gray, 
                                     (int(image_gray.shape[0] * 0.5), 
                                      int(image_gray.shape[1] * 0.5)), 
                                     anti_aliasing=True)
    
    # Convert the image to floating point representation (similar to im2double)
    image = img_as_float(image_resized)

    # Step 3: Display the Original Image
    plt.figure(figsize=(12, 16))  # Adjust the figure size as needed
    plt.subplot(4, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')  # Hide axis ticks

    # Step 4: Perform Singular Value Decomposition (SVD)
    # X = U * S * Vt
    U, S, Vt = np.linalg.svd(image, full_matrices=False)

    # Step 5: Extract Singular Values (Diagonal Entries of S)
    singular_values = S.copy()

    # Step 6: Define the Desired Ranks for Compression
    # These ranks determine how many singular values to keep
    ranks = [320, 160, 80, 40, 20, 10, 5]

    # Determine the maximum possible rank based on the image dimensions
    max_rank = min(image.shape)
    # Adjust ranks to not exceed the maximum possible rank
    ranks = [rank for rank in ranks if rank <= max_rank]

    # If the image is smaller than the smallest desired rank, add it
    if ranks and ranks[-1] > max_rank:
        ranks.append(max_rank)
    elif not ranks:
        # If all desired ranks are larger than max_rank, set ranks to max_rank
        ranks = [max_rank]

    # Step 7: Iterate Over Each Desired Rank to Compress the Image
    for i, rank in enumerate(ranks):
        # Step 7a: Zero Out Singular Values Beyond the Current Rank
        compressed_singular_values = singular_values.copy()
        if rank < len(compressed_singular_values):
            compressed_singular_values[rank:] = 0
        # If rank equals the number of singular values, no change is needed

        # Step 7b: Create the Compressed Sigma Matrix
        compressed_S = np.diag(compressed_singular_values)

        # Step 7c: Reconstruct the Compressed Image
        # Using only the top 'rank' singular values
        approx_image = np.dot(U, np.dot(compressed_S, Vt))

        # Step 7d: Display the Compressed Image
        plt.subplot(4, 2, i + 2)  # Subplot positions: 2 to 8
        plt.imshow(approx_image, cmap='gray')
        plt.title(f'Rank {rank} Image')
        plt.axis('off')  # Hide axis ticks

    # Step 8: Adjust Layout and Display All Plots
    plt.tight_layout()
    plt.show()

# Example Usage:
CompressImage('/workspaces/MMC-assignment/image.png')
