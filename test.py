from skimage import io, color
import matplotlib.pyplot as plt
filename = "image.png"  
image_rgb = io.imread(filename)


image_gray = color.rgb2gray(image_rgb)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image (RGB)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image_gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.ion()

plt.show()
