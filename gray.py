from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  

# Load your own image
logo_path = './bachkhoa.jpg'
logo = np.array(Image.open(logo_path))

# # Display the original image
# plt.imshow(logo)
# plt.title("Original Image")
# plt.show()

# Convert the image to grayscale manually
logo = np.dot(logo[..., :3], [0.2989, 0.5870, 0.1140])

# Calculate the SVD and plot the image
U, S, V_T = svd(logo, full_matrices=False)
S = np.diag(S)

# Plot approximations for different ranks
fig, ax = plt.subplots(5, 2, figsize=(8, 20))

curr_fig = 0

for r in [5, 10, 70, 100, 200]:
    logo_approx = U[:, :r] @ S[0:r, :r] @ V_T[:r, :]
    ax[curr_fig][0].imshow(logo_approx, cmap='gray')
    ax[curr_fig][0].set_title("k = " + str(r))
    ax[curr_fig, 0].axis('off')

    ax[curr_fig][1].set_title("Ảnh gốc")
    ax[curr_fig][1].imshow(logo, cmap='gray')
    ax[curr_fig, 1].axis('off')

    curr_fig += 1

plt.show()