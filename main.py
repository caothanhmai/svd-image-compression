from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load your own color image
image_path = './bachkhoa.jpg'
image = np.array(Image.open(image_path))

# Perform SVD separately for each color channel
U_r, S_r, V_T_r = svd(image[:, :, 0], full_matrices=False)
U_g, S_g, V_T_g = svd(image[:, :, 1], full_matrices=False)
U_b, S_b, V_T_b = svd(image[:, :, 2], full_matrices=False)

# Diagonal matrices from singular values
S_r = np.diag(S_r)
S_g = np.diag(S_g)
S_b = np.diag(S_b)

# Plot approximations for different ranks for each color channel
fig, ax = plt.subplots(5, 2, figsize=(8, 20))

curr_fig = 0

for r in [5, 10, 70, 100, 200]:
    # Reconstruct each channel separately
    image_approx = np.zeros_like(image)
    for i, (U, S, V_T) in enumerate([(U_r, S_r, V_T_r), (U_g, S_g, V_T_g), (U_b, S_b, V_T_b)]):
        image_approx[:, :, i] = U[:, :r] @ S[0:r, :r] @ V_T[:r, :]

    print('origin')
    ax[curr_fig][0].imshow(image_approx.astype(np.uint8))
    ax[curr_fig][0].set_title("k = " + str(r))
    ax[curr_fig, 0].axis('off')

    ax[curr_fig][1].set_title("Original Color Image")
    print('test')
    ax[curr_fig][1].imshow(image)
    ax[curr_fig, 1].axis('off')

    curr_fig += 1

print('ok')
plt.show()
