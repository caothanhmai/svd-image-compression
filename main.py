from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Upload hinh anh
image_path = './bachkhoa.jpg'
image = np.array(Image.open(image_path))

# Ap dung SVD cho tung kenh mau (rgb)
# Các số 0, 1, 2 tương ứng với chỉ số của kênh màu trong không gian màu RGB, theo thứ tự là đỏ, xanh lá cây, và xanh dương.
U_r, S_r, V_T_r = svd(image[:, :, 0], full_matrices=False)
U_g, S_g, V_T_g = svd(image[:, :, 1], full_matrices=False)
U_b, S_b, V_T_b = svd(image[:, :, 2], full_matrices=False)

# Tạo ma trận đường chéo từ các giá trị singular values của ba kênh màu RGB: S_r, S_g, và S_b.
S_r = np.diag(S_r)
S_g = np.diag(S_g)
S_b = np.diag(S_b)

fig, ax = plt.subplots(5, 2, figsize=(8, 20))

curr_fig = 0

for r in [5, 10, 70, 100, 200]:

    image_approx = np.zeros_like(image)
    for i, (U, S, V_T) in enumerate([(U_r, S_r, V_T_r), (U_g, S_g, V_T_g), (U_b, S_b, V_T_b)]):
        image_approx[:, :, i] = U[:, :r] @ S[0:r, :r] @ V_T[:r, :]

    ax[curr_fig][0].imshow(image_approx.astype(np.uint8))
    ax[curr_fig][0].set_title("k = " + str(r))
    ax[curr_fig, 0].axis('off')

    ax[curr_fig][1].set_title("Ảnh gốc")
    ax[curr_fig][1].imshow(image)
    ax[curr_fig, 1].axis('off')

    curr_fig += 1

plt.show()
