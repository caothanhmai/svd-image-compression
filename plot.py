from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Đọc và Số hóa Dữ liệu
def load_and_convert_image(file_path):
    image = np.array(Image.open(file_path))
    return image

image_path = './bachkhoa.jpg'
original_image = load_and_convert_image(image_path)

# Áp dụng SVD cho từng kênh màu (rgb)
U_r, S_r, V_T_r = svd(original_image[:, :, 0], full_matrices=False)
U_g, S_g, V_T_g = svd(original_image[:, :, 1], full_matrices=False)
U_b, S_b, V_T_b = svd(original_image[:, :, 2], full_matrices=False)

# Tạo ma trận đường chéo S
S_r = np.diag(S_r)
S_g = np.diag(S_g)
S_b = np.diag(S_b)

# Giảm chiều dữ liệu
k_values = [5, 10, 70, 100, 200]
mse_values = []
compression_ratios = []
image_sizes = []

# Convert bytes to kilobytes
def bytes_to_kilobytes(bytes_size):
    return bytes_size / 1024.0

for r in k_values:
    # Tạo ảnh tái tạo cho mỗi kênh màu
    image_approx = np.zeros_like(original_image)
    for i, (U, S, V_T) in enumerate([(U_r, S_r, V_T_r), (U_g, S_g, V_T_g), (U_b, S_b, V_T_b)]):
        image_approx[:, :, i] = U[:, :r] @ S[0:r, :r] @ V_T[:r, :]

    # Đánh giá sai số sử dụng Mean Square Error (MSE - Sai số toàn phương trung bình)
    mse = np.sum((original_image - image_approx) ** 2) / float(original_image.size)
    mse_values.append(mse)

    # Compression ratio
    compression_ratio = original_image.size / (U[:, :r].size + S[0:r, :r].size + V_T[:r, :].size)
    compression_ratios.append(compression_ratio)

    # Image size in kilobytes
    image_size = U[:, :r].size + S[0:r, :r].size + V_T[:r, :].size
    image_size_kb = bytes_to_kilobytes(image_size)
    image_sizes.append(image_size_kb)

# # Plotting the graph
# plt.figure(figsize=(10, 6))
# plt.plot(k_values, image_sizes, marker='o', label='Image Size (KB)')
# plt.xlabel('k (Number of Singular Values)')
# plt.ylabel('Image Size (KB)')
# plt.title('Image Size')
# plt.legend()
# plt.grid(True)

# # Display the plot
# plt.show()


# Plotting the graph for MSE
plt.figure(figsize=(10, 6))
plt.plot(k_values, mse_values, marker='o', label='MSE')
plt.xlabel('k (Number of Singular Values)')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# Display the plot for MSE
plt.show()