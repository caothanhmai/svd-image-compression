from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from sklearn.metrics import mean_squared_error


# Đọc và Số hóa Dữ liệu
def load_and_convert_image(file_path):
    image = np.array(Image.open(file_path))
    return image

image_path = './bachkhoa.jpg'
original_image = load_and_convert_image(image_path)

# Function to get file size in kilobytes
def get_file_size(file_path):
    return os.path.getsize(file_path) / 1024

# Print the size of the original image
original_image_size_kb = get_file_size(image_path)
print("Kích thước ảnh gốc:", original_image_size_kb, "KB")

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(image1, image2):
    return mean_squared_error(image1, image2)


# Áp dụng SVD cho từng kênh màu (rgb)
U_r, S_r, V_T_r = svd(original_image[:, :, 0], full_matrices=False)
U_g, S_g, V_T_g = svd(original_image[:, :, 1], full_matrices=False)
U_b, S_b, V_T_b = svd(original_image[:, :, 2], full_matrices=False)

# Tạo ma trận đường chéo S
S_r = np.diag(S_r)
S_g = np.diag(S_g)
S_b = np.diag(S_b)

fig, ax = plt.subplots(5, 2, figsize=(8, 20))
curr_fig = 0

r_values = [5, 10, 70, 100, 200]
mse_values = []
compression_ratios = []
image_sizes = []

for r in [5, 10, 70, 100, 200]:
    # Giam chieu du lieu
    image_approx = np.zeros_like(original_image)
    for i, (U, S, V_T) in enumerate([(U_r, S_r, V_T_r), (U_g, S_g, V_T_g), (U_b, S_b, V_T_b)]):
        image_approx[:, :, i] = U[:, :r] @ S[:r, :r] @ V_T[:r, :]
        
    # Hien thi anh nen va anh goc
    ax[curr_fig][0].imshow(image_approx.astype(np.uint8))
    ax[curr_fig][0].set_title(f"k = {r}")
    ax[curr_fig, 0].axis('off')

    ax[curr_fig][1].set_title("Ảnh gốc")
    ax[curr_fig][1].imshow(original_image)
    ax[curr_fig, 1].axis('off')
    
    print('***** r = ', r)

    # Tinh kich thuoc anh nen
    compressed_image_path = f'compressed_image_r_{r}.jpg'
    plt.imsave(compressed_image_path, image_approx.astype(np.uint8))
    compressed_image_size_kb = get_file_size(compressed_image_path)
    print(f"Kích thước ảnh sau khi nén: {compressed_image_size_kb} KB")
    image_sizes.append(compressed_image_size_kb)

    # Tinh MSE giua anh goc va anh nen
    mse = calculate_mse(original_image.flatten(), image_approx.flatten())
    print(f"MSE giữa ảnh gốc và ảnh nén: {mse}")
    mse_values.append(mse)
    
    # Tinh ti le nen
    compression_ratio = original_image_size_kb / compressed_image_size_kb
    print(f"Tỷ lệ nén: {compression_ratio}")
    compression_ratios.append(compression_ratio)

    curr_fig += 1


plt.figure(figsize=(10, 6))
plt.plot(r_values, image_sizes, marker='o', label='Size')
plt.xlabel('r (Number of Singular Values)')
plt.ylabel('Size (KB)')
plt.legend()
plt.grid(True)
plt.show()
