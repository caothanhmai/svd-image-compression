import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image

compression_factor = 1  


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    return image

def get_svd_compressed(singular_values, matrix_to_decompose):
    (U, S, V) = np.linalg.svd(matrix_to_decompose, full_matrices=False)
    U_c = U[:, :singular_values]
    S_c = S[:singular_values]
    V_c = V[:singular_values, :]
    return (U_c, S_c, V_c)


def multiply_compressed(U, S, V):
    return np.matmul(np.matmul(U, np.diag(S)), V)


def reconstruct(compressed_image):
    result = []
    for element in compressed_image:
        U, S, V = element
        result.append(multiply_compressed(U, S, V))
    return np.dstack(result).astype(np.uint8)


def get_compressed_image_size(compressed_image):
    result = 0
    counter = 0
    for dim in compressed_image:
        for element in dim:
            result += element.size
            counter += 1
    print(f"Went through {counter} dimensions")
    return result


if __name__ == '__main__':
    image = load_image("./logo.png")
    print(f"Image shape when loaded: {image.shape}")
    print(f"Original image size: {image.size}")

    # split into 2d arrays for SVD
    R, G, B = [image[:, :, i] for i in range(3)]
    
    no_of_singular_values = math.ceil(max(R.shape) * compression_factor)
    compressed_image = [get_svd_compressed(no_of_singular_values, dim) for dim in (R, G, B)]
    
    compressed_size = get_compressed_image_size(compressed_image)
    print(f"Compressed image size: {compressed_size}")

    reconstructed_image = reconstruct(compressed_image)

    space_saved = image.size - compressed_size
    space_saved_percent = (1 - (compressed_size / image.size)) * 100
    print(f"Space saved in bytes: {space_saved} ({space_saved_percent:.2f}%)")
    
    
    plt.imshow(reconstructed_image, interpolation="nearest")
    plt.show()
    
    matplotlib.image.imsave(f'Compressed-{space_saved_percent:.2f}%.png', reconstructed_image)
