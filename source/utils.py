import cv2
import numpy as np


def mask_cell_areas(image, cell_mask, dilation_kernel_size):
    image_masked = image.copy()
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    cell_mask_dilated = cv2.dilate(cell_mask, kernel)
    for i in range(0, 3):
        image_masked[:, :, i][cell_mask_dilated == 0] = 0
    return image_masked, cell_mask_dilated


