import cv2
import numpy as np


def convolution(image, kernel):
    image_width, image_height = image.shape[:2]
    kernel_width, kernel_height = kernel.shape[:2]

    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    result_image = np.zeros(image.shape, dtype=np.float32)
    for x in range(image_width):
        for y in range(image_height):
            image_patch = padded_image[x: x + kernel_width, y: y + kernel_height]
            result_image[x, y] = np.sum(image_patch * kernel)

    return result_image


def convolution_av(image, kernel, value):
    image_width, image_height = image.shape[:2]
    kernel_width, kernel_height = kernel.shape[:2]

    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    result_image = np.zeros(image.shape, dtype=np.float32)
    for x in range(image_width):
        for y in range(image_height):
            image_patch = padded_image[x: x + kernel_width, y: y + kernel_height]
            result_image[x, y] = (np.sum(image_patch * kernel)) / value

    return result_image


def average(image, kernel):
    return image / np.sum(kernel)


def gradient_mag(gx, gy):
    gx_square = np.square(gx)
    gy_square = np.square(gy)
    return np.sqrt(gx_square + gy_square)


def angle(gx, gy):
    return np.arctan2(gy, gx)


def logarithm(image):
    epsilon = 1e-6
    return np.log10(image + epsilon) / 20


if __name__ == "__main__":
    # Read input image
    image_path = "path/to/image"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # edge detection
    kernel_prewitt = np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]])
    kernel_sobelX = np.array([[1, 2, 1],
                              [0, 0, 0],
                              [-1, -2, -1]])

    kernel_sobelY = np.array([[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]])

    kernel_average_blur = np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]])
    kernel_gaussian_blur = np.array([[1, 2, 1],
                                     [2, 10, 2],
                                     [1, 2, 1]])
    kernel_weight_lpf = np.array([[1 / 16, 1 / 8, 1 / 16],
                                  [1 / 8, 1 / 2, 1 / 8],
                                  [1 / 16, 1 / 8, 1 / 16]])

    resultimage_perwitt = convolution_av(image, kernel_prewitt, 100)
    resultimage_sobelX = convolution_av(image, kernel_sobelX, 100)
    resultimage_sobelY = convolution_av(image, kernel_sobelY, 100)
    resultimage_average_blur = convolution_av(image, kernel_average_blur, 2000)
    resultimage_gaussian_blur = convolution_av(image, kernel_gaussian_blur, 4000)
    sharpimage = np.absolute((resultimage_average_blur - image) / 200)

    resultimage_lpf = convolution_av(image, kernel_weight_lpf, 300)

    cv2.imshow('Input Image', image)
    cv2.imshow('prewitt', resultimage_perwitt)
    cv2.imshow('sobelx', resultimage_sobelX)
    cv2.imshow('sobely', resultimage_sobelY)
    cv2.imshow('averageblur', resultimage_average_blur)
    cv2.imshow('sharp', sharpimage)
    cv2.imshow('gaussian', resultimage_gaussian_blur)
    cv2.imshow('lpf', resultimage_lpf)

cv2.waitKey(0)
cv2.destroyAllWindows()
