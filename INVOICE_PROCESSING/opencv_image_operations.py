import cv2
import numpy as np
from whitening import whiten
import PIL.Image


def resize_with_ratio(image, resize_ratio):
    height, width, color = image.shape
    resize = cv2.resize(image, (int(width * resize_ratio), int(height * resize_ratio)))
    return resize


def draw_rectangle_text_with_ratio(resize, left, top, width, height, text, resize_ratio):
    resize = cv2.rectangle(resize, (int(left * resize_ratio), int(top * resize_ratio)),
                           (int((left + width) * resize_ratio),
                            int((top + height) * resize_ratio)),
                           (255, 0, 0), 2)

    resize = cv2.putText(resize, str(text),
                         (int(left * resize_ratio), int(top * resize_ratio)),
                         cv2.FONT_HERSHEY_TRIPLEX, resize_ratio, (0, 0, 255), 1)
    return resize


def pre_process_images_before_scanning(image):
    """
    return a colored image, preprocessed
    :param image:
    :return:
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image_blur = cv2.medianBlur(image_gray, 5)
    kernel = np.ones((1, 1), np.uint8)
    image_blur = cv2.erode(image_gray, kernel, iterations=1)
    image_blur = cv2.cvtColor(image_blur, cv2.COLOR_GRAY2BGR)
    # whiten images
    image_foreground, image_background = whiten(image_blur, kernel_size=20, downsample=4)
    return PIL.Image.fromarray(image_foreground)