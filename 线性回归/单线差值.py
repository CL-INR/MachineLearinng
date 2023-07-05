import numpy as np
import cv2


images = 'lenna.png'


def ToGray(images):
    img = cv2.imread(images)
    h, w = img.shape[:2]
    img_gray = np.zeros([h, w], img.dtype)
    img_black = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
    print('img_gray:', img_gray)
    cv2.imshow('Gray', img_gray)
    cv2.imwrite("Gray.png", img_gray)
    rows, cols = img_gray.shape
    for i in range(rows):
        for j in range(cols):
            if img_gray[i, j]/255 <= 0.5:
                img_black[i, j] = 0
            else:
                img_black[i, j] = 255
    print(img_black)
    cv2.imshow('Black and white', img_black)
    cv2.imwrite('Black and white.png', img_black)
    cv2.waitKey()


if __name__ == '__main__':
    ToGray(images)
