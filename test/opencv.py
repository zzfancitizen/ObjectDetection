import cv2

if __name__ == '__main__':
    img = cv2.imread('../jpg/IMG_0076_Copy.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    binary = cv2.resize(binary, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)

    cv2.imshow('1', binary)
    cv2.waitKey(0)
