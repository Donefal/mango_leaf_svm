import cv2

def load_and_resize(image_path, size=(120, 150)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

