import cv2


def LR_image(img,de_num=4):
    shape = img.shape

    img = cv2.resize(img,(shape[1]//de_num,shape[0]//de_num))
    img = cv2.resize(img,(shape[1],shape[0]))
    return img

def imread(path,type=None):
    return cv2.imread(path)
def imshow(img):
    cv2.imshow('',img)
    key = cv2.waitKey(-1)
    return key

def resize(img,scale):
    h,w,_ = img.shape
    shape = (int(w*scale),int(h*scale))
    return cv2.resize(img,shape)