import numpy as np 
import cv2 
import skimage

# input : image (size = (224 * 224))
# output: list image from 


def agumentation(img):

    img_gen = []
    img = cv2.resize(img, (224,224))
    

    # Flip
    img_gen.append(np.fliplr(img))

    #Smooth
    img_gen.append(cv2.medianBlur(img, 3))
    img_gen.append( cv2.blur(img,(5,5)) )
    img_gen.append(cv2.GaussianBlur(img,(5,5),0))

    #Rotation: 90, 180, 270
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    
    angle90 = 90
    angle180 = 180
    angle270 = 270


    scale = 1.0
    
    # Perform the counter clockwise rotation holding at the center
    # 90 degrees

    M = cv2.getRotationMatrix2D(center, angle90, scale)
    rotated90 = cv2.warpAffine(img, M, (h, w))
    
    img_gen.append(rotated90)
    # 180 degrees
    M = cv2.getRotationMatrix2D(center, angle180, scale)
    rotated180 = cv2.warpAffine(img, M, (w, h))
    img_gen.append(rotated180)

    # 270 degrees
    M = cv2.getRotationMatrix2D(center, angle270, scale)
    rotated270 = cv2.warpAffine(img, M, (h, w))
    img_gen.append(rotated270)

    # Scale
    # scale_out = skimage.transform.rescale(img, scale=2.0, mode='constant')
    # scale_out2 = skimage.transform.rescale(img, scale=3.0, mode='constant')
    # scale_in = skimage.transform.rescale(img, scale=0.5, mode='constant')
    # scale_in2 = skimage.transform.rescale(img, scale=0.7, mode='constant')
    # cv2.imshow('no0', img)
    # cv2.imshow('no',scale_out)
    # cv2.waitKey(0)
    # img_gen.append(scale_out)
    # img_gen.append(scale_in)
    # img_gen.append(scale_in2)
    # img_gen.append(scale_out2)

    return img_gen

# img = cv2.imread("test.jpg")

# list_img = agumentation(img)
# from imutils import build_montages

# print(len(list_img))
# montage = build_montages(list_img, (224, 224), (5, 5))[0]
	
# # show the output montage
# title = "Result"
# cv2.imshow(title, montage)
# cv2.waitKey(0)



