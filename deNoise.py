import numpy as np
import cv2
from PIL import Image

for i in range(1,101):
    img = cv2.imread('D:/Atik/pythonScripts/WCNN/Dataset/cnnData/NoiseRed/OR/fakeFIG_{}.png'.format(i))
    b,g,r = cv2.split(img)           # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb

    # Denoising
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    b,g,r = cv2.split(dst)           # get b,g,r
    rgb_dst = cv2.merge([r,g,b])     # switch it to rgb
    
    im = Image.fromarray(rgb_dst)

    im.save('D:/Atik/pythonScripts/WCNN/Dataset/cnnData/NoiseRed/OR/fakeFIG_{}.png'.format(i))
    