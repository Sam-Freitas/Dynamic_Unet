import os
from posixpath import join
import cv2
import numpy as np 
import pandas as pd 
import PIL
from PIL import Image
import glob
import shutil
from random import shuffle

import matplotlib.pyplot as plt 

def noisy(noise_typ,image,mean,var):
    if noise_typ == "gauss":
        sigma = var ** 0.5

        img_shape = image.shape

        gaussian = np.random.normal(mean, sigma, img_shape)

        noisy_image = np.zeros(image.shape, np.float32)

        if len(img_shape) == 2:
            noisy_image = image + gaussian
        else:
            noisy_image[:, :, 0] = image[:, :, 0] + gaussian[:, :, 0]
            noisy_image[:, :, 1] = image[:, :, 1] + gaussian[:, :, 1]
            noisy_image[:, :, 2] = image[:, :, 2] + gaussian[:, :, 2] 

        noisy_image[noisy_image<0] = 0
        noisy_image[noisy_image>255] = 255

        return noisy_image
    elif noise_typ == "none":
        return image

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

# get pwd
curr_path = os.getcwd()
# get path to exp images
data_path = os.path.join(curr_path,"Lightsaver_NN")

image_dir_path = os.path.join(curr_path, "images")
mask_dir_path = os.path.join(curr_path,"masks")
test_dir_path = os.path.join(curr_path,"testing")

try:
    os.mkdir(image_dir_path)
    os.mkdir(mask_dir_path)
    os.mkdir(test_dir_path)
except:
    shutil.rmtree(image_dir_path)
    shutil.rmtree(mask_dir_path)
    shutil.rmtree(test_dir_path)
    os.mkdir(image_dir_path)
    os.mkdir(mask_dir_path)
    os.mkdir(test_dir_path)

data_list = glob.glob(os.path.join(data_path,"*.png"))

data_list_rand = data_list

shuffle(data_list_rand)

noise_list = ["gauss","none"]

df = pd.DataFrame(columns=['images','masks'])

for i,this_path in enumerate(data_list_rand):

    this_img = cv2.imread(this_path)

    img_width = this_img.shape[0]

    color_image = this_img[0:img_width, 0:img_width]

    color_mask = this_img[0:img_width, img_width:img_width+img_width]

    gray_mask = ((rgb2gray(color_mask) > 0)*255 ).astype(np.uint8)

    for j,noise_type in enumerate(noise_list):

        this_img_name = str(i) + '_' + str(j) + '.png'
        this_mask_name = str(i) + '_' + str(j) + '_mask.png'

        this_img_path = os.path.join(image_dir_path,this_img_name)
        this_mask_path = os.path.join(mask_dir_path,this_mask_name)

        df = df.append({'images' : this_img_name, 'masks' : this_mask_name}, 
                    ignore_index = True)

        # "s&p" "speckle" "poisson": "guass"
        noise_img = noisy(noise_type,color_image,2,100).astype('uint8')

        cv2.imwrite(this_img_path,noise_img)
        cv2.imwrite(this_mask_path,gray_mask)
    
    print(i)

    if not (i%10):

        this_test_path = os.path.join(test_dir_path,str(i) + '.png')
        cv2.imwrite(this_test_path,color_image)
        
        zero_mask = np.zeros((img_width,img_width,3))
        noise_img = noisy("gauss",zero_mask,128,128).astype('uint8')

        cv2.imwrite(this_img_path,noise_img)
        cv2.imwrite(this_mask_path,zero_mask)

    if not (i%100):
        
        this_test_path = os.path.join(test_dir_path,str(i) + '.png')
        cv2.imwrite(this_test_path,color_image)

        zero_mask = np.zeros((img_width,img_width,3))

        cv2.imwrite(this_img_path,zero_mask)
        cv2.imwrite(this_mask_path,zero_mask)


df.to_csv('dataset.csv',index = False)

print("End of script")