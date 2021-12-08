from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras import backend as K
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import cv2
import os

def dynamic_unet_cnn(height,width,channels,num_layers = 4,starting_filter_size = 16, use_dropout = True): #Unet-cnn model 
    inputs = Input((height, width, channels))
    s = Lambda(lambda x: x/255)(inputs)

    for i in range(num_layers):
        if i == 0:
            curr_filter_size = starting_filter_size
            # print(curr_filter_size)

            conv = Conv2D(curr_filter_size, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (s)
            if use_dropout:
                conv = Dropout(0.1)(conv)
            conv = Conv2D(curr_filter_size, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(conv)
            pool = MaxPooling2D((2,2))(conv)

            conv_list = list([conv])
            pool_list = list([pool])

        else: 
            curr_filter_size = curr_filter_size*2
            # print(curr_filter_size)

            conv_list.append(Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool_list[i-1]))
            if use_dropout:
                conv_list[i] = Dropout(0.1) (conv_list[i])
            conv_list[i] = Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv_list[i])
            pool = MaxPooling2D((2, 2)) (conv_list[i])

            pool_list.append(pool)

    curr_filter_size = curr_filter_size*2
    # print(curr_filter_size)

    conv_list_reverse = conv_list.copy()
    conv_list_reverse.reverse()
    
    conv_list.append(Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pool_list[num_layers-1]))
    if use_dropout:
        conv_list[-1] = Dropout(0.3) (conv_list[-1])
    conv_list[-1] = Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv_list[-1])

    for i in range(num_layers):
        if i == 0:
            curr_filter_size = int(curr_filter_size/2)
            # print(curr_filter_size)

            u = Conv2DTranspose(curr_filter_size, (2, 2), strides=(2, 2), padding='same') (conv_list[-1])
            u = concatenate([u, conv_list_reverse[i]])
            conv_list.append(Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u))
            if use_dropout:
                conv_list[-1] = Dropout(0.1) (conv_list[-1])
            conv_list[-1] = Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv_list[-1])

            u_list = list([u])

        elif i == (num_layers-1):
            curr_filter_size = int(curr_filter_size/2)
            # print(curr_filter_size)

            u_list.append(Conv2DTranspose(curr_filter_size, (2, 2), strides=(2, 2), padding='same') (conv_list[-1]))
            u_list[i] = concatenate([u_list[i], conv_list_reverse[i]],axis=3)
            conv_list.append(Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u_list[i]))
            if use_dropout:
                conv_list[-1] = Dropout(0.1) (conv_list[-1])
            conv_list[-1] = Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv_list[-1])

        else: 
            curr_filter_size = int(curr_filter_size/2)
            # print(curr_filter_size)

            u_list.append(Conv2DTranspose(curr_filter_size, (2, 2), strides=(2, 2), padding='same') (conv_list[-1]))
            u_list[i] = concatenate([u_list[i], conv_list_reverse[i]])
            conv_list.append(Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u_list[i]))
            if use_dropout:
                conv_list[-1] = Dropout(0.1) (conv_list[-1])
            conv_list[-1] = Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv_list[-1])

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv_list[-1])
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return(model)

def plot_figures(image,pred_mask,num, orig_mask = None): #function for plotting figures

    if orig_mask is not None:
        plt.figure(num,figsize=(12,12))
        plt.subplot(131)
        plt.imshow(image)
        plt.title("Image")
        plt.subplot(132)
        plt.imshow(orig_mask.squeeze(),cmap='gray')
        plt.title("Original Mask")
        plt.subplot(133)
        plt.imshow(pred_mask.squeeze(),cmap='gray')
        plt.title('Predicted Mask')
    else:
        plt.figure(num)
        plt.subplot(121)
        plt.imshow(image)
        plt.title("Image")
        plt.subplot(122)
        plt.imshow(pred_mask.squeeze(),cmap='gray')
        plt.title('Predicted Mask')

def plot_acc_loss(results): #plot accuracy and loss
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
        
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

def data_generator(dataset, image_path, mask_path, height, width, channels): #function for generating data
    print('Loading in training data')
    X_train = np.zeros((len(dataset),height,width,channels), dtype = np.uint8) #initialize training sets (and testing sets)
    y_train = np.zeros((len(dataset),height,width,1), dtype = np.uint8)

    sys.stdout.flush() #write everything to buffer ontime 

    for i in tqdm(range(len(dataset)),total=len(dataset)): #iterate through datatset and build X_train,y_train

        new_image_path = os.path.join(image_path,dataset.iloc[i][0])
        new_mask_path = os.path.join(mask_path,dataset.iloc[i][1])

        if channels == 1:
            image = cv2.imread(new_image_path,0)
            image = np.expand_dims(image,axis = -1)
        else:
            image = cv2.imread(new_image_path)
        mask = cv2.imread(new_mask_path)[:,:,:1]

        img_resized = cv2.resize(image,(height,width))
        mask_resized = cv2.resize(mask,(height,width))

        # mask_resized = np.expand_dims(mask_resized,axis=2)

        img_resized = np.atleast_3d(img_resized)
        mask_resized = np.atleast_3d(mask_resized)

        # img_resized = resize(image,(height,width), mode = 'constant',preserve_range = True)
        # mask_resized = resize(mask, (height,width), mode = 'constant', preserve_range = True)

        X_train[i] = img_resized
        y_train[i] = mask_resized

    return X_train, y_train