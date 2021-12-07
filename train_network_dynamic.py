import numpy as np #import needed libraries and commands
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from cv2 import imread
import os
import sys
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

from dynamic_unet import dynamic_unet_cnn, plot_figures, plot_acc_loss, data_generator

plt.ion() #turn ploting on

dataset_path = os.getcwd()
image_path = os.path.join(dataset_path, "images")
mask_path = os.path.join(dataset_path,"masks")
dataset = pd.read_csv('dataset.csv')
 
total = len(dataset) #set variables
test_split = 0.2
height = 512
width = 512
channels = 3 
batch_size = 32

num_layers_of_unet = 4
starting_kernal_size = 16

model = dynamic_unet_cnn(height,width,channels,
    num_layers = num_layers_of_unet,starting_filter_size = starting_kernal_size, use_dropout = False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly = True)
# model.summary() #display model summary

try:
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_dtype=False,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    )
except:
    print("Exporting model to png failed")
    print("Necessary packages: pydot (pip) and graphviz (brew)")

#######Training
train, test = train_test_split(dataset, test_size = test_split, random_state = 50) #randomly split up the test and training datasets
X_train, y_train = data_generator(train, image_path, mask_path, height, width) #set up training data
y_train = y_train / 255 #thresh y_training set

model_path = "lightsaver_weights.h5" #store model here
checkpoint = ModelCheckpoint(model_path,monitor="val_loss",mode="min",save_best_only = True,verbose=1) #use checkpoint instead of sequential() module
earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0.01, patience = 5, verbose = 1,restore_best_weights = True) #stop at best epoch
results = model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=100,callbacks=[earlystop, checkpoint]) #fit model

plot_acc_loss(results) #plot the accuracy and loss functions

model = load_model('lightsaver_weights.h5') #load weights
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1) 
preds_train_t = (preds_train > 0.5).astype(np.uint8) #predict mask
ix = random.randint(1, 10)
plot_figures(X_train[ix],y_train[ix],preds_train[ix], 1) #plot images and masks

#######Testing

#model = load_model("lightsaver_weights.h5") #reload model for testing
X_test,y_test = data_generator(test,image_path, mask_path,height,width) #get test set
y_test = y_test / 255 #thresh y_test
results = model.evaluate(X_test,y_test,steps=1) #get evaluation results

count = 1 #counter for figures in for loops
for image,mask in zip(X_test,y_test): #for loop for plotting images
    
    img = image.reshape((1,height,width,channels)).astype(np.uint8)
    pred_mask = model.predict(img)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    plot_figures(image,mask,pred_mask, count)
    count += 1

    if count>20:
        break

plt.ioff()
plt.show()
