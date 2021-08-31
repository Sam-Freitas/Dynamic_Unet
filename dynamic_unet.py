from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf

def dynamic_unet_cnn(height,width,channels,num_layers,starting_filter_size): #Unet-cnn model 
    inputs = Input((height, width, channels))
    s = Lambda(lambda x: x/255)(inputs)



    for i in range(num_layers):
        if i == 0:
            curr_filter_size = starting_filter_size
            # print(curr_filter_size)

            conv = Conv2D(curr_filter_size, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (s)
            conv = Dropout(0.1)(conv)
            conv = Conv2D(curr_filter_size, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(conv)
            pool = MaxPooling2D((2,2))(conv)

            conv_list = list([conv])
            pool_list = list([pool])

        else: 
            curr_filter_size = curr_filter_size*2
            # print(curr_filter_size)

            conv_list.append(Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool_list[i-1]))
            conv_list[i] = Dropout(0.1) (conv_list[i])
            conv_list[i] = Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv_list[i])
            pool = MaxPooling2D((2, 2)) (conv_list[i])

            pool_list.append(pool)

    curr_filter_size = curr_filter_size*2
    # print(curr_filter_size)

    conv_list_reverse = conv_list.copy()
    conv_list_reverse.reverse()
    
    conv_list.append(Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pool_list[num_layers-1]))
    conv_list[-1] = Dropout(0.3) (conv_list[-1])
    conv_list[-1] = Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv_list[-1])

    for i in range(num_layers):
        if i == 0:
            curr_filter_size = int(curr_filter_size/2)
            # print(curr_filter_size)

            u = Conv2DTranspose(curr_filter_size, (2, 2), strides=(2, 2), padding='same') (conv_list[-1])
            u = concatenate([u, conv_list_reverse[i]])
            conv_list.append(Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u))
            conv_list[-1] = Dropout(0.1) (conv_list[-1])
            conv_list[-1] = Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv_list[-1])

            u_list = list([u])

        elif i == (num_layers-1):
            curr_filter_size = int(curr_filter_size/2)
            # print(curr_filter_size)

            u_list.append(Conv2DTranspose(curr_filter_size, (2, 2), strides=(2, 2), padding='same') (conv_list[-1]))
            u_list[i] = concatenate([u_list[i], conv_list_reverse[i]],axis=3)
            conv_list.append(Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u_list[i]))
            conv_list[-1] = Dropout(0.1) (conv_list[-1])
            conv_list[-1] = Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv_list[-1])

        else: 
            curr_filter_size = int(curr_filter_size/2)
            # print(curr_filter_size)

            u_list.append(Conv2DTranspose(curr_filter_size, (2, 2), strides=(2, 2), padding='same') (conv_list[-1]))
            u_list[i] = concatenate([u_list[i], conv_list_reverse[i]])
            conv_list.append(Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u_list[i]))
            conv_list[-1] = Dropout(0.1) (conv_list[-1])
            conv_list[-1] = Conv2D(curr_filter_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv_list[-1])

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv_list[-1])
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return(model)