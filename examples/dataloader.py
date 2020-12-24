import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras import Model
import os
import pandas as pd
import numpy as np
import subprocess
import gc
import glob
from tensorflow.keras.applications import ResNet50


# key parameter assignment
EPOCH = 20
BATCH_SIZE = 16
PATIENCE = 5
DATA_PATH = '/kaggle/input/state-farm-distracted-driver-detection'


classes = [f'c{i}' for i in range(10)]
seed = 2020
validation_split = 0.2

driver_list = pd.read_csv(f'{DATA_PATH}/driver_imgs_list.csv')
drivers = np.unique(driver_list['subject'].values)

split = int(np.floor(validation_split * len(drivers)))
np.random.seed(seed)
trn_idx, val_idx = drivers[split:], drivers[:split]
print(f'train idx {trn_idx} \n val idx {val_idx}')


# mkdirs
split_dir = 'driver_split'
if not os.path.exists(split_dir):
    cmd = f'mkdir {split_dir}'
    subprocess.call(cmd, shell=True)
    for d in ['train', 'valid', 'test']:
        cmd = f'mkdir {split_dir}/{d}'
        subprocess.call(cmd, shell=True)
        if d == 'test':
            continue
        for cl in classes:
            cmd = f'mkdir {split_dir}/{d}/{cl}'
            subprocess.call(cmd, shell=True)
            
# ../driver_split/train/c0-c9
# ../driver_split/valid/c0-c9


# train and valid
trn_cnt = 0
val_cnt = 0
for i, driver_info in driver_list.iterrows():
    driver = driver_info['subject']
    label = driver_info['classname']
    img_path = driver_info['img']

    if driver in trn_idx:
        if not os.path.exists(f'{split_dir}/train/{label}/{img_path}'):
            os.symlink(os.path.abspath(f'{DATA_PATH}/imgs/train/{label}/{img_path}'), f'{split_dir}/train/{label}/{img_path}')
        trn_cnt += 1
    else:
        if not os.path.exists(f'{split_dir}/valid/{label}/{img_path}'):
            os.symlink(os.path.abspath(f'{DATA_PATH}/imgs/train/{label}/{img_path}'), f'{split_dir}/valid/{label}/{img_path}')
        val_cnt += 1
        
        
        

test_data_path = '/kaggle/working/driver_split/test/data'
if not os.path.exists(test_data_path):
    subprocess.call(f'mkdir {test_data_path}', shell=True)

cnt = 0

test_files = []
for file in glob.glob(f'{DATA_PATH}/imgs/test/*.jpg'):
    cnt += 1
    base_name = os.path.basename(file)
    if not os.path.exists(f'{test_data_path}/{base_name}'):
        os.symlink(file, f'{test_data_path}/{base_name}')
        test_files.append(base_name)

print(f'total {cnt} files linked')


train_dir = f'{split_dir}/train/'
val_dir = f'{split_dir}/valid/'
test_dir = '/kaggle/working/driver_split/test'


# tf.data.Dataset object

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir, 
                                                                    labels='inferred', 
                                                                    label_mode='categorical',
                                                                    batch_size=32,
                                                                    image_size=(224, 224))


val_dataset = tf.keras.preprocessing.image_dataset_from_directory(val_dir, 
                                                                  labels='inferred', 
                                                                  label_mode='categorical',
                                                                  batch_size=32,
                                                                  image_size=(224, 224))


test_dataset = tf.keras.preprocessing.image_dataset_from_directory(test_dir, 
                                                                  label_mode=None,
                                                                  batch_size=32,
                                                                  image_size=(224, 224))


norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1/255.)

norm_train_dataset = train_dataset.map(lambda x, y: (norm_layer(x), y))
norm_val_dataset = val_dataset.map(lambda x, y: (norm_layer(x), y))
norm_test_dataset = test_dataset.map(lambda x: norm_layer(x))

for b_X, b_y in norm_train_dataset:
    print('batch X shape', b_X.shape)
    print('batch y shape', b_y.shape)
    print(f'max {np.max(b_X[0])}  min {np.min(b_X[0])}')
    break
    

norm_train_dataset = norm_train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
norm_val_dataset = norm_val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


input_size = (224, 224, 3)

def get_model():
    model_res = ResNet50(include_top=True, input_shape=input_size, weights='imagenet')
    # take the last global average pooling with fewer parameters
    x = model_res.layers[-2].output
    
    x = Dense(2048)(x)
    x = Activation('relu')(x)
    x = Dropout(.5)(x)
    
    x = Dense(2048)(x)
    x = Activation('relu')(x)
    x = Dropout(.5)(x)
    
    x = Dense(10)(x)
    outputs = Activation('softmax')(x)

    model = Model(model_res.input, outputs)
    return model


model = get_model()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
checkpoint = tf.keras.callbacks.ModelCheckpoint('/tmp/checkpoint', monitor='val_accuracy', save_best_only=True)

model.fit(norm_train_dataset, validation_data=norm_val_dataset, epochs=20, callbacks=[callback, checkpoint])

# prediction
test_pred = model.predict(test_dataset)