import keras
from keras import backend as K
import numpy as np
import os
import matplotlib.pylab as plt
from PIL import Image as pil_image
from keras.preprocessing.image import ImageDataGenerator
import keras.layers as layers
import keras.models as models
from keras.initializers import orthogonal
import torchvision.transforms as transforms
import torch
from keras.optimizers import SGD, Adam


P_MODELSAVE = 'saved_models'
P_LOGS = 'logs'
P_IMGSAVE = 'denoised_images'

dirs = [P_MODELSAVE, P_LOGS, P_IMGSAVE]

for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)

dataset_path = '../data/ILSVRC2012_img_val/'
batch_size = 20
epochs = 150
input_shape = (224, 224)
noise_factor = 1
N = 104

# the path to save the weight of the model
saved_weight = os.path.join(P_MODELSAVE, 'dataweights.{epoch:02d}-{val_acc:.2f}.hdf5')

# For preprocessing since all images must have same shape
raw_walk_gen = os.walk(dataset_path + "raw/", topdown=True)
_, _, files_raw = next(raw_walk_gen)
files = [f for f in files_raw if f != ".gitignore"] # Remove .gitignore
sorted_files = sorted(files, key=lambda item: int(item[18:23]))
end = N if N != 0 else len(sorted_files)

data_gen_args = dict(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.5, 1.2],
    shear_range=0.01,
    horizontal_flip=True,
    rescale=1/255,
    fill_mode='reflect',
    data_format='channels_last')

data_flow_args = dict(
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='input') # Since we want to reconstruct the input

def preprocess():
    for _, name in enumerate(sorted_files[0:end]):
        orig_img = pil_image.open(dataset_path + 'raw/' + name)
            
        # Preprocessing only works for colour images
        if (orig_img.mode == "L"):
            orig_img = orig_img.convert(mode="RGB")

        mean = [ 0.485, 0.456, 0.406 ]
        std = [ 0.229, 0.224, 0.225 ]
        
        im = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224)])(orig_img)

        if (os.path.exists('../data/ILSVRC2012_img_val/autoencoder_train/') != 1):
            os.mkdir('../data/ILSVRC2012_img_val/autoencoder_train/')
        im.save('../data/ILSVRC2012_img_val/autoencoder_train/' + name, 'JPEG')
    

def random_crop(img, random_crop_size):
    width, height = img.size # PIL format
    dx, dy = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img.crop((x, y, x+dx, y+dy))


def load_img_extended(path, grayscale=False, color_mode='rgb', target_size=None,
                      interpolation='nearest'):
    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rbg", or "rgba"')
    
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            img = random_crop(img, width_height_tuple) # here comes the magic
    return img

# Overriding method
keras.preprocessing.image.image.load_img = load_img_extended

def Conv2DLayer(x, filters, kernel, strides, padding, block_id, kernel_init=orthogonal()):
    prefix = f'block_{block_id}_'
    x = layers.Conv2D(filters, kernel_size=kernel, strides=strides, padding=padding,
                      kernel_initializer=kernel_init, name=prefix+'conv')(x)
    x = layers.LeakyReLU(name=prefix+'lrelu')(x)
    x = layers.Dropout(0.2, name=prefix+'drop')((x))
    x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
    return x

def Transpose_Conv2D(x, filters, kernel, strides, padding, block_id, kernel_init=orthogonal()):
    prefix = f'block_{block_id}_'
    x = layers.Conv2DTranspose(filters, kernel_size=kernel, strides=strides, padding=padding,
                               kernel_initializer=kernel_init, name=prefix+'de-conv')(x)
    x = layers.LeakyReLU(name=prefix+'lrelu')(x)
    x = layers.Dropout(0.2, name=prefix+'drop')((x))
    x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
    return x

def AutoEncoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # 256 x 256
    conv1 = Conv2DLayer(inputs, 64, 3, strides=1, padding='same', block_id=1)
    conv2 = Conv2DLayer(conv1, 64, 3, strides=2, padding='same', block_id=2)
    
    # 128 x 128
    conv3 = Conv2DLayer(conv2, 128, 5, strides=2, padding='same', block_id=3)
    
    # 64 x 64
    conv4 = Conv2DLayer(conv3, 128, 3, strides=1, padding='same', block_id=4)
    conv5 = Conv2DLayer(conv4, 256, 5, strides=2, padding='same', block_id=5)
    
    # 32 x 32
    conv6 = Conv2DLayer(conv5, 512, 3, strides=2, padding='same', block_id=6)
    
    # 16 x 16
    deconv1 = Transpose_Conv2D(conv6, 512, 3, strides=2, padding='same', block_id=7)
    
    # 32 x 32
    skip1 = layers.concatenate([deconv1, conv5], name='skip1')
    conv7 = Conv2DLayer(skip1, 256, 3, strides=1, padding='same', block_id=8)
    deconv2 = Transpose_Conv2D(conv7, 128, 3, strides=2, padding='same', block_id=9)
    
    # 64 x 64
    skip2 = layers.concatenate([deconv2, conv3], name='skip2')
    conv8 = Conv2DLayer(skip2, 128, 5, strides=1, padding='same', block_id=10)
    deconv3 = Transpose_Conv2D(conv8, 64, 3, strides=2, padding='same', block_id=11)
    
    # 128 x 128
    skip3 = layers.concatenate([deconv3, conv2], name='skip3')
    conv9 = Conv2DLayer(skip3, 64, 5, strides=1, padding='same', block_id=12)
    deconv4 = Transpose_Conv2D(conv9, 64, 3, strides=2, padding='same', block_id=13)
    
    # 256 x 256
    skip3 = layers.concatenate([deconv4, conv1])
    conv10 = layers.Conv2D(3, 3, strides=1, padding='same', activation='sigmoid',
                       kernel_initializer=orthogonal(), name='final_conv')(skip3)

    
    return models.Model(inputs=inputs, outputs=conv10)


if __name__ == "__main__":
    preprocess()

    train_datagen = ImageDataGenerator(**data_gen_args)
    val_datagen = ImageDataGenerator(**data_gen_args)
    noisy_train_datagen = ImageDataGenerator(**data_gen_args)
    noisy_val_datagen = ImageDataGenerator(**data_gen_args)

    train_batches = train_datagen.flow_from_directory(
    dataset_path + '/autoencoder_train',
    **data_flow_args)

    val_batches = val_datagen.flow_from_directory(
    dataset_path + '/autoencoder_train',
    **data_flow_args)

    train_noisy_batches = noisy_train_datagen.flow_from_directory(dataset_path + '/pert_autoencoder', **data_flow_args)

    val_noisy_batches = noisy_val_datagen.flow_from_directory(dataset_path + '/pert_autoencoder', **data_flow_args)

    model = AutoEncoder((*input_shape, 3))
    # model_opt = SGD(lr=0.005, decay=1-0.995, momentum=0.7, nesterov=False)
    model_opt = Adam(lr=0.002)

    model.compile(optimizer=model_opt, loss='mse', metrics=['accuracy'])

    modelchk = keras.callbacks.ModelCheckpoint(saved_weight, 
                                      monitor='val_acc', 
                                      verbose=1,
                                      save_best_only=True, 
                                      save_weights_only=False,
                                      mode='auto',
                                      period=2)

    tensorboard = keras.callbacks.TensorBoard(log_dir=P_LOGS,
                                          histogram_freq=0,
                                          write_graph=True,
                                          write_images=True)

    csv_logger = keras.callbacks.CSVLogger(f'{P_LOGS}/keras_log.csv',
                                       append=True)
    
    model.fit(train_noisy_batches,
                    steps_per_epoch = train_batches.samples // batch_size,
                    epochs=epochs,
                    verbose=1, 
                    validation_data=val_noisy_batches)
