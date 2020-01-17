# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 13:54:43 2020

@author: bened
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
#Fashon mnist:
#Labels: 0 Tshirt/Top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat
#       5 Sandal, 6 shirt, 7 sneakers, 8 bag, 9 Ankle boot
# src: https://jovianlin.io/datasets-within-keras/, https://keras.io/datasets/
#(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # normalize the images to [-1,1]

BUFFER_SIZE = 60000
BATCH_SIZE = 64

allowed_labels = [7]

#label filtering
filtered_images = []
filtered_labels = []

for i in range(len(train_labels)):
   if train_labels[i] in allowed_labels:
       filtered_images.append(train_images[i])
       filtered_labels.append(train_labels[i])
       
#train_images = filtered_images
#train_labels = filtered_labels  


# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#Generator
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)
    
    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7,7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model
    

generator = make_generator_model()

noise = tf.random.normal([1,100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()

#Discriminator

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)

print(decision)

#Loss and optimize
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

learning_rate = 1e-4
#learning_rate = 0.0002
#learning_rate = 0.001

generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 1000
noise_dim = 100
#How many images should be created for image saving
examples_width = 5
examples_height = 5

seed = tf.random.normal([examples_width * examples_height, noise_dim])

model_name = 'mnist_default_batch_64_e1-4'
img_dir = model_name + '_images'
model_save_dir = 'trained_generator'

folder_exist = os.path.isdir(img_dir)
if not folder_exist:
    os.makedirs(img_dir)
    print("Created folder ", img_dir)
#tf.function anotation causes to compile

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss =discriminator_loss(real_output, fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        
        for image_batch in dataset:
            train_step(image_batch)
        
        # Images for GIf
        display.clear_output(wait=True)
        #generate_and_save_images(generator, epoch + 1, seed)
        
        #Save model every 15 epoch
        #if (epoch + 1) % 15 == 0:
        #checkpoint.save(file_prefix = checkpoint_prefix)
        #if (epoch + 1) % 10 == 0:
        generate_and_save_images(generator, epoch + 1, seed)
        print('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start))
        
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)
    checkpoint.save(file_prefix = checkpoint_prefix)
    
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(examples_width, examples_height))
    
    for i in range(predictions.shape[0]):
        plt.subplot(examples_width, examples_height,i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray_r')
        plt.axis('off')
    
    plt.savefig(img_dir+'/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
        
train(train_dataset, EPOCHS)
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def display_image(epoch_no):
    return PIL.Image.open(img_dir+'/image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

anim_file = 'dcgan.gif'

folder_exist = os.path.isdir(model_save_dir)
if not folder_exist:
    os.makedirs(model_save_dir)
    print("Created folder ", img_dir)


generator.save(model_save_dir + '/' + model_name + '.h5')
print('Model saved in: ' + model_save_dir + '/' + model_name + '.h5')

with imageio.get_writer(img_dir+'/'+anim_file, mode='I') as writer:
    filenames = glob.glob(img_dir+'/image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2 * (i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

import IPython

if IPython.version_info > (6,2,0,''):
    display.Image(filename=img_dir+'/'+anim_file)