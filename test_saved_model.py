# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:12:33 2020

@author: br-182508
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import os

model_load_dir = 'trained_generator'
model_name = 'mnist_tuned_batch_256_lr_001'

img_dir = 'generator_test'

examples_width = 10
examples_height = 10
noise_dim = 100

def generate_and_save_images(model, img_name, test_input, save='True'):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(examples_width, examples_height))
    
    for i in range(predictions.shape[0]):
        plt.subplot(examples_width, examples_height,i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray_r')
        plt.axis('off')
    
    folder_exist = os.path.isdir(img_dir)
    
    if not folder_exist:
        os.makedirs(img_dir)
        print("Created folder ", img_dir)
    
    if(save):
        plt.savefig(img_dir+'/' + img_name + '.png')
        print('Generated image saved in:' + img_dir + '/' + img_name + '.png')
    plt.show()

model = tf.keras.models.load_model(model_load_dir + '/' + model_name + '.h5')

model.summary()

seed = tf.random.normal([examples_width * examples_height, noise_dim])

generate_and_save_images(model, model_name, seed, save=True)