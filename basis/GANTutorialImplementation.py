# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:08:56 2020

@author: bened
"""
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
tf.disable_v2_behavior()
#import Mnist
import input_data
mnist = input_data.read_data_sets("data\\", one_hot=True)

image_dim = 784
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100

gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

def glorot_init(shape):
    std_dev = 1./tf.sqrt(shape[0] / 2.)
    return tf.random_normal(shape=shape, stddev=std_dev)

#Store Layers weight & bias for discriminator and generator
weights = {
        'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
        'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
        
        'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim])),
        'disc_out': tf.Variable(glorot_init([gen_hidden_dim, 1])),
        }

biases = {
        'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
        'gen_out': tf.Variable(tf.zeros([image_dim])),
        
        'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
        'disc_out': tf.Variable(tf.zeros([1])),
        
        }


disc_real = discriminator(disc_input)

disc_fake = discriminator(generator(gen_input))

gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss= -tf.reduce_mean(tf.log(disc_real) + tf.log(1. -disc_fake))

#Build optimizers

learning_rate = 0.0002

optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

gen_vars = [weights['gen_hidden1'], weights['gen_out'],
            biases['gen_hidden1'], biases['gen_out']]
    
disc_vars = [weights['disc_hidden1'], weights['disc_out'],
             biases['disc_hidden1'], biases['disc_out']]

train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

#Training
num_steps = 70000
batch_size = 128

sess = tf.Session()
sess.run(tf.global_variables_initializer())

g_loss = []
d_loss = []

for i in range(1, num_steps+1):
    
    batch_x, _ = mnist.train.next_batch(batch_size)
    
    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
    
    feed_dict = {disc_input: batch_x, gen_input: z}
    _, _, gl, dl = sess.run([
                train_gen,
                train_disc,
                gen_loss,
                disc_loss],
                feed_dict = feed_dict)
    
    g_loss.append(gl)
    d_loss.append(dl)
    
    if i % 2000 == 0 or i == 1:
        print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
    
print('Training finished')

#Visualization
plt.plot(g_loss, label='Loss of the Generator')
plt.plot(d_loss, label='Loss of the Discriminator')
plt.xlabel('Iterations')
plt.ylabel('loss')
plt.legend()
plt.show()

#Testing
#Generate images from a latent random variable, using the generator network.
#firs create an empty array
canvas = np.empty((28,28))

#Generate the latent random Variable drawing from uniform distribution in [-1, 1]
#as a 1x100 tensor
z = np.random.uniform(-1., 1., size=[1, noise_dim])

#will feed z as input to generator

g = sess.run(generator(gen_input), feed_dict={gen_input: z})

# The generated sample is reshaped and visualized
canvas[0:28, 0:28] = g[0].reshape([28,28])

plt.figure()
plt.imshow(canvas, cmap="gray")
plt.show()

# Generate n*n image from noise, using the generator network.
n = 15
canvas = np.empty((28 * n, 28 * n))
for i in range(n):
    z = np.random.uniform(-1., 1., size=[n, noise_dim])
    #Generate image from noise.
    g = sess.run(generator(gen_input), feed_dict={gen_input: z})
    
    for j in range(n):
        #Draw the generated digits
        canvas[i * 28:(i+1) * 28, j * 28:(j+1) * 28] = g[j].reshape([28,28])
        
plt.figure(figsize=(n,n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()