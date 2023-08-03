# SPIES_code
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(784, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))
    return model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(latent_dim,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    model = tf.keras.Model(gan_input, gan_output)
    return model
latent_dim = 100
epochs = 1000
batch_size = 128
custom_dataset_dir = "D:/Personal/Competitions-Clubs-Internships-Programs/Samsung Prism/pics/data"
data_generator = ImageDataGenerator(rescale=1.0/255.0)
custom_dataset = data_generator.flow_from_directory(
    custom_dataset_dir,
    target_size=(28, 28),
    batch_size=batch_size,
    class_mode=None,  
    color_mode='grayscale'
)
generator = build_generator(latent_dim)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)
discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')
gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')

for epoch in range(epochs+1):
    generator_noise = tf.random.normal(shape=(batch_size, latent_dim))
    gan_noise = tf.random.normal(shape=(batch_size, latent_dim))

    generated_images = generator(generator_noise)
    real_images = next(custom_dataset)
    num_samples = min(real_images.shape[0], batch_size)
    real_images = real_images[:num_samples]
    generated_images = generated_images[:num_samples]

    combined_images = tf.concat([generated_images, real_images], axis=0)

    labels = tf.concat([tf.ones((num_samples, 1)), tf.zeros((num_samples, 1))], axis=0)
    labels += 0.05 * tf.random.uniform(labels.shape)

    discriminator.trainable = True
    discriminator_loss = discriminator.train_on_batch(combined_images, labels)

    gan_labels = tf.ones((batch_size, 1))
    discriminator.trainable = False
    gan_loss = gan.train_on_batch(gan_noise, gan_labels)

    discriminator.trainable = False
    gan_loss = gan.train_on_batch(generator_noise, gan_labels)

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}/{epochs}, Discriminator Loss: {discriminator_loss}, GAN Loss: {gan_loss}')
        plt.figure()
        plt.imshow(tf.reshape(generated_images[0], (28, 28)), cmap='gray')
        plt.title('Fake Image')
        plt.axis('off')
        plt.figure()
        plt.imshow(tf.reshape(real_images[0], (28, 28)), cmap='gray')
        plt.title('Real Image')
        plt.axis('off')

        plt.show()
