"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""
import tensorflow as tf
import time


class AAE(tf.keras.models.Model):
    """
    source: "Adversarial Autoencoders" A.Makhzani 2015 and it's also cited in the project paper
    paper link: https://arxiv.org/abs/1511.05644
    """
    def __init__(self, latent_dim, image_channels, aae_optimizer=None, **kwargs):
        """
        Args:
            latent_dim: the dimension of the MRI representation.
            image_channels: the channels of MRI images and it could be 1 or 3 channels.
            aae_optimizer: any optimizer defined in tf.keras.optimizers.
        Return:
            N/A
        """
        super(AAE, self).__init__(**kwargs)
        self._optimizer = aae_optimizer
        self._latent_dim = latent_dim
        self._channels = image_channels
        self.encoder = self.encoder2D()
        self.decoder = self.decoder2D()
        self.discriminator = self.discriminator_model()

    def _reparameterize(self, arg):
        mean, logvar = arg
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + eps * tf.exp(logvar * .5)

    def encoder2D(self):
        """
        The method to define an encoder of VAE and it's also a generator of GAN
        """
        encoder_ipt = tf.keras.Input(shape=(100, 100, self._channels), name='image_tensor')
        opt = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', name='Cov2D_1')(encoder_ipt)
        opt = tf.keras.layers.LeakyReLU(name='Cov2D_1_leakyRelu')(opt)
        opt = tf.keras.layers.MaxPooling2D()(opt)
        opt = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', name='Cov2D_2')(opt)
        opt = tf.keras.layers.LeakyReLU(name='Cov2D_2_leakyRelu')(opt)
        opt = tf.keras.layers.MaxPooling2D()(opt)
        opt = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', name='Cov2D_3')(opt)
        opt = tf.keras.layers.LeakyReLU(name='Cov2D_3_leakyRelu')(opt)
        opt = tf.keras.layers.MaxPooling2D()(opt)
        opt = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', name='Cov2D_4')(opt)
        opt = tf.keras.layers.LeakyReLU(name='Cov2D_4_leakyRelu')(opt)
        opt = tf.keras.layers.MaxPooling2D()(opt)
        opt = tf.keras.layers.Flatten()(opt)
        opt = tf.keras.layers.Dense(64)(opt)
        opt = tf.keras.layers.LeakyReLU(name='Dense_leakyRelu')(opt)
        mean = tf.keras.layers.Dense(self._latent_dim, name='mean')(opt)
        mean = tf.keras.layers.LeakyReLU(name='mean_leakyRelu')(mean)
        logvar = tf.keras.layers.Dense(self._latent_dim, name='logvar')(opt)
        logvar = tf.keras.layers.LeakyReLU(name='logvar_leakyRelu')(logvar)
        rep = tf.keras.layers.Lambda(self._reparameterize, name='compressed_data')((mean, logvar))
        encoder_model = tf.keras.models.Model(encoder_ipt, [mean, logvar, rep], name='encoder')
        return encoder_model

    def decoder2D(self):
        """
        The method to define an decoder
        """
        decoder_ipt = tf.keras.Input(shape=(self._latent_dim,))
        opt = tf.keras.layers.Dense(64, name='decoder_dense_1')(decoder_ipt)
        opt = tf.keras.layers.LeakyReLU(name='decoder_dense_leakyRelu_1')(opt)
        opt = tf.keras.layers.Dense(128, name='decoder_dense_2')(opt)
        opt = tf.keras.layers.LeakyReLU(name='decoder_dense_leakyRelu_2')(opt)
        opt = tf.keras.layers.Dense(14 * 14 * self._channels, name='decoder_dense_3')(opt)
        opt = tf.keras.layers.LeakyReLU(name='decoder_dense_leakyRelu_3')(opt)
        opt = tf.keras.layers.Reshape(target_shape=(14, 14, self._channels))(opt)
        opt = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same', name='Cov2DT_1')(
            opt)
        opt = tf.keras.layers.LeakyReLU(name='Cov2DT_1_leakyRelu')(opt)
        opt = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', name='Cov2DT_2')(
            opt)
        opt = tf.keras.layers.LeakyReLU(name='Cov2DT_2_leakyRelu')(opt)
        opt = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', name='Cov2DT_3')(
            opt)
        opt = tf.keras.layers.LeakyReLU(name='Cov2DT_3_leakyRelu')(opt)
        opt = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=1, padding='valid', name='Cov2D_5')(opt)
        opt = tf.keras.layers.LeakyReLU(name='Cov2D_5_leakyRelu')(opt)
        opt = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=1, padding='valid', name='Cov2D_6')(opt)
        opt = tf.keras.layers.LeakyReLU(name='Cov2D_6_leakyRelu')(opt)
        opt = tf.keras.layers.Conv2DTranspose(filters=self._channels, kernel_size=3, strides=2,
                                              padding='same',
                                              activation='sigmoid',
                                              name='Cov2DT_7')(opt)
        decoder_model = tf.keras.models.Model(decoder_ipt, opt, name='decoder')
        return decoder_model

    def discriminator_model(self):
        """
        The method to define a discriminator for GAN
        """
        discriminator_ipt = tf.keras.Input(shape=(self._latent_dim,))
        opt = tf.keras.layers.Dense(128)(discriminator_ipt)
        opt = tf.keras.layers.LeakyReLU(name='Dense_leakyRelu_0')(opt)

        opt = tf.keras.layers.Dense(64)(opt)
        opt = tf.keras.layers.LeakyReLU(name='Dense_leakyRelu_1')(opt)

        opt = tf.keras.layers.Dense(32)(opt)
        opt = tf.keras.layers.LeakyReLU(name='Dense_leakyRelu_2')(opt)

        opt = tf.keras.layers.Dense(1, activation='sigmoid')(opt)
        model = tf.keras.models.Model(discriminator_ipt, opt, name='discriminator')
        return model

    @tf.function
    def _train_step(self, data, batch_size, noise_mean, noise_std):
        data = data['image_tensor']
        noise = tf.random.normal(shape=(batch_size, self._latent_dim), mean=noise_mean, stddev=noise_std)
        # train gan discriminator with discriminator loss
        with tf.GradientTape() as tapeI:
            self.encoder.trainable = False
            self.decoder.trainable = False
            self.discriminator.trainable = True
            real_output = self.discriminator(noise)
            fake_output = self.discriminator(self.encoder(data)[2])
            discriminator_loss = -tf.reduce_mean(tf.math.log(real_output + 1e-4) + tf.math.log(1 - fake_output) + 1e-4)
        grads = tapeI.gradient(discriminator_loss, self.trainable_weights)
        self._optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # train gan generator with discriminator & encoder loss
        with tf.GradientTape() as tapeII:
            self.encoder.trainable = True
            self.decoder.trainable = False
            self.discriminator.trainable = False
            fake_output = self.discriminator(self.encoder(data)[2])
            generator_loss = -tf.reduce_mean(tf.math.log(fake_output))
        grads = tapeII.gradient(generator_loss, self.trainable_weights)
        self._optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # train decoder & encoder with reconstruction loss
        with tf.GradientTape() as tapeIII:
            self.encoder.trainable = True
            self.decoder.trainable = True
            self.discriminator.trainable = False
            reconstruction = self.decoder(self.encoder(data)[2])
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(data, reconstruction))
        grads = tapeIII.gradient(reconstruction_loss, self.trainable_weights)
        self._optimizer.apply_gradients(zip(grads, self.trainable_weights))

        total_loss = discriminator_loss + generator_loss + reconstruction_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "discriminator_loss": discriminator_loss,
            "generator_loss": generator_loss,
        }

    def fit(self, image_dataset, checkpoints_path, batch_size=1, real_mean=0, real_std=1, epochs=100):
        """
        Args:
            image_dataset: 4-channel MRI image tensors
            checkpoints_path: Path to save model weights per traning epoch
            batch_size: The MRI image tensor batch
            real_mean: The mean value of the noise distribution
            real_std: The standard deviation value of the noise distribution
            epochs: Training epochs
        Return:
            N/A
        """
        epochs = epochs
        for epoch in range(epochs):
            total_loss = []
            discriminator_loss = []
            generator_loss = []
            reconstruction_loss = []
            epoch += 1
            start_time = time.time()
            for step, batch_image in enumerate(image_dataset):
                loss = self._train_step(batch_image, batch_size, real_mean, real_std)
                total_loss.append(loss['loss'])
                discriminator_loss.append(loss['discriminator_loss'])
                generator_loss.append(loss['generator_loss'])
                reconstruction_loss.append(loss['reconstruction_loss'])
            self.save_weights(checkpoints_path)
            epoch_total_loss = tf.reduce_mean(tf.convert_to_tensor(total_loss, dtype=tf.float32))
            epoch_discriminator_loss = tf.reduce_mean(tf.convert_to_tensor(discriminator_loss, dtype=tf.float32))
            epoch_generator_loss = tf.reduce_mean(tf.convert_to_tensor(generator_loss, dtype=tf.float32))
            epoch_reconstruction_loss = tf.reduce_mean(tf.convert_to_tensor(reconstruction_loss, dtype=tf.float32))
            end_time = time.time()
            epoch_time = end_time - start_time

            print(
                'epoch:%3s, epoch_time: %2.5f, total_loss: %2.5f, discriminator_loss: %2.5f, generator_loss: %2.5f, '
                'reconstruction_loss: %2.5f' %
                (epoch, epoch_time, epoch_total_loss, epoch_discriminator_loss, epoch_generator_loss,
                 epoch_reconstruction_loss))
