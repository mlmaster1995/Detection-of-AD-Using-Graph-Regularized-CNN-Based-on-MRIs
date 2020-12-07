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


class VAE(tf.keras.models.Model):
    def __init__(self, latent_dim, channels, optimizer, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self._latent_dim = latent_dim
        self._channels = channels
        self._optimizer = optimizer
        self.encoder = self.encoder2D()
        self.decoder = self.decoder2D()

    def _reparameterize(self, arg):
        mean, logvar = arg
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + eps * tf.exp(logvar * .5)

    def encoder2D(self):
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

    @tf.function
    def _train_step(self, data):
        """train step function"""
        data = data['image_tensor']
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(data, reconstruction))
            kl_loss = -5e-4 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            # kl_loss = -5e-1 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self._optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def fit(self, image_dataset, checkpoints_path, epochs=100):
        epochs = epochs
        for epoch in range(epochs):
            total_loss = []
            kl_loss = []
            reconstruction_loss = []
            epoch += 1
            start_time = time.time()
            for step, batch_image in enumerate(image_dataset):
                loss = self._train_step(batch_image)
                total_loss.append(loss['loss'])
                kl_loss.append(loss['kl_loss'])
                reconstruction_loss.append(loss['reconstruction_loss'])
            self.save_weights(checkpoints_path)
            epoch_total_loss = tf.reduce_mean(tf.convert_to_tensor(total_loss, dtype=tf.float32))
            epoch_kl_loss = tf.reduce_mean(tf.convert_to_tensor(kl_loss, dtype=tf.float32))
            epoch_reconstruction_loss = tf.reduce_mean(tf.convert_to_tensor(reconstruction_loss, dtype=tf.float32))
            end_time = time.time()
            epoch_time = end_time - start_time

            print('epoch:%3s, epoch_time: %2.5f, total_loss: %2.5f, kl_loss: %2.5f, reconstruction_loss: %2.5f' %
                  (epoch, epoch_time, epoch_total_loss, epoch_kl_loss, epoch_reconstruction_loss))
