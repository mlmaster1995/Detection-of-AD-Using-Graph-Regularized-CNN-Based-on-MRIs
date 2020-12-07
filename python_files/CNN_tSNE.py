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
from tqdm.auto import tqdm
import time


class tSNE(object):
    """
        source: "VisualizingDatausingt-SNE" by Laurens van der Maaten and Geoffrey Hinton in 2008, and it's also cited
        in the project paper.
        paper link: http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf.
    """
    def __init__(self, perplexity):
        self.__perplexity = perplexity
        self.__nominator = None

    def __neg_euc_dist_mat(self, mat):
        """
            calculate negative euclidean distance matrix for furture conditoinal prob computation

            Args:
                mat: input matrix or tensors
            Returns:
                negative distance matrix of input tensor
        """
        mat_norm = tf.reduce_sum(mat * mat, 1)
        mat_norm = tf.reshape(mat_norm, shape=[-1, 1])
        dist_mat = mat_norm - 2 * tf.matmul(mat, tf.transpose(mat)) + tf.transpose(mat_norm)
        return -dist_mat

    def __prob_j_i_mat(self, neg_dist_mat, sigma_vec=None):
        """
            calculate p(j|i) or q(j|i) matrix for SNE algorithm
            Args:
                neg_dist_mat: negative euclidean distance
                sigma_vec: sigma values in a vector
            Return:
                conditional prob matrix P(j|i)
        """
        if sigma_vec:
            sigmas = tf.reshape(sigma_vec, shape=[-1, 1])
            sigmas = 2 * tf.square(sigmas)
            neg_dist_mat /= sigmas
            return tf.nn.softmax(neg_dist_mat, axis=1)
        else:
            return tf.nn.softmax(neg_dist_mat, axis=1)

    def __info_entropy_vec(self, prob_ji_mat):
        """
            calculate information entroy vecotor

            Args:
                prob_ji_mat: conditional prob matrix P(j|i)
            Returns:
                information entropy vector based on the conditional prob matrix P(j|i)
        """
        return -tf.reduce_sum(prob_ji_mat * (tf.math.log(prob_ji_mat) / tf.math.log(2.0)), axis=1)

    def __perplexity_vec(self, entropy_vec):
        """
            calculate perplexity

            Args:
                entropy_vec: information entorpy vector
            Returns:
                perplexity
        """
        return 2 ** entropy_vec

    def __perplexity_wrapper(self, neg_dist_mat, sigma_vec=None):
        """
            perplexity wrapper to wrap up all perplexity function to binary search

            Args:
                neg_dist_mat: negative distance matrix of input tensors
                sigma_vec: all sigmas in the vector
            Return:
                a perplexity function wrapper
        """
        prob_ji_mat = self.__prob_j_i_mat(neg_dist_mat, sigma_vec)
        entropy_vec = self.__info_entropy_vec(prob_ji_mat)
        return 2 ** entropy_vec

    def __increasing_binary_search(self, func_wrapper, target, tolerence=1e-10, epochs=1000, lower_bound=1e-20,
                                   upper_bound=1000.0):
        """
            monotonically increasing function parameter binary search algorithm and it's a CPU computing;

            Args:
                func_wrapper: perplexity function wrapper
                target: user-defined perplexity value
            Returns:
                x_search: one sigma value
        """
        for i in range(epochs):
            x_search = (lower_bound + upper_bound) / 2
            res = func_wrapper(x_search)
            if res > target:
                upper_bound = x_search
            else:
                lower_bound = x_search
            if abs((res - target)) <= tolerence:
                break
        return x_search

    def __search_sigma(self, neg_dist_mat):
        """
            binary search sigmas for conditional probability matrix;

            Args:
                neg_dist_mat: negative euclidean distance matrix
            Returns:
                sigma_vec: a vector of sigmas
        """
        sigma_vec = []
        for row_index in tqdm(range(tf.shape(neg_dist_mat)[0])):
            neg_dist_row = neg_dist_mat[row_index:row_index + 1, :]
            perplexity_func_sigma = lambda sigma: self.__perplexity_wrapper(neg_dist_row, sigma_vec=sigma)
            sigma = self.__increasing_binary_search(func_wrapper=perplexity_func_sigma, target=self.__perplexity)
            sigma_vec.append(sigma)
        return sigma_vec

    def __p_ij_joint(self, neg_dist_mat, sigma_vec=None):
        """
            calculate joint p_ij for both SNE and tSNE algorithms

            Args:
                neg_dist_mat: negative euclidean distance matrix
                sigma_vec: a vector of sigmas
            Returns:
                p_ij: joint prob of the input tensor
        """
        p = self.__prob_j_i_mat(neg_dist_mat, sigma_vec=sigma_vec)
        p_t = tf.transpose(p)
        n = tf.cast(tf.shape(p)[0], dtype=tf.float32)
        p_ij = (p + p_t) / (2 * n)
        return p_ij

    def __q_ij_joint(self, neg_dist_mat):
        """
            calculate joint q_ij for SNE algorithm

            Args:
                neg_dist_mat: negative euclidean distance matrix
                sigma_vec: a vector of sigmas
            Returns:
                q_ij: joint prob of the target tensor
        """
        return tf.exp(neg_dist_mat) / tf.reduce_sum(tf.exp(neg_dist_mat), axis=None)

    def __q_ij_joint_tsne(self, neg_dist_mat):
        """
            calculate joint q_ij for tSNE algorithm

            Args:
                neg_dist_mat: negative euclidean distance matrix
                sigma_vec: a vector of sigmas
            Returns:
                q_ij: joint prob of the target tensor
        """
        nominator = tf.pow(tf.math.subtract(1, neg_dist_mat), -1.0)
        denominator = tf.reduce_sum(nominator, axis=None)
        self.__nominator = nominator
        return nominator / denominator

    def __calc_gradient_sne(self, q_ij, p_ji, Y):
        """
            calculate gradient of KL-divergence loss function for SNE algorithms

            Args:
                q_ij: joint prob of the target tensor
                p_ji: joint prob of the input tensor
                Y: target tensor with low dimension
            Returns:
                grad: a gradient matrix to update Y
        """
        p_q = p_ji - q_ij
        p_q_update = tf.expand_dims(p_q, 1)
        Y_j = tf.expand_dims(Y, 0)
        Y_i = tf.expand_dims(Y, 1)
        Y_update = Y_i - Y_j
        grad = tf.matmul(p_q_update, Y_update)
        grad = tf.reduce_sum(grad, axis=1)
        return grad

    def __calc_gradient_tsne(self, q_ij, p_ji, Y):
        """
            calculate gradient of KL-divergence loss function for tSNE algorithms

            Args:
                q_ij: joint prob of the target tensor
                p_ji: joint prob of the input tensor
                Y: target tensor with low dimension
            Returns:
                grad: a gradient matrix to update Y
        """
        p_q = p_ji - q_ij
        p_q_update = tf.expand_dims(p_q, 2)
        Y_i = tf.expand_dims(Y, 1)
        Y_j = tf.expand_dims(Y, 0)
        Y_update = Y_i - Y_j
        nominator = tf.expand_dims(self.__nominator, 2)
        Y_nominator = Y_update * nominator
        grad = tf.reduce_sum(4 * p_q_update * Y_nominator, axis=1)
        return grad

    def calc_p_ji(self, train_data):
        """calculate joint prob p(ji) for image tensor input"""
        shape = tf.shape(train_data)
        train_data = tf.reshape(train_data, shape=(shape[0], shape[1] * shape[2]))
        neg_dist_mat = self.__neg_euc_dist_mat(train_data)
        sigma_vec = self.__search_sigma(neg_dist_mat)
        p_ji = self.__p_ij_joint(neg_dist_mat, sigma_vec=sigma_vec)
        return p_ji

    def calc_q_ij(self, Y):
        """calculate joint prob q(ij)"""
        neg_dist_mat_y = self.__neg_euc_dist_mat(Y)
        q_ij = self.__q_ij_joint_tsne(neg_dist_mat_y)
        return q_ij

    def tsne_training(self, x_train, epochs, lr, momentum=0.9, ydim=2, y_initial='random_normal'):
        """
            fit tSNE model
            Args:
                x_train: the complete fit dataset
                epochs: fit epoch
                lr: learning rate
                momentum: uer-defined momentum value
                perplexity:  uer-defined perplexity value
                ydim: the target dimension size
            Return:
                N/A
        """
        if y_initial == 'uniform_normal':
            Y = tf.random.uniform(minval=0, maxval=1, shape=(tf.shape(x_train)[0], ydim))
        if y_initial == 'random_normal':
            Y = tf.random.normal(shape=(tf.shape(x_train)[0], ydim), mean=0, stddev=1)
        neg_dist_mat = self.__neg_euc_dist_mat(x_train)
        sigma_vec = self.__search_sigma(neg_dist_mat)
        p_ji = self.__p_ij_joint(neg_dist_mat, sigma_vec=sigma_vec)
        if momentum:
            Y_m2 = tf.identity(Y)
            Y_m1 = tf.identity(Y)
        for epoch in range(epochs):
            print(f'epoch: {epoch + 1}...')
            neg_dist_mat_y = self.__neg_euc_dist_mat(Y)
            q_ij = self.__q_ij_joint_tsne(neg_dist_mat_y)
            grad = self.__calc_gradient_tsne(q_ij, p_ji, Y)
            Y = Y - lr * grad
            if momentum:  # Add momentum
                Y += momentum * tf.subtract(Y_m1, Y_m2)
                Y_m2 = tf.identity(Y_m1)
                Y_m1 = tf.identity(Y)
        return Y

    def sne_training(self, x_train, epochs, lr, momentum=0.9, perplexity=2, ydim=2):
        """
            fit SNE model

            Args:
                x_train: the complete fit dataset
                epochs: fit epoch
                lr: learning rate
                momentum: uer-defined momentum value
                perplexity:  uer-defined perplexity value
                ydim: the target dimension size
            Return:
                N/A
        """
        #     Y = tf.random.uniform(minval=0, maxval=1, shape=(tf.shape(x_train)[0], ydim))
        Y = tf.random.normal(shape=(tf.shape(x_train)[0], ydim), mean=0, stddev=1)
        neg_dist_mat = self.__neg_euc_dist_mat(x_train)
        sigma_vec = self.__search_sigma(neg_dist_mat, perplexity)
        p_ji = self.__p_ij_joint(neg_dist_mat, sigma_vec=sigma_vec)
        if momentum:
            Y_m2 = tf.identity(Y)
            Y_m1 = tf.identity(Y)
        for epoch in range(epochs):
            print(f'epoch: {epoch + 1}...')
            neg_dist_mat_y = self.__neg_euc_dist_mat(Y)
            q_ij = self.__q_ij_joint(neg_dist_mat_y)
            grad = self.__calc_gradient_sne(q_ij, p_ji, Y)
            Y = Y - lr * grad
            if momentum:  # Add momentum
                Y += momentum * tf.subtract(Y_m1, Y_m2)
                Y_m2 = tf.identity(Y_m1)
                Y_m1 = tf.identity(Y)
        return Y


class CNN_tSNE(tSNE):
    """
        CNN_tSNE is a subclass of tSNE class and it connects a CNN model to tSNE model fit CNN model to generate
        the similar data embeddings based on tSNE model. CNN_tSNE is the major class used to reduce the dimension
         of MRI images in the proposed methodology

        Args:
            perplexity: user-defined perplexity value around 5-30
            cnn_model: user-defined CNN model
        Return:
            N/A
    """

    def __init__(self, perplexity, cnn_model):
        super(CNN_tSNE, self).__init__(perplexity=perplexity)
        self.perplexity = perplexity
        self.cnn_model = cnn_model

    def __train_tsne(self, p_ji, Y, epochs, lr, momentum=0.9, verbose=-1):
        """fit t_sne model in cnn_tsne_model_list"""
        if momentum:
            Y_m2 = tf.identity(Y)
            Y_m1 = tf.identity(Y)
        for epoch in range(epochs):
            if verbose != -1:
                print(f'epoch: {epoch + 1}...')
            q_ij = self.calc_q_ij(Y)
            grad = self._tSNE__calc_gradient_tsne(q_ij, p_ji, Y)
            Y = Y - lr * grad
            if momentum:  # Add momentum
                Y += momentum * tf.subtract(Y_m1, Y_m2)
                Y_m2 = tf.identity(Y_m1)
                Y_m1 = tf.identity(Y)
        return Y, q_ij

    def __train_sne(self, p_ji, Y, epochs, lr, momentum=0.9, verbose=-1):
        """fit sne model in cnn_tsne_model_list """
        if momentum:
            Y_m2 = tf.identity(Y)
            Y_m1 = tf.identity(Y)
        for epoch in range(epochs):
            if verbose != -1:
                print(f'epoch: {epoch + 1}...')
            neg_dist_mat_y = self.__neg_euc_dist_mat(Y)
            q_ij = self.__q_ij_joint(neg_dist_mat_y)
            grad = self._tSNE__calc_gradient_sne(q_ij, p_ji, Y)
            Y = Y - lr * grad
            if momentum:  # Add momentum
                Y += momentum * tf.subtract(Y_m1, Y_m2)
                Y_m2 = tf.identity(Y_m1)
                Y_m1 = tf.identity(Y)
        return Y

    def __grad(self, target, data):
        """calculate gradients to update cnn_tsne_model_list"""
        loss_func = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            loss = loss_func(target, self.cnn_model(data))
        return loss, tape.gradient(loss, self.cnn_model.trainable_variables)

    def train_model(self, x_train, epochs, tsne_lr):
        """
            train cnn_tsne_model_list with the complete dataset

            Args:
                x_train: the complete train dataset
                epochs: fit epochs
                tsne_lr: tSNE model learning rate
            Returns:
                cnn model fit loss history, tSNE model fit loss history, the train data embeddings, trained cnn_model
        """
        history_CNN_loss = []
        history_tSNE_loss = []
        '''define _optimizer'''
        Adam = tf.keras.optimizers.Adam()
        ''' sigmas searching'''
        print('binary searching sigma...')
        p_ji = self.calc_p_ji(x_train)
        Y_pred = self.cnn_model(x_train)
        '''model fit'''
        for epoch in range(epochs):
            '''time starts'''
            time_start = time.time()
            '''t-SNE predict and update tSNE model '''
            Y_pred, q_ij = self.__train_tsne(p_ji=p_ji, Y=Y_pred, epochs=1, lr=tsne_lr)
            '''calc CNN loss & tSNE loss'''
            CNN_loss, gradient = self.__grad(Y_pred, x_train)
            tSNE_loss = tf.reduce_mean(tf.keras.losses.kullback_leibler_divergence(q_ij, p_ji), axis=-1)
            '''update weights CNN weights'''
            Adam.apply_gradients(zip(gradient, self.cnn_model.trainable_variables))
            ''' record fit loss and acc per epochs '''
            history_CNN_loss.append(CNN_loss)
            history_tSNE_loss.append(tSNE_loss)
            '''time ends'''
            time_end = time.time()
            '''print metrics per epoch'''
            epoch_time = time_end - time_start
            epoch = epoch + 1
            CNN_acc = (1 - CNN_loss) ** 2
            tSNE_acc = (1 - tSNE_loss) ** 2
            if epoch % 10 == 0:
                print('epoch:%5s, epoch_time: %2.5f, CNN_loss: %2.5f,  tSNE_loss: %2.5f' %
                      (epoch, epoch_time, CNN_loss, tSNE_loss))
        return history_CNN_loss, history_tSNE_loss, Y_pred, self.cnn_model

    def train_model_batch(self, x_train, epochs, tsne_lr):
        """
            train cnn_tsne_model_list with the batch dataset

            Args:
                x_train: train dataset in batches
                epochs: fit epochs
                tsne_lr: tSNE model learning rate
            Returns:
                cnn model fit loss history, tSNE model fit loss history, trained cnn_model
        """
        '''history matrix'''
        history_CNN_loss = []
        history_tSNE_loss = []
        '''define _optimizer'''
        Adam = tf.keras.optimizers.Adam()
        '''searching sigmags'''
        print('Note: sigmas binary searching starts in CPU computing: ')
        p_ji_list = []
        for index, batch in enumerate(x_train):
            print(f'batch: {index + 1}')
            p_ji = self.calc_p_ji(batch)
            p_ji_list.append(p_ji)
        '''model fit'''
        print('Note: cnn_tsne model fit starts in GPU computing: ')
        for epoch in range(epochs):
            epoch += 1
            step_CNN_loss = tf.keras.metrics.Mean()
            step_tSNE_loss = tf.keras.metrics.Mean()
            '''time starts'''
            time_start = time.time()
            for index, batch in enumerate(x_train):
                p_ji = p_ji_list[index]
                Y_pred = self.cnn_model(batch)
                '''t-SNE predict and update tSNE model '''
                Y_pred, q_ij = self.__train_tsne(p_ji=p_ji, Y=Y_pred, epochs=1, lr=tsne_lr)
                '''calc CNN loss & tSNE loss'''
                CNN_loss, gradient = self.__grad(Y_pred, batch)
                tSNE_loss = tf.reduce_mean(tf.keras.losses.kullback_leibler_divergence(q_ij, p_ji), axis=-1)
                '''update weights CNN weights'''
                Adam.apply_gradients(zip(gradient, self.cnn_model.trainable_variables))
                '''record each step loss and acc '''
                step_CNN_loss.update_state(CNN_loss)
                step_tSNE_loss.update_state(tSNE_loss)
            '''time ends'''
            time_end = time.time()
            '''' record fit loss and acc per epochs '''
            history_CNN_loss.append(step_CNN_loss.result())
            history_tSNE_loss.append(step_tSNE_loss.result())
            if epoch % 10 == 0:
                print('epoch:%5s, epoch_time: %2.5f, CNN_loss: %2.5f,  tSNE_loss: %2.5f' % (
                    epoch, (time_end - time_start), step_CNN_loss.result(), step_tSNE_loss.result()))
        return history_CNN_loss, history_tSNE_loss, self.cnn_model

    def train_model_batch_tfr(self, x_train, epochs, tsne_lr):
        """history matrix"""
        history_CNN_loss = []
        history_tSNE_loss = []
        '''define _optimizer'''
        Adam = tf.keras.optimizers.Adam()
        '''searching sigmags'''
        print('Note: sigmas binary searching starts in CPU computing: ')
        p_ji_list = []
        for index, batch in enumerate(x_train):
            print(f'batch: {index + 1}')
            p_ji = self.calc_p_ji(batch['image_tensor'])
            p_ji_list.append(p_ji)
        '''model training'''
        print('Note: cnn_tsne model training starts in GPU computing: ')
        for epoch in range(epochs):
            epoch += 1
            step_CNN_loss = tf.keras.metrics.Mean()
            step_tSNE_loss = tf.keras.metrics.Mean()
            '''time starts'''
            time_start = time.time()
            for index, batch in enumerate(x_train):
                p_ji = p_ji_list[index]
                Y_pred = self.cnn_model(batch['image_tensor'])
                '''t-SNE predict and update tSNE model '''
                Y_pred, q_ij = self.__train_tsne(p_ji=p_ji, Y=Y_pred, epochs=1, lr=tsne_lr)
                '''calc CNN loss & tSNE loss'''
                CNN_loss, gradient = self.__grad(Y_pred, batch)
                tSNE_loss = tf.reduce_mean(tf.keras.losses.kullback_leibler_divergence(q_ij, p_ji), axis=-1)
                '''update weights CNN weights'''
                Adam.apply_gradients(zip(gradient, self.cnn_model.trainable_variables))
                '''record each step loss and acc '''
                step_CNN_loss.update_state(CNN_loss)
                step_tSNE_loss.update_state(tSNE_loss)
            '''time ends'''
            time_end = time.time()
            '''' record training loss and acc per epochs '''
            history_CNN_loss.append(step_CNN_loss.result())
            history_tSNE_loss.append(step_tSNE_loss.result())
            if epoch % 10 == 0:
                print('epoch:%5s, epoch_time: %2.5f, CNN_loss: %2.5f,  tSNE_loss: %2.5f' % (
                    epoch, (time_end - time_start), step_CNN_loss.result(), step_tSNE_loss.result()))
        return history_CNN_loss, history_tSNE_loss, self.cnn_model

    def __grad_multiple_loss(self, target, data):
        """calculate gradients to update cnn_tsne_model_list with classification loss"""
        loss_func1 = tf.keras.losses.MeanSquaredError()
        loss_func2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        with tf.GradientTape() as tape:
            loss1 = loss_func1(target[0], self.cnn_model(data)[0])
            loss2 = loss_func2(target[1], self.cnn_model(data)[1])
            loss = loss1 + loss2
        return loss1, loss2, tape.gradient(loss, self.cnn_model.trainable_variables)

    def train_model_multiple_loss(self, x_train, x_label, epochs, tsne_lr):
        """
             train cnn_tsne_model_list with the complete dataset & classification loss

             Args:
                x_train: the train dataset
                x_label: the train label
                epochs: fit epochs
                tsne_lr: tSNE model learning rate
             Returns:
                history_CNN_reg_loss: cnn model regression loss history
                history_tSNE_loss: tsne model fit loss history
                history_CNN_cls_loss: cnn model classification loss history
                Y_pred: data embeddings
                cnn_model: trained cnn model
         """
        '''history matrix'''
        history_CNN_reg_loss = []
        history_tSNE_loss = []
        history_CNN_cls_loss = []
        '''define _optimizer'''
        Adam = tf.keras.optimizers.Adam()
        ''' sigmas searching'''
        print('sigmas searching...')
        p_ji = self.calc_p_ji(x_train)
        Y_pred = self.cnn_model(x_train)[0]
        '''model fit'''
        print('CNN, tSNE fit...')
        for epoch in range(epochs):
            '''time starts'''
            time_start = time.time()
            '''t-SNE predict and update tSNE model '''
            Y_pred, q_ij = self.__train_tsne(p_ji, Y=Y_pred, epochs=1, lr=tsne_lr)
            '''calc CNN reg loss & CNN cls loss & tSNE loss'''
            CNN_reg_loss, CNN_cls_loss, gradient = self.__grad_multiple_loss([Y_pred, x_label], x_train)
            tSNE_loss = tf.reduce_mean(tf.keras.losses.kullback_leibler_divergence(q_ij, p_ji), axis=-1)
            '''update weights CNN weights'''
            Adam.apply_gradients(zip(gradient, self.cnn_model.trainable_variables))
            ''' record fit loss and acc per epochs '''
            history_CNN_reg_loss.append(CNN_reg_loss)
            history_tSNE_loss.append(tSNE_loss)
            history_CNN_cls_loss.append(CNN_cls_loss)
            '''time ends'''
            time_end = time.time()
            '''print metrics per epoch'''
            epoch_time = time_end - time_start
            epoch = epoch + 1
            if epoch % 10 == 0:
                print('epoch:%5s, epoch_time: %2.5f, CNN_reg_loss: %2.5f,  CNN_cls_loss: %2.5f, tSNE_loss: %2.5f,' %
                      (epoch, epoch_time, CNN_reg_loss, CNN_cls_loss, tSNE_loss,))
        return history_CNN_reg_loss, history_tSNE_loss, history_CNN_cls_loss, Y_pred, self.cnn_model
