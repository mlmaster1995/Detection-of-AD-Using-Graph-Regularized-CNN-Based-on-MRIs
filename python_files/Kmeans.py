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
import numpy as np
from python_files.graph_data_processing import GraphDataProcess


class KMeans(object):
    def __init__(self, data, K):
        self.__data = data
        self.__K = K
        self.__k_list = tf.range(0, K, 1)
        self.__k_centroids = self.__initialize_k_centorids()
        self.new_centroids = tf.convert_to_tensor([])
        self.centroids_mean = None

    def get_K(self):
        return self.__K

    def searching_K(self, K_limit, epoch_per_K, verbose=1):
        c_means_clc = []
        for k in range(K_limit):
            tf.keras.backend.clear_session()
            kmeans = KMeans(data=self.__data, K=k + 1)
            kmeans.set_centroids(optimize=True)
            kmeans.fit(epoch=epoch_per_K, verbose=verbose)
            cluster_data = kmeans.get_clusters()
            c_means_clc.append(kmeans.get_centroids_mean(kmeans.new_centroids, cluster_data))
        return c_means_clc, K_limit

    def get_centroids_mean(self, centroids, cluster_data):
        def calc_mean_dist_center_data(center, data):
            if np.array(data).size != 0:
                return np.mean(np.sqrt(np.sum(np.power((center - data), 2), axis=1)))
            else:
                return 0.0

        return np.mean([calc_mean_dist_center_data(center=centroids[data_pair[1]],
                                                   data=data_pair[0]) for data_pair in cluster_data])

    def set_centroids(self, centroids=None, optimize=False):
        """
            set up centroids with custom center values or optimized center values

            args:
                centroids: the custom centroid value, if optimize is true, the value will be overrided
                optimize: pick the best initial centroid from the dataset
            return:
                N/A, but initialize k center values internally
        """
        self.__k_centroids = centroids
        if optimize:
            self.__k_centroids = self.__initialize_k_centorids_optimize()

    def get_clusters(self):
        min_distance_cluster_index = self.__get_min_distance_cluster_index()
        cluster_data = []
        for k in range(self.__K):
            k_index = tf.where(min_distance_cluster_index == k)
            if tf.size(k_index).numpy():
                data_cluster = tf.gather_nd(self.__data, k_index)
                cluster_data.append((data_cluster.numpy(), k, tf.shape(data_cluster).numpy()))
            else:
                cluster_data.append(([], k))
        return cluster_data

    def __neg_euc_dist_mat(self, mat):
        """
            Calculate euclidean distance matrix for furture conditoinal prob computation

            Args:
                mat: input matrix or tensors
            Returns:
                negative distance matrix of input tensor
        """
        mat_norm = tf.reduce_sum(mat * mat, 1)
        mat_norm = tf.reshape(mat_norm, shape=[-1, 1])
        dist_mat = mat_norm - 2 * tf.matmul(mat, tf.transpose(mat)) + tf.transpose(mat_norm)
        return dist_mat

    def __initialize_k_centorids_optimize(self):
        rows, cols = tf.shape(self.__data)
        if self.__K <= rows:
            distance_matrix = self.__neg_euc_dist_mat(self.__data)
            distance_mean = tf.reduce_mean(distance_matrix, axis=1)
            value, indices = tf.math.top_k(input=distance_mean, k=self.__K, sorted=True)
            indices = tf.expand_dims(indices, axis=1)
            k_centroids = tf.gather_nd(self.__data, indices)
            return k_centroids
        else:
            raise Exception('cluster number is over the data size!')

    def __initialize_k_centorids(self):
        rows, cols = tf.shape(self.__data)
        if self.__K <= rows:
            k_index = tf.random.uniform(shape=(self.__K,), minval=0, maxval=rows - 1, dtype=tf.int32)
            k_index = tf.expand_dims(k_index, axis=1)
            return tf.gather_nd(params=self.__data, indices=k_index)
        else:
            raise Exception('cluster number is over the data size!')

    def __get_min_distance_cluster_index(self, ):
        distance_matrix = self.__calc_distance()
        min_distance_cluster_index = tf.math.argmin(input=distance_matrix, axis=0)
        min_distance_cluster_index = tf.cast(min_distance_cluster_index, dtype=tf.int32)
        return min_distance_cluster_index

    def __calc_distance(self):
        square_diff = tf.math.squared_difference(x=tf.expand_dims(self.__k_centroids, axis=1),
                                                 y=tf.expand_dims(self.__data, axis=0))
        square_diff_sum = tf.reduce_sum(square_diff, axis=2)
        distance_matrix = tf.sqrt(square_diff_sum)
        return distance_matrix

    def __update_centroids(self, min_distance_cluster_index, epoch):
        miss_k = tf.where(tf.convert_to_tensor([k in min_distance_cluster_index for k in self.__k_list]) == False)
        miss_k = tf.cast(miss_k, dtype=tf.int32)
        self.new_centroids = tf.convert_to_tensor([])

        def update_centroid(i):
            if i in miss_k:
                miss_centroid_index = tf.gather_nd(miss_k, tf.where(i == miss_k))
                miss_centroid = tf.gather_nd(self.__k_centroids, miss_centroid_index)
                miss_centroid = tf.squeeze(miss_centroid)
                self.new_centroids = tf.concat([self.new_centroids, miss_centroid], axis=0)
                print('epoch: ', epoch, ',miss_centroid: ', miss_centroid)
            else:
                data_index = tf.where(min_distance_cluster_index == i)
                data = tf.gather_nd(self.__data, data_index)
                new_center = tf.reduce_mean(data, axis=0)
                self.new_centroids = tf.concat([self.new_centroids, new_center], axis=0)
            return (tf.add(i, 1),)

        condition = lambda i: tf.less(i, self.__K)
        index = tf.constant(0)
        _ = tf.while_loop(condition, update_centroid, [index])
        cols = tf.size(self.new_centroids) / self.__K
        self.new_centroids = tf.reshape(self.new_centroids,
                                        shape=(self.__K, cols))
        self.__k_centroids = self.new_centroids

    def fit(self, epoch, verbose=1):
        """fit k-means model"""

        def training_step(i):
            min_distance_cluster_index = self.__get_min_distance_cluster_index()
            self.__update_centroids(min_distance_cluster_index, epoch=i.numpy() + 1)
            # print('epoch: ', i.numpy() + 1,
            #       ',centroid mean: ', tf.reduce_mean(self.new_centroids).numpy(),
            #       '\ncentroids:\n', self.new_centroids.numpy())
            if verbose == 1:
                print('epoch: ', i.numpy() + 1,
                      ',centroid mean: ', tf.math.abs(tf.reduce_mean(self.new_centroids)).numpy())
            return (tf.add(i, 1),)

        index = tf.constant(0)
        condition = lambda i: tf.less(i, epoch)
        _ = tf.while_loop(condition, training_step, [index])
        self.centroids_mean = tf.reduce_mean(self.new_centroids)


class KMeansModels(object):
    @staticmethod
    def generate_model_list(path_list, rep_dim, K_list, epoch, batch=1000):
        model_list = []
        for path_index, path in enumerate(path_list):
            tf.keras.backend.clear_session()
            print(f'\n====================== Model {path_index} training ======================\n')
            image_rep_data = GraphDataProcess.parse_tfr_to_image_rep(path_list=[path],
                                                                     rep_dim=rep_dim,
                                                                     batch_size=batch)
            iteration = iter(image_rep_data)
            data = iteration.get_next()
            data_rep = data['representation']
            kmeans = KMeans(data=data_rep, K=K_list[path_index])
            kmeans.set_centroids(optimize=True)
            kmeans.fit(epoch=epoch)
            model_list.append(kmeans)
        return model_list
