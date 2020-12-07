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
from sklearn.cluster import AgglomerativeClustering
from copy import deepcopy
from graph_data_processing import GraphDataProcess


class Hierchaical(object):
    def __init__(self,
                 data=None,
                 n_clusters=None,
                 affinity='euclidean',
                 memory=None,
                 connectivity=None,
                 compute_full_tree='auto',
                 linkage='ward',
                 distance_threshold=None):
        self.n_cluster = n_clusters
        self.affinity = affinity
        self.connectivity = connectivity
        self.memory = memory
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.data = data

    def _call_model(self):
        return AgglomerativeClustering(n_clusters=self.n_cluster,
                                       affinity=self.affinity,
                                       memory=self.memory,
                                       connectivity=self.connectivity,
                                       compute_full_tree=self.compute_full_tree,
                                       linkage=self.linkage,
                                       distance_threshold=self.distance_threshold)

    def fit(self):
        return self._call_model().fit(self.data)

    def search_clusters(self):
        """return the number of clusters and labels of each instances"""
        model = self._call_model()
        model = model.fit(self.data)
        if self.n_cluster is None:
            return model.n_clusters_, model.labels_
        else:
            return self.n_cluster, model.labels_

    def get_clusters(self):
        n_cluster, labels = self.search_clusters()
        dict_cluster = {}
        for data_index, data in enumerate(self.data):
            label = labels[data_index]
            if label not in dict_cluster:
                dict_cluster[label] = []
                dict_cluster[label].append(data)
            else:
                dict_cluster[label].append(data)
        cluster_list = [dict_cluster[key] for key in dict_cluster.keys()]
        cluster_length = [len(lst) for lst in cluster_list]
        return cluster_list, cluster_length

    def get_centroids(self, cluster_list):
        for cluster_index, cluster in enumerate(cluster_list):
            centroid_mean = tf.reduce_mean(cluster, axis=0)
            if cluster_index == 0:
                centroid_mean = tf.expand_dims(centroid_mean, axis=0)
                centroid_list = deepcopy(centroid_mean)
            else:
                centroid_mean = tf.expand_dims(centroid_mean, axis=0)
                centroid_list = tf.concat([centroid_list, centroid_mean], axis=0)
        centroid_root = tf.expand_dims(tf.reduce_mean(centroid_list, axis=0), axis=0)
        centroid_complete = tf.concat([centroid_root, centroid_list], axis=0)
        return centroid_complete


class HierchaicalModels(object):
    @staticmethod
    def generate_model_list(path_list, rep_dim, cluster_num_list, batch=5000):
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
            tree_model = Hierchaical(data=data_rep, n_clusters=cluster_num_list[path_index], affinity='euclidean',
                                     compute_full_tree='auto', linkage='ward', distance_threshold=None)
            tree_model.fit()
            model_list.append(tree_model)
            print("Done...")
        return model_list
