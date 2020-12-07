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
import os
import tensorflow as tf
import neural_structured_learning as nsl
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import shutil


class GraphDataProcess(object):
    """
        GraphDataProcess class contains all methods for data balancing, generate & parse MRI image dataset, generate &
        parse MRI representation dataset, MRI presentation graph generation. It's heavily used in both Experiment I and
        II for building the data pipline.
    """

    @staticmethod
    def setup_train_test_folders(root_path, label_list, ratio):
        """
            This function is to create both train and test folders with all label subfolders under root_path. The MRI
            images are divided and copied to train and test based on the ratio under the root_path.

            arg:
                root_path: path to containing all the dataset
                label_list: a list of labels
                ratio: train, test ratio between 0 and 1
            return:
                N/A
        """

        # copy mir images
        def copy_mri(src_path, target_path, label_index):
            name_is_new = True
            target_name_list = os.listdir(target_path)
            while name_is_new:
                sample_name = f'{label_index}_{int(np.random.uniform(0, max_size - 1))}.jpg'
                if not (sample_name in target_name_list):
                    sample_path = f'{src_path}{sample_name}'
                    copy_path = f'{target_path}{sample_name}'
                    shutil.copy(sample_path, copy_path)
                    name_is_new = False

        # generate train and test data size for each label
        max_size = max(list(map(lambda x: len(os.listdir(f'{root_path}{x}/')), label_list)))
        print("train_data_size/label(rougly): ", int(max_size * ratio))
        print("test_data_size/label(rougly): ", max_size - int(max_size * ratio))

        # create folders for both train and test dataset
        train_path = f'{root_path}train/'
        test_path = f'{root_path}test/'
        label_path_train = [f'{train_path}{label}/' for label in label_list]
        label_path_test = [f'{test_path}{label}/' for label in label_list]
        if not os.path.exists(train_path):
            os.mkdir(train_path)
            for single_label_path in label_path_train: os.mkdir(single_label_path)
        if not os.path.exists(test_path):
            os.mkdir(test_path)
            for single_label_path in label_path_test: os.mkdir(single_label_path)

        # seperate data
        for label_index, label in enumerate(label_list):
            src_path = f'{root_path}{label}/'
            train_path = label_path_train[label_index]
            test_path = label_path_test[label_index]
            for _ in range(max_size):
                if np.random.uniform(0, 1) <= ratio:
                    copy_mri(src_path, train_path, label_index)
                else:
                    copy_mri(src_path, test_path, label_index)
            print(f'{label} folder is split...')

    @staticmethod
    def rename_images(label_list, root_path):
        """
            Rename all the image names with format "label_number+image_index" such as "011", 0->label 0, 11->11th image
            under this label;

            Args:
                label_list: a list of label names ['NonDemented', 'VeryMildDemented','MildDemented', 'ModerateDemented']
                root_path: a root path of all images folders

            Return:
                N/A
        """
        for label_index, label in enumerate(label_list):
            name_list = os.listdir(f'{root_path}{label}/')
            for name_index, name in enumerate(name_list):
                original_name = f'{root_path}{label}/{name}'
                new_name = f'{root_path}{label}/{label_index}_{name_index}.jpg'
                os.rename(original_name, new_name)

    @staticmethod
    def _augmentation(image_tensor, path, add_noise, image_size=(208, 176), visual=False):
        def visualize(original, augmented, similarity):
            plt.figure(figsize=(8, 10))
            plt.subplot(1, 2, 1)
            plt.title('Original Image')
            plt.imshow(original / 255)

            plt.subplot(1, 2, 2)
            plt.title(f'Augmented Image({similarity})')
            plt.imshow(augmented / 255)

        aug_dict = {'sat': np.random.uniform(high=5, low=0),
                    'crop': np.random.uniform(low=0.955, high=1),
                    'rotate': np.random.uniform(low=-tf.constant(np.pi / 40), high=tf.constant(np.pi / 40))}
        noise = tf.random.normal(shape=tf.shape(image_tensor), mean=0, stddev=1)
        opt = image_tensor + noise if add_noise else image_tensor
        opt = tf.image.flip_left_right(opt)
        opt = tf.image.adjust_saturation(opt, aug_dict['sat'])
        opt = tf.image.central_crop(opt, aug_dict['crop'])
        opt = tfa.image.rotate(opt, aug_dict['rotate'])
        final = tf.image.resize(opt, size=image_size)
        cosine_sim = tf.keras.metrics.CosineSimilarity(axis=1)
        simi = np.round(cosine_sim(image_tensor, final).numpy(), 3)
        if visual:
            visualize(image_tensor, final, simi)
        tf.keras.preprocessing.image.save_img(path=path, x=final)

    @classmethod
    def balance_dataset(cls, root_path, label_list, image_size=(208, 176), balance_vision=False):
        """
              Balance the whole dataset with Bootstrapping and Augmentation

              Args:
                  root_path: the root path for the dataset and it's set up at the beginning of the notebook
                  label_list: a list for all labels, and it must match the name of the data folder under the root path
                  image_size: a tuple (Height, Weight) for image size
                  balance_vision: display the balancing process if it's true, but this process could causes huge ram and
                  delay the processing

              Return:
                  N/A;
        """
        path_list = [f'{root_path}{label}/' for label in label_list]
        # count images for each label
        image_counts = {}
        for label in label_list:
            image_counts[f'{label}_image_counts'] = len(os.listdir(f'{root_path}{label}/'))
        print(f"Note: image dataset counts is {image_counts}")
        original_count_list = list(image_counts.values())
        # optimize balance_target
        balance_target = max(image_counts.values())
        print(f"Note: dataset balance target is {balance_target} for each dataset")
        # calc count difference
        diff_count_list = [(balance_target - count) for count in image_counts.values()]
        print(f"Note: balance difference with regards to the target is {diff_count_list}")
        # balance dataset
        for count_index, diff_count in enumerate(diff_count_list):
            # get label & path for one dataset
            label = label_list[count_index]
            path = path_list[count_index]
            # balance dataset
            if diff_count > 0:
                if balance_vision:
                    print(f'*****************{label}_dataset balancing *****************')
                for copy_count in range(diff_count):
                    # sample an image and generate a new image path
                    name_list = os.listdir(path_list[count_index])
                    sample_name = name_list[int(np.random.uniform(low=0, high=len(name_list)))]
                    src = f'{path}{sample_name}'
                    dst = f'{path}{count_index}_{original_count_list[count_index] + copy_count}.jpg'
                    src_index = int(sample_name.split('_')[1].split('.')[0])
                    noise_flag = True if src_index <= original_count_list[count_index] else False
                    # load and augment image
                    image = tf.keras.preprocessing.image.load_img(src)
                    image_tensor = tf.keras.preprocessing.image.img_to_array(image)
                    cls._augmentation(image_tensor=image_tensor, path=dst, image_size=image_size,
                                      visual=balance_vision, add_noise=noise_flag)
        print(f'*****************dataset balancing done*****************')

    @staticmethod
    def _int64_feature(label):
        """
            args:
               label: image labels 0,1,2,3
            return:
                tf.train.Feature with the Int64 type
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

    @staticmethod
    def _bytes_feature(image_name):
        """
            args:
                image_name: image saving path as names
            return:
                tf.train.Feature with the Byte type
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_name]))

    @staticmethod
    def _float_feature(image_represent):
        """
            args:
                image_represent: low-dim image representation
            return:
                tf.train.Feature with the float type
        """
        return tf.train.Feature(float_list=tf.train.FloatList(value=image_represent.numpy()[0].tolist()))

    @classmethod
    def _image_example_raw(cls, image_name, image_label, image_string):
        """
            args:
                image_name: image saving path;
                image_label: 0,1,2,3;
                image_string: image in binary format;
            return:
                tf.train.Example message with features as {id: **, image_raw: **, label: **}
        """
        feature = {
            'id': cls._bytes_feature(image_name),
            'image_raw': cls._bytes_feature(image_string),
            'label': cls._int64_feature(image_label),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @classmethod
    def generate_tfr_raw(cls, path_list, tfr_list):
        """
            write all image dataset into binary strings in .tfr files

            args:
                path_list: paths of all images folders in a list
                tfr_list: tfr file names in a list
            return:
                N/A
        """
        for path_index, path in enumerate(path_list):
            name_list = os.listdir(path)
            tfr_file = tfr_list[path_index]

            with tf.io.TFRecordWriter(tfr_file) as writer:
                for name_index, name in enumerate(name_list):
                    image_name = name
                    image_label = path_index
                    image_string = open(f'{path}{name}', 'rb').read()
                    tf_example = cls._image_example_raw(image_name.encode('utf-8'), image_label, image_string)
                    writer.write(tf_example.SerializeToString())

    @classmethod
    def _image_example_represent(cls, image_name, image_label, image_represent):
        """
            args:
                image_name: the image file name;
                image_label: the image label;
                image_represent: the low-dim image representation;
            return:
                tf.train.Example message with features as {id: 25, label: 0, representation: [0.12, 0.5] }
        """
        feature = {
            'id': cls._bytes_feature(image_name),
            'label': cls._int64_feature(image_label),
            'representation': cls._float_feature(image_represent),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @classmethod
    def generate_tfr_vae_represent(cls, image_dataset_list, vae_model, tfr_rep_list):
        """
            write VAE represents into binary strings in the .tfr file

            args:
                image_dataset_list: tf.data.TFRecordDataset with batch size at "1";
                vae_model: trained Variational Autoencoder model;
                tfr_rep_file: a list of tfr file paths to save represents of images;
            return:
                N/A
        """
        for image_dataset_index, image_dataset in enumerate(image_dataset_list):
            tfr_file = tfr_rep_list[image_dataset_index]
            with tf.io.TFRecordWriter(tfr_file) as writer:
                for index, batch_image in enumerate(image_dataset):
                    _m, _l, rep = vae_model.encoder(batch_image['image_tensor'])
                    image_name = tf.compat.as_str_any(batch_image['id'].numpy()[0]).split('.')[0].encode('utf-8')
                    image_label = batch_image['label'].numpy()[0]
                    image_represent = rep
                    tf_example = cls._image_example_represent(image_name, image_label, image_represent)
                    writer.write(tf_example.SerializeToString())

    @classmethod
    def generate_tfr_aae_represent(cls, image_dataset_list, aae_model, tfr_rep_path):
        """
            write AAE represents into binary strings in the .tfr file

            args:
                image_dataset_list: tf.data.TFRecordDataset with batch size at "1";
                aae_model: trained Adversary Autoencoder model;
                tfr_rep_path: a list of tfr file paths to save represents of images;
            return:
                N/A
        """
        for image_dataset_index, image_dataset in enumerate(image_dataset_list):
            tfr_file = tfr_rep_path[image_dataset_index]
            with tf.io.TFRecordWriter(tfr_file) as writer:
                for index, batch_image in enumerate(image_dataset):
                    _m, _l, rep = aae_model.encoder(batch_image['image_tensor'])
                    image_name = tf.compat.as_str_any(batch_image['id'].numpy()[0]).split('.')[0].encode('utf-8')
                    image_label = batch_image['label'].numpy()[0]
                    image_represent = rep
                    tf_example = cls._image_example_represent(image_name, image_label, image_represent)
                    writer.write(tf_example.SerializeToString())

    @classmethod
    def generate_tfr_tsne_represent(cls, image_dataset_list, cnn_tsne_model_list, tfr_list):
        """
            write t-SNE represents into binary strings in the .tfr file

            args:
                image_dataset_list: tf.data.TFRecordDataset with batch size at "1";
                cnn_tsne_model_list: trained Variational Autoencoder model;
                tfr_list: a list of tfr file paths;
            return:
                N/A
        """
        for image_dataset_index, image_dataset in enumerate(image_dataset_list):
            tfr_file = tfr_list[image_dataset_index]
            with tf.io.TFRecordWriter(tfr_file) as writer:
                for index, batch_image in enumerate(image_dataset):
                    rep = cnn_tsne_model_list[image_dataset_index](batch_image['image_tensor'])
                    image_name = tf.compat.as_str_any(batch_image['id'].numpy()[0]).split('.')[0]
                    image_name = image_name.encode('utf-8')
                    image_label = batch_image['label'].numpy()[0]
                    image_represent = rep
                    tf_example = cls._image_example_represent(image_name, image_label, image_represent)
                    writer.write(tf_example.SerializeToString())

    @staticmethod
    def parse_tfr_to_image_rep(path_list, rep_dim, batch_size=1, shuffle=False):
        """
            Generate tf.data.Dataset with format such as {id: 256 , label: 1, representation: [0.2,0.5]}

            args:
                path_list: image representations tfr files in a list;
                batch_size: batch size on the data pipeline;
                rep_dim: dimensions of the image representation;
                shuffle: boolean value to decide shuffling the dataset;
            return:
                image_batch_dataset: the image represent dataset in batches
        """
        rep_feature_description = {
            'id': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'representation': tf.io.FixedLenFeature([rep_dim], tf.float32)
        }

        def parse_rep_function(example_proto):
            """Parse the input tf.Example proto using the dictionary above"""
            return tf.io.parse_single_example(example_proto, rep_feature_description)

        raw_rep_dataset = tf.data.TFRecordDataset(path_list)
        rep_dataset = raw_rep_dataset.map(parse_rep_function)
        if shuffle:
            rep_dataset = rep_dataset.shuffle(buffer_size=10000, seed=None, reshuffle_each_iteration=shuffle)
        rep_batch_dataset = rep_dataset.batch(batch_size)
        return rep_batch_dataset

    @staticmethod
    def parse_tfr_to_image_tensor(path_list, batch_size, channels=3, size=(208, 176), shuffle=False):
        """
            generate tf.data.Dataset with format {id: **, label: **, image_tensor: **}

            args:
                path_list: tfr files in a list;
                batch_size: data pipline batch size;
                channels: the Int values for image channels at 1 or 3;
                size: a tuple for image size (width, height);
                shuffle: boolean value to decide shuffling the dataset;
            return:
                image_batch_dataset: the image dataset in batches
        """
        image_feature_description = {
            'id': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

        def parse_image_function(example_proto):
            """Parse the input tf.Example proto using the dictionary above"""
            return tf.io.parse_single_example(example_proto, image_feature_description)

        def parse_image_raw(x):
            """Parse each image_raw to image_tensor"""
            # image_tensor = tf.io.decode_image(contents=x['image_raw'], channels=3, dtype=tf.float32)
            image_tensor = tf.image.decode_jpeg(x['image_raw'], channels=channels)
            # image_tensor = tf.cast(image_tensor, dtype=tf.float32)
            image_tensor = tf.image.resize(image_tensor, size=size)
            image_tensor = image_tensor / 255
            return {'id': x['id'],
                    'label': x['label'],
                    'image_tensor': image_tensor, }

        raw_image_dataset = tf.data.TFRecordDataset(path_list)
        parsed_image_dataset = raw_image_dataset.map(parse_image_function)
        image_dataset = parsed_image_dataset.map(parse_image_raw)

        if shuffle:
            image_dataset = image_dataset.shuffle(buffer_size=20000, seed=None, reshuffle_each_iteration=shuffle)
        image_batch_dataset = image_dataset.batch(batch_size)
        return image_batch_dataset

    @classmethod
    def generate_single_graph_with_cluster_kmeans(cls, data_tfr_rep_path, represent_dim, id_prefix, KMeans_model=None,
                                                  file_output_path=None, similarity_threshold=0.95):
        """
            The method to generate a graph data for one label with KMeans clusters

            Args:
                data_tfr_rep_path: the tfr file path of data representation
                file_output_path: the .tsv file path to save graph data
                represent_dim: the dimension of data representation
                id_prefix: prefix of id to differentiate labels and normally set up as the label value
                KMeans_model: the trained KMneas cluster model
                similarity_threshold: cosine similarity threshold to decide if two nodes are connected

            Return:
                The single graph data
        """
        all_cluster_edges_list = []
        # generate a temp folder to hold all intermediate files
        path = './temp_graph_processing/'
        if not os.path.isdir(path):
            os.mkdir(path)
        temp_tfr_file_path = f'{path}temp.tfr'
        temp_tsv_graph_path = f'{path}graph.tsv'
        # add centroid_mean into the cluster centroids
        centroids = KMeans_model.new_centroids
        centroids_mean = tf.reduce_mean(centroids, axis=0)
        centroids_mean = tf.expand_dims(centroids_mean, axis=0)
        centroids_complete = tf.concat([centroids_mean, centroids], axis=0)

        # generate cluster centroids tfr file
        with tf.io.TFRecordWriter(temp_tfr_file_path) as writer:
            for index, center in enumerate(centroids_complete):
                represent_id = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[f'{id_prefix}c{index}'.encode('utf-8')]))
                represent = tf.train.Feature(float_list=tf.train.FloatList(value=center.numpy().tolist()))
                feature = {'id': represent_id,
                           'representation': represent, }
                tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(tf_example.SerializeToString())

        # generate cluster centroids base graph tsv file
        nsl.tools.build_graph(embedding_files=[temp_tfr_file_path],
                              output_graph_path=temp_tsv_graph_path,
                              similarity_threshold=similarity_threshold,
                              id_feature_name='id',
                              embedding_feature_name='representation')

        # load dataset with batch_size at 1
        data_org = cls.parse_tfr_to_image_rep(path_list=[data_tfr_rep_path], rep_dim=represent_dim, batch_size=1)

        # generate one complete graph data
        centroids = tf.expand_dims(centroids, axis=1)
        for cluster_index, cluster in enumerate(tf.range(0, KMeans_model.get_K(), 1)):
            cluster_tfr_path = f'{path}{id_prefix}_cluster_{cluster_index + 1}.tfr'
            cluster_graph_path = f'{path}{id_prefix}_cluster_{cluster_index + 1}_graph.tsv'
            # generate one cluster tfr files
            with tf.io.TFRecordWriter(cluster_tfr_path) as writer:
                for index, dt in enumerate(data_org):
                    data = tf.expand_dims(dt['representation'], axis=0)
                    distance = tf.sqrt(tf.reduce_sum(tf.pow((data - centroids), 2), axis=2))
                    expected_k = tf.math.argmin(distance).numpy()[0]
                    if expected_k == cluster_index:
                        str_id = tf.compat.as_str_any(dt['id'].numpy()[0])
                        represent_id = tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[f'{str_id}'.encode('utf-8')]))
                        represent = tf.train.Feature(
                            float_list=tf.train.FloatList(value=dt['representation'].numpy()[0].tolist()))
                        feature = {'id': represent_id,
                                   'representation': represent, }
                        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(tf_example.SerializeToString())
            # generate one cluster graph tsv
            nsl.tools.build_graph(embedding_files=[cluster_tfr_path],
                                  output_graph_path=cluster_graph_path,
                                  similarity_threshold=similarity_threshold,
                                  id_feature_name='id',
                                  embedding_feature_name='representation')
            # parse one cluster graph tsv and save it into a list
            cluster_graph = nsl.tools.read_tsv_graph(cluster_graph_path)
            for item in cluster_graph.items():
                for item_II in item[1].items():
                    edge_item = [item[0], item_II[0], item_II[1]]
                    all_cluster_edges_list.append(edge_item)

        # merge all cluster graphs into base centroid graph
        complete_graph = nsl.tools.read_tsv_graph(temp_tsv_graph_path)
        for edge in all_cluster_edges_list:
            nsl.tools.add_edge(complete_graph, edge)
        # save graph data into .tsv file
        if file_output_path:
            nsl.tools.write_tsv_graph(filename=file_output_path, graph=complete_graph)
        return complete_graph

    @classmethod
    def generate_complete_graph_with_cluster_kmeans(cls,
                                                    tfr_rep_path_list,
                                                    prefix_list,
                                                    model_list,
                                                    represent_dim,
                                                    file_output_path,
                                                    similarity_threshold=0.95):
        """
         The method to generate the complete graph for the whole dataset with KMeans clusters

         Args:
             tfr_rep_path_list: the tfr file path of data representation
             file_output_path: the .tsv file path to save graph data
             represent_dim: the dimension of data representation
             prefix_list: prefix of id to differentiate labels and normally set up as the label value
             model_list: the list of the trained KMeans model
             similarity_threshold: cosine similarity threshold to decide if two nodes are connected
        Return:
            The complete graph data
        """
        global final_graph
        graph_list = []
        for path_index, path in enumerate(tfr_rep_path_list):
            graph = cls.generate_single_graph_with_cluster_kmeans(data_tfr_rep_path=path,
                                                                  represent_dim=represent_dim,
                                                                  id_prefix=prefix_list[path_index],
                                                                  KMeans_model=model_list[path_index],
                                                                  similarity_threshold=similarity_threshold)
            graph_list.append(graph)
        # merge all graphs
        for graph_index, graph in enumerate(graph_list):
            all_cluster_edges_list = []
            if graph_index == 0:
                final_graph = graph_list[graph_index]
            else:
                for item in graph_list[graph_index].items():
                    for item_II in item[1].items():
                        edge_item = [item[0], item_II[0], item_II[1]]
                        all_cluster_edges_list.append(edge_item)
            for edge in all_cluster_edges_list:
                nsl.tools.add_edge(final_graph, edge)
        # save the final graph
        if file_output_path:
            nsl.tools.write_tsv_graph(filename=file_output_path, graph=final_graph)
        return final_graph

    @classmethod
    def generate_single_graph_with_cluster_hierarch(cls,
                                                    data_tfr_rep_path,
                                                    represent_dim,
                                                    id_prefix,
                                                    hierarchy_model=None,
                                                    file_output_path=None,
                                                    similarity_threshold=0.95):
        """
            The method to generate a graph data for one label of image represents

            Args:
                data_tfr_rep_path: the tfr file path of data representation
                file_output_path: the .tsv file path to save graph data
                represent_dim: the dimension of data representation
                id_prefix: prefix of id to differentiate labels and normally set up as the label value
                hierarchy_model: the trained hierarchy_model cluster model
                similarity_threshold: cosine similarity threshold to decide if two nodes are connected
            Return:
                The single graph data
        """
        all_cluster_edges_list = []
        # generate a temp folder to hold all intermediate files
        path = './temp_graph_processing/'
        if not os.path.isdir(path):
            os.mkdir(path)
        temp_tfr_file_path = f'{path}temp.tfr'
        temp_tsv_graph_path = f'{path}graph.tsv'
        # add centroid_mean into the cluster centroids
        num_cluster, _ = hierarchy_model.search_clusters()
        cluster_list, _ = hierarchy_model.get_clusters()
        centroids_complete = hierarchy_model.get_centroids(cluster_list)

        # generate cluster centroids tfr file
        with tf.io.TFRecordWriter(temp_tfr_file_path) as writer:
            for index, center in enumerate(centroids_complete):
                represent_id = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[f'{id_prefix}c{index}'.encode('utf-8')]))
                represent = tf.train.Feature(float_list=tf.train.FloatList(value=center.numpy().tolist()))
                feature = {'id': represent_id,
                           'representation': represent, }
                tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(tf_example.SerializeToString())

        # generate cluster centroids base graph tsv file
        nsl.tools.build_graph(embedding_files=[temp_tfr_file_path],
                              output_graph_path=temp_tsv_graph_path,
                              similarity_threshold=similarity_threshold,
                              id_feature_name='id',
                              embedding_feature_name='representation')

        # load dataset with batch_size at 1
        data_org = GraphDataProcess.parse_tfr_to_image_rep(path_list=[data_tfr_rep_path], rep_dim=represent_dim,
                                                           batch_size=1)
        # generate one complete graph data
        for cluster_index, cluster in enumerate(tf.range(0, num_cluster, 1)):
            cluster_tfr_path = f'{path}{id_prefix}_cluster_{cluster_index + 1}.tfr'
            cluster_graph_path = f'{path}{id_prefix}_cluster_{cluster_index + 1}_graph.tsv'
            # generate one cluster tfr files
            with tf.io.TFRecordWriter(cluster_tfr_path) as writer:
                for index, dt in enumerate(data_org):
                    data = dt['representation']
                    for cluster_rep in cluster_list[cluster_index]:
                        if tf.reduce_all(data == cluster_rep):
                            str_id = tf.compat.as_str_any(dt['id'].numpy()[0])
                            represent_id = tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[f'{str_id}'.encode('utf-8')]))
                            represent = tf.train.Feature(
                                float_list=tf.train.FloatList(value=dt['representation'].numpy()[0].tolist()))
                            feature = {'id': represent_id,
                                       'representation': represent, }
                            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                            writer.write(tf_example.SerializeToString())
                            break

            # generate one cluster graph tsv
            nsl.tools.build_graph(embedding_files=[cluster_tfr_path],
                                  output_graph_path=cluster_graph_path,
                                  similarity_threshold=similarity_threshold,
                                  id_feature_name='id',
                                  embedding_feature_name='representation')

            # parse one cluster graph tsv and save it into a list
            cluster_graph = nsl.tools.read_tsv_graph(cluster_graph_path)
            for item in cluster_graph.items():
                for item_II in item[1].items():
                    edge_item = [item[0], item_II[0], item_II[1]]
                    all_cluster_edges_list.append(edge_item)

        # merge all cluster graphs into base centroid graph
        complete_graph = nsl.tools.read_tsv_graph(temp_tsv_graph_path)
        for edge in all_cluster_edges_list:
            nsl.tools.add_edge(complete_graph, edge)

        # save graph data into .tsv file
        if file_output_path:
            nsl.tools.write_tsv_graph(filename=file_output_path, graph=complete_graph)
        return complete_graph

    @classmethod
    def generate_complete_graph_with_cluster_hierarch(cls,
                                                      tfr_rep_path_list,
                                                      prefix_list,
                                                      model_list,
                                                      represent_dim,
                                                      file_output_path,
                                                      similarity_threshold):
        """
            The method to generate a graph data for one label of image represents

            Args:
                tfr_rep_path_list: the tfr file path of data representation
                file_output_path: the .tsv file path to save graph data
                represent_dim: the dimension of data representation
                prefix_list: prefix of id to differentiate labels and normally set up as the label value
                model_list: the trained hierarchy_model cluster model
                similarity_threshold: cosine similarity threshold to decide if two nodes are connected
            Return:
                The complete graph data
        """
        global final_graph
        graph_list = []
        for path_index, path in enumerate(tfr_rep_path_list):
            graph = cls.generate_single_graph_with_cluster_hierarch(data_tfr_rep_path=path,
                                                                    represent_dim=represent_dim,
                                                                    id_prefix=prefix_list[path_index],
                                                                    hierarchy_model=model_list[path_index],
                                                                    similarity_threshold=similarity_threshold)
            graph_list.append(graph)
        # merge all graphs
        for graph_index, graph in enumerate(graph_list):
            all_cluster_edges_list = []
            if graph_index == 0:
                final_graph = graph_list[graph_index]
            else:
                for item in graph_list[graph_index].items():
                    for item_II in item[1].items():
                        edge_item = [item[0], item_II[0], item_II[1]]
                        all_cluster_edges_list.append(edge_item)
            # add all edges to the graph
            for edge in all_cluster_edges_list:
                nsl.tools.add_edge(final_graph, edge)
        # save the final graph
        if file_output_path:
            nsl.tools.write_tsv_graph(filename=file_output_path, graph=final_graph)
        return final_graph
