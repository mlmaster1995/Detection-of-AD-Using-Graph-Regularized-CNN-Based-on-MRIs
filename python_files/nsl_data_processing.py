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
import random
import os
from copy import deepcopy

# Constants used to identify neighbor features in the input.
NBR_FEATURE_PREFIX = 'NL_nbr_'
NBR_WEIGHT_SUFFIX = '_weight'


class GenerateTrainTestDict(object):
    """
    To generate train and test dict data set with tf.train.Example instances

    Args:
        path_list: a list of image folder path
        TRAIN_PERCENTAGE: A float indicating the percentage of shuffle examples over the dataset.

    Returns:
        train_examples: A dict with keys being example IDs (string) and values being `tf.train.Example` instances.
        test_examples: A dict with keys being example IDs (string) and values being 'tf.train.Example` instances.
    """

    @staticmethod
    def bytes_feature(value):
        """Returns bytes tf.train.Feature from a string."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def int64_feature(value):
        """Returns int64 tf.train.Feature from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @classmethod
    def get_train_test_dict(cls, path_list, train_percentage):
        # Fixes the random seed so the train/test split can be reproduced.
        random.seed(1)
        train_examples = {}
        test_examples = {}

        for path_index, path in enumerate(path_list):
            name_list = os.listdir(path)
            for name_index, name in enumerate(name_list):
                image_id = name.split('.')[0]
                image_label = path_index
                image_string = open(f'{path}{name}', 'rb').read()

                feature = {
                    'id': cls.bytes_feature(image_id.encode('utf-8')),
                    'tensor': cls.bytes_feature(image_string),
                    'label': cls.int64_feature(image_label),
                }
                example_features = tf.train.Example(features=tf.train.Features(feature=feature))

                if random.uniform(0, 1) <= train_percentage:
                    train_examples[image_id] = example_features
                else:
                    test_examples[image_id] = example_features

        return train_examples, test_examples


class NSLDataFormat(object):
    """
     Joins the `seeds` and `nbrs` Examples using the edges in `graph`. This generator joins and augments each labeled
     Example in `seed_exs` with the features of at most `max_nbrs` of the seed's neighbors according to the given
     `graph`, and yields each merged result.

     Args:
         seed_dict_examples: A dictionary with examples as a base for NSL data, such as train_example generated from
         GenerateTrainTestDict class;
         rem_dict_examples: A remaining dictionary with examples, such as test_example generated from
         GeneratedTrainTestDict class;
         graph: A `dict`: source -> (target, weight);
         max_nb's: The maximum number of neighbors to merge into each seed Example, or `None` if the number of neighbors
         per node is unlimited.

     Return:
        N/A
    """

    def __init__(self, seed_dict_examples=None, rem_dict_examples=None, graph=None, max_nbrs=None):
        self.__seed_exs = seed_dict_examples
        self.__rem_exs = rem_dict_examples
        self.__graph = graph
        self.__max_nbrs = max_nbrs

    def _lookup_node(self, node_id):
        """Returns the node features from `__seed_exs` or `__rem_exs` with the given ID."""
        return self.__seed_exs[node_id] if node_id in self.__seed_exs else self.__rem_exs[node_id]

    def _merge_seed_nbrs(self, seed_ex, nbr_wt_ex_list):
        """
        Merges neighbor Examples into the given seed Example `seed_ex`.

        Args:
          seed_ex: A labeled Example.
          nbr_wt_ex_list: A list of (nbr_wt, nbr_id) pairs (in decreasing nbr_wt order) representing the neighbors of
          'seed_ex'.

        Returns:
          The Example that results from merging the features of the neighbor Examples (as well as creating a feature for
          each neighbor's edge weight) into `seed_ex`. See the `join()` description above for how the neighbor features
          are named in the result.
        """
        # Make a deep copy of the node to a new tensor variable.
        merged_ex = tf.train.Example()
        merged_ex.CopyFrom(seed_ex)

        # Add a feature for the number of neighbors.
        merged_ex.features.feature['NL_num_nbrs'].int64_list.value.append(len(nbr_wt_ex_list))

        # Enumerate the neighbors, and merge in the features of each.
        for index, (nbr_wt, nbr_id) in enumerate(nbr_wt_ex_list):
            prefix = f'{NBR_FEATURE_PREFIX}{index}'  # NBR_FEATURE_PREFIX = 'NL_nbr_index'
            # add weight value into the seed node map
            weight_feature = prefix + NBR_WEIGHT_SUFFIX  # NBR_WEIGHT_SUFFIX = '_weight' -> 'NL_nbr_index_weight'
            merged_ex.features.feature[weight_feature].float_list.value.append(nbr_wt)

            # Copy each of the neighbor Examples features, prefixed with 'prefix'.
            nbr_ex = self._lookup_node(nbr_id)
            for (feature_name, feature_val) in nbr_ex.features.feature.items():
                # 'NL_nbr_index_"feature_name"'
                new_feature = merged_ex.features.feature[prefix + '_' + feature_name]
                new_feature.CopyFrom(feature_val)
        return merged_ex

    def _has_node(self, node_id):
        """Returns true if 'node_id' is in the '__seed_exs' or '__rem_exs dict'."""
        result = (node_id in self.__seed_exs) or (node_id in self.__rem_exs)
        if not result:
            # logging.warning('No tf.train.Example found for edge target ID: "%s"', node_id)
            print('No tf.train.Example found for edge target ID: "%s"', node_id)
        return result

    def _get_seed_nbrs(self, seed_id):
        """
        Joins the seed with ID `seed_id` to its out-edge __graph neighbors. This also has the side-effect of maintaining
        the `__out_degree_count`.

        Args:
            seed_id: The ID of the seed Example to start from.
            __max_nbrs: maximum node neighbour of each seed

        Returns:
            A list of (nbr_wt, nbr_id) pairs (in decreasing weight order) of the seed Example's top `__max_nbrs`
            neighbors. So the resulting list will have size at most `__max_nbrs`, but it may be less (or even empty if
            the seed Example has no out-edges).
        """
        nbr_dict = self.__graph[seed_id] if seed_id in self.__graph else {}
        nbr_wt_ex_list = [(nbr_wt, nbr_id) for (nbr_id, nbr_wt) in nbr_dict.items() if self._has_node(nbr_id)]
        temp_list = deepcopy(nbr_wt_ex_list)
        if len(temp_list) != self.__max_nbrs and len(temp_list) > 0:
            for _ in range(self.__max_nbrs - len(temp_list)):
                nbr_wt_ex_list.append(temp_list[random.randint(0, len(temp_list) - 1)])
        result = sorted(nbr_wt_ex_list, reverse=True)[:self.__max_nbrs]
        return result

    def node_nbrs_generator_test(self):
        for (seed_id, seed_ex) in self.__seed_exs.items():
            if seed_id in self.__graph:
                return self._merge_seed_nbrs(seed_ex, self._get_seed_nbrs(seed_id))

    def _node_nbrs_generator(self):
        for (seed_id, seed_ex) in self.__seed_exs.items():
            if seed_id in self.__graph:
                yield self._merge_seed_nbrs(seed_ex, self._get_seed_nbrs(seed_id))

    def generate_node_nbrs_tfr(self, output_file_path):
        with tf.io.TFRecordWriter(output_file_path) as writer:
            for merged_example in self._node_nbrs_generator():
                writer.write(merged_example.SerializeToString())

    @staticmethod
    def _parse_example_message(example_proto, max_neighbor_number):
        """
            Args:
                example_proto: An instance of `tf.train.Example`, a single tf.Example message
                hparams: hyper-parameters defined in HyperParams class

            Return:
                A pair whose first value is a dictionary containing relevant features and whose second value contains
                the ground truth label
        """
        # The 'words' feature is a multi-hot, bag-of-words representation of the original raw text. A default value is
        # required for examples that don't have the feature.
        feature_spec = {
            'id': tf.io.FixedLenFeature([], tf.string),
            # 'NL_num_nbrs': tf.io.FixedLenFeature([],tf.int64),
            'tensor': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }

        # We also extract corresponding neighbor features in a similar manner to the features above.
        for i in range(max_neighbor_number):
            nbr_feature_key = f'{NBR_FEATURE_PREFIX}{i}_tensor'  # NBR_FEATURE_PREFIX = 'NL_nbr_'->'NL_nbr_index_tensor'
            nbr_weight_key = f'{NBR_FEATURE_PREFIX}{i}{NBR_WEIGHT_SUFFIX}'  # NBR_WEIGHT_SUFFIX = '_weight'
            nbr_id_key = f'{NBR_FEATURE_PREFIX}{i}_id'  # 'NL_nbr_index_id'

            feature_spec[nbr_feature_key] = tf.io.FixedLenFeature([], tf.string)
            feature_spec[nbr_weight_key] = tf.io.FixedLenFeature([1], tf.float32, default_value=tf.constant([0.0]))
            feature_spec[nbr_id_key] = tf.io.FixedLenFeature([], tf.string)

        features = tf.io.parse_single_example(example_proto, feature_spec)
        # label = features.pop('label')  # label is removed from the parsed features
        return features

    @staticmethod
    def _parse_image_string(x, max_neighbor_number, image_size, image_channels=3):
        """Parse each image_raw to image_tensor"""
        # convert from tf.string to image tensor
        def parse_image(ipt, size, nme, channels):
            parsed_image_data = tf.image.decode_jpeg(ipt[nme], channels=channels)
            parsed_image_data = tf.image.resize(parsed_image_data, size=size)
            parsed_image_data = parsed_image_data / 255
            return parsed_image_data

        # parse seed image tensor
        image_tensor = parse_image(ipt=x, size=image_size, nme='tensor', channels=image_channels)
        x['tensor'] = image_tensor
        # parse nbr image tensor of the seed
        for i in range(max_neighbor_number):
            name = f'{NBR_FEATURE_PREFIX}{i}_tensor'  # 'NL_nbr_index_tensor'
            image_tensor_nbr = parse_image(ipt=x, size=image_size, nme=name, channels=image_channels)
            x[name] = image_tensor_nbr
        label = x.pop('label')  # label is removed from the parsed features
        label = tf.one_hot(label, depth=4)
        return x, label

    @classmethod
    def parse_tfr_to_dataset(cls, file_path_list, batch_size, max_neighbor_number=5,
                             image_size=(208, 176),
                             image_channels=3,
                             shuffle=False):
        """
            generate a `tf.data.TFRecordDataset` from a list of tfr files

            Args:
              file_path_list: a list of tfr files
              shuffle: boolean value to check if dataset needs to be shuffled
              batch_size: batch size of data
              max_neighbor_number: max neighbour nodes
              image_size: (weight, height)

            Returns:
              An instance of `tf.data.TFRecordDataset` containing the `tf.train.Example` objects.
        """
        dataset = tf.data.TFRecordDataset(file_path_list)
        if shuffle:
            dataset = dataset.shuffle(10000)

        dataset = dataset.map(lambda x: cls._parse_example_message(example_proto=x,
                                                                   max_neighbor_number=max_neighbor_number))
        image_dataset = dataset.map(lambda x: cls._parse_image_string(x,
                                                                      max_neighbor_number=max_neighbor_number,
                                                                      image_channels=image_channels,
                                                                      image_size=image_size))
        image_dataset = image_dataset.batch(batch_size)
        return image_dataset

    @staticmethod
    def __parse_example_message_test(example_proto, max_neighbor_number):
        """
        Args:
            example_proto: An instance of `tf.train.Example`, a single tf.Example message
            hparams: hyper-parameters defined in HyperParams class
        Return:
            A pair whose first value is a dictionary containing relevant features and whose second value
            contains the ground truth label
        """
        # The 'words' feature is a multi-hot, bag-of-words representation of the original raw text. A default value is
        # required for examples that don't have the feature.
        feature_spec = {
            'tensor': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'id': tf.io.FixedLenFeature([], tf.string),
        }

        # We also extract corresponding neighbor features in a similar manner to the features above.
        for i in range(max_neighbor_number):
            nbr_feature_key = f'{NBR_FEATURE_PREFIX}{i}_tensor'  # NBR_FEATURE_PREFIX = 'NL_nbr_'
            nbr_weight_key = f'{NBR_FEATURE_PREFIX}{i}{NBR_WEIGHT_SUFFIX}'  # NBR_WEIGHT_SUFFIX = '_weight'

            feature_spec[nbr_feature_key] = tf.io.FixedLenFeature([], tf.string)
            feature_spec[nbr_weight_key] = tf.io.FixedLenFeature([1], tf.float32, default_value=tf.constant([0.0]))

        features = tf.io.parse_single_example(example_proto, feature_spec)
        label = features.pop('label')  # label is removed from the parsed features
        return features, label

    @classmethod
    def __parse_tfr_to_dataset_test(cls, file_path_list, batch_size, max_neighbor_number, image_size, shuffle=False):
        """
            generate a `tf.data.TFRecordDataset` from a list of tfr files

            Args:
              file_path_list: a list of tfr files
              shuffle: boolean value to check if dataset needs to be shuffled

            Returns:
              An instance of `tf.data.TFRecordDataset` containing the `tf.train.Example` objects.
        """
        dataset = tf.data.TFRecordDataset(file_path_list)
        if shuffle:
            dataset = dataset.shuffle(10000)

        dataset = dataset.map(lambda x: cls._parse_example_message_test(example_proto=x,
                                                                        max_neighbor_number=max_neighbor_number))
        image_dataset = dataset.batch(batch_size)
        return image_dataset

    @staticmethod
    def generate_test_tfr_image_tensor(path_list, batch_size, channels=3, size=(208, 176), shuffle=False):
        """
            generate tf.data.Dataset with format {id: **, label: **, tensor: **}

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
                    'tensor': image_tensor, }

        raw_image_dataset = tf.data.TFRecordDataset(path_list)
        parsed_image_dataset = raw_image_dataset.map(parse_image_function)
        image_dataset = parsed_image_dataset.map(parse_image_raw)
        if shuffle:
            image_dataset = image_dataset.shuffle(buffer_size=20000, seed=None, reshuffle_each_iteration=shuffle)

        image_dataset = image_dataset.map(lambda x: (x, tf.one_hot(x.pop("label"), depth=4)))
        image_batch_dataset = image_dataset.batch(batch_size)
        return image_batch_dataset
