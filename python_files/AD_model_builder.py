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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd


class AD_params(object):
    """
    # AD_params class is a container for all hyper-parameters involved by the AD model building and training process.
    # This class is heavily used in the Expriement II to simplify the training and tuning process in the notebook.
    """

    def __init__(self, ):
        self.label_list = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        self.root_path = None
        self.image_path_list = None
        self.tfr_file_path_list = None
        self.image_rep_tfr_file_path_list = None
        self.train_tfr_path = None
        self.test_tfr_path = None
        self.graph_path = None
        self.image_size = (100, 100)
        self.image_channels = 3
        self.max_seed_nbr = 5
        self.batch_size = 128
        self.early_stop_base_line = 0.92
        self.train_epoch = 50
        self.checkpoint_path = None
        self.checkpoint_restore_path = None
        self.learning_rate = 0.001
        self.shuffle = None
        self.neighbor_prefix = "NL_nbr_"
        self.neighbor_weight_suffix = '_weight'
        self.ad_base_model = 'vgg19'
        self.ad_base_model_input_shape = (100, 100, 3)
        self.nsl_multiplier = 0.1
        self.nsl_sum_over_axis = -1
        self.nsl_distance_type = None


class AccEarlyStop(tf.keras.callbacks.EarlyStopping):
    """
    # AccEarlyStop class extends from tf.keras.callbacks.EarlyStopping.
    # At the end of each epoch, the best model weights are restored and also check if the model performance reaches the
    earlyStopping threshold as val_accuracy
    # At the tend of each epoch training process, the earlyStopping is triggered based on current epoch val_accuracy
    """

    def __init__(self, val_acc_base):
        """
        # val_acc_base: the threshold acc for early stop training between 0 and 1
        """
        super(AccEarlyStop, self).__init__(monitor='val_accuracy',
                                           verbose=1,
                                           baseline=val_acc_base,
                                           restore_best_weights=True)
        self.__best_weights = None
        self.__bestWeightEpoch = None
        self.__weights = []
        self.__val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        # restore best model weights
        if self.restore_best_weights:
            # save weights & val_acc for each epoch
            self.__weights.append(self.model.get_weights())
            self.__val_acc.append(logs['val_accuracy'])

            # update the best weights
            self.__bestWeightEpoch = self.__val_acc.index(max(self.__val_acc))
            self.__best_weights = self.__weights[self.__bestWeightEpoch]

        # early stopping check
        if logs[self.monitor] >= self.baseline:
            self.model.stop_training = True
            self.stopped_epoch = epoch + 1

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        else:
            self.model.set_weights(self.__best_weights)
            print(f'Early stopping is not triggered, but best model is restored at epoch {self.__bestWeightEpoch + 1}')


class ADModelBuilder(object):
    """
    ADModelBuilder class is a wrapper for all pretrained models involved in this project including VGG19, DenseNet121,
    and Xception. This class could load the pretrained model with non-trainable imagenet weights from TensorFlow, and
    also unfreeze or freeze layers by layer name (VGG19 only) or layer numbers (VGG19, DenseNet121 and Xception)

    ADModelBuilder also has two static methods for both confusion matrix computation and the confusion matrix plotting,
    both of which are developed via scikit-learn
    """

    def __init__(self, input_shape=(100, 100, 3), base_model='vgg19'):
        """
        Args:
            input_shape: MRI image shape and default is 3-channel with size at 100 * 100
            base_model: it must be one of "vgg19", "densenet121" and "xception"
        """
        self._input_shape = input_shape
        self._base_model = base_model
        self._vgg19 = None
        self._ad_model = None
        self._densenet121 = None
        self._xception = None

        if base_model == 'vgg19':
            print(
                'Note: default vgg19 not includes top and has imagenet weights. It is not a trainable model in '
                'ADModelBuilder. Setup specific trainable layers in setup_VGG19_by_layer_names or setup_VGG19 '
                'function based on the VGG19 layer name or layer number')
        if base_model == 'densenet121':
            print(
                'Note: default DenseNet121 not includes top and has imagenet weights. It is not a trainable model in '
                'ADModelBuilder. Please setup trainable layers in setup_DenseNet121 function based on the "Layer '
                'Number"')
        if base_model == 'xception':
            print(
                'Note: default Xception not includes top and has imagenet weights. It is not a trainable model in '
                'ADModelBuilder. Please setup trainable layers in setup_Xception function based on the "Layer '
                'Number"')

    def setup_VGG19_by_layer_names(self, trainable_layers=[None]):
        """
        The method to setup certain layer weights as trainable or non-trainable via layer numbers and it's only for
        VGG19 pretrained model. All the layer name could be acquired by calling get_VGG19_layer_names method.

        Args:
            trainable_layers: a list of layer names, and all names could be from get_VGG19_layer_names method.
        Return:
            self
        """
        self._vgg19 = None
        self._vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=self._input_shape)
        for name in trainable_layers:
            for layer in self._vgg19.layers:
                if layer.name == name:
                    print('trainable layer: ', name)
                    layer.trainable = True
                else:
                    layer.trainable = False
        self._vgg19.summary()
        return self

    def setup_VGG19(self, top_layers=None, middle_layers=None, bottom_layers=None):
        """
        The method to setup certain layer weights as trainable or non-trainable via layer numbers.

        Args:
            top_layers: an Int value to specify the stop layer from 0 layer in one block, such as 5, and it means first
            5 layers from the beginning as trainable
            middle_layers: a list with two Int values to specify one block, such as [20, 30], and it means 20th to 30th
            layers as trainable
            bottom_layers: an Int value to specify the stop layer from final layer in one block, such as 50, and it means the
            last 50 layers as trainable

        Return: self
        """
        self._vgg19 = None
        self._vgg19 = tf.keras.applications.DenseNet121(include_top=False,
                                                        weights='imagenet',
                                                        input_shape=self._input_shape)
        for layer in self._vgg19.layers:
            layer.trainable = False
        if top_layers:
            for layer in self._vgg19.layers[0:top_layers]:
                layer.trainable = True
        if middle_layers and len(middle_layers) == 2:
            for layer in self._vgg19.layers[middle_layers[0]:middle_layers[1]]:
                layer.trainable = True
        if bottom_layers:
            for layer in self._vgg19.layers[-bottom_layers:]:
                layer.trainable = True
        self._vgg19.summary()
        return self

    # Define layer setup function via layer numbers
    def setup_DenseNet121(self, top_layers=None, middle_layers=None, bottom_layers=None):
        """
        The method to setup certain layer weights as trainable or non-trainable

        Args:
            top_layers: an Int value to specify the stop layer from 0 layer in one block, such as 5, and it means first 5
            layers from the beginning as trainable
            middle_layers: a list with two Int values to specify one block, such as [20, 30], and it means 20th to 30th
            layers as trainable
            bottom_layers: an Int value to specify the stop layer from final layer in one block, such as 50, and it means the
            last 50 layers as trainable

        Return: self
        """
        self._densenet121 = None
        self._densenet121 = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet',
                                                              input_shape=self._input_shape)
        for layer in self._densenet121.layers:
            layer.trainable = False
        if top_layers:
            for layer in self._densenet121.layers[0:top_layers]:
                layer.trainable = True
        if middle_layers and len(middle_layers) == 2:
            for layer in self._densenet121.layers[middle_layers[0]:middle_layers[1]]:
                layer.trainable = True
        if bottom_layers:
            for layer in self._densenet121.layers[-bottom_layers:]:
                layer.trainable = True
        self._densenet121.summary()
        return self

    def setup_Xception(self, top_layers=None, middle_layers=None, bottom_layers=None):
        """
        The method to setup certain layer weights as trainable or non-trainable

        Args:
            top_layers: an Int value to specify the stop layer from 0 layer in one block, such as 5, and it means first 5
            layers from the beginning as trainable
            middle_layers: a list with two Int values to specify one block, such as [20, 30], and it means 20th to 30th
            layers as trainable
            bottom_layers: an Int value to specify the stop layer from final layer in one block, such as 50, and it means the
            last 50 layers as trainable

        Return:
            self
        """
        self._xception = None
        self._xception = tf.keras.applications.Xception(include_top=False, weights='imagenet',
                                                        input_shape=self._input_shape)
        for layer in self._xception.layers:
            layer.trainable = False
        if top_layers:
            for layer in self._xception.layers[0:top_layers]:
                layer.trainable = True
        if middle_layers and len(middle_layers) == 2:
            for layer in self._xception.layers[middle_layers[0]:middle_layers[1]]:
                layer.trainable = True
        if bottom_layers:
            for layer in self._xception.layers[-bottom_layers:]:
                layer.trainable = True
        self._xception.summary()
        return self

    def get_VGG19_layer_names(self):
        """
        The method to acquire names of trainable layer names
        """
        if self._base_model == 'vgg19':
            return ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2',
                    'block3_conv3', 'block3_conv4', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4',
                    'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4', ]
        else:
            print("Note: base_model is not vgg19 and no names return...")

    def get_VGG19(self):
        """
        The method to acquire the VGG19 model. If the model is setup via layer name or layer number already,
        the model with certain trainable weights will be returned. If the model is not setup via layer name or layer
        number, the model with non-trainable weights will be returned.
        """
        if self._vgg19:
            return self._vgg19
        else:
            model = tf.keras.applications.VGG19(include_top=False, weights='imagenet',
                                                input_shape=self._input_shape)
            model.trainable = False
            return model

    def get_DenseNet121(self):
        """
        The method to acquire the DenseNet121 model. If the model is setup via layer number already,
        the model with certain trainable weights will be returned. If the model is not setup via layer number, the model
        with non-trainable weights will be returned.
        """
        if self._densenet121:
            return self._densenet121
        else:
            model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet',
                                                      input_shape=self._input_shape)
            model.trainable = False
            return model

    def get_Xception(self):
        """
        The method to acquire the Xception model. If the model is setup via layer number already, the model
        with certain trainable weights will be returned. If the model is not setup via layer number, the model with
        non-trainable weights will be returned.
        """
        if self._xception:
            return self._xception
        else:
            model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=self._input_shape)
            model.trainable = False
            return model

    def get_ADModel(self):
        """
        The method to acquire the final AD model with both pretrained model and the final output layers, and the model
        could be used for training directly.
        """
        self._build_ADModel()
        return self._ad_model

    def _build_ADModel(self):
        self._ad_model = None
        if self._base_model == 'vgg19':
            if self._vgg19 is None:
                vgg19_layer = tf.keras.applications.VGG19(include_top=False, weights='imagenet',
                                                          input_shape=self._input_shape)
                vgg19_layer.trainable = False
            else:
                vgg19_layer = self._vgg19
            ipt = tf.keras.Input(shape=self._input_shape, name='tensor')
            opt = vgg19_layer(ipt)
            opt = tf.keras.layers.Flatten()(opt)
            opt = tf.keras.layers.Dense(64)(opt)
            opt = tf.keras.layers.BatchNormalization()(opt)
            opt = tf.keras.layers.LeakyReLU()(opt)
            opt = tf.keras.layers.Dropout(0.5)(opt)
            opt = tf.keras.layers.Dense(4, activation='softmax')(opt)
            self._ad_model = tf.keras.Model(ipt, opt, name="VGG19_ADModel")
            self._ad_model.summary()

        elif self._base_model == 'densenet121':
            if self._densenet121 is None:
                densenet_layer = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet',
                                                                   input_shape=self._input_shape)
                densenet_layer.trainable = False
            else:
                densenet_layer = self._densenet121
            ipt = tf.keras.Input(shape=self._input_shape, name='tensor')
            opt = densenet_layer(ipt)
            opt = tf.keras.layers.Flatten()(opt)
            opt = tf.keras.layers.Dropout(0.3)(opt)
            opt = tf.keras.layers.Dense(128)(opt)
            opt = tf.keras.layers.BatchNormalization()(opt)
            opt = tf.keras.layers.LeakyReLU()(opt)
            opt = tf.keras.layers.Dropout(0.4)(opt)
            opt = tf.keras.layers.Dense(4, activation='softmax')(opt)
            self._ad_model = tf.keras.Model(ipt, opt, name="DenseNet121_ADModel")
            self._ad_model.summary()

        elif self._base_model == 'xception':
            if self._xception is None:
                xception_layer = tf.keras.applications.Xception(include_top=False, weights='imagenet',
                                                                input_shape=self._input_shape)
                xception_layer.trainable = False
            else:
                xception_layer = self._xception
            ipt = tf.keras.Input(shape=self._input_shape, name='tensor')
            opt = xception_layer(ipt)
            opt = tf.keras.layers.Flatten()(opt)
            opt = tf.keras.layers.Dropout(0.3)(opt)
            opt = tf.keras.layers.Dense(128)(opt)
            opt = tf.keras.layers.BatchNormalization()(opt)
            opt = tf.keras.layers.LeakyReLU()(opt)
            opt = tf.keras.layers.Dropout(0.4)(opt)
            opt = tf.keras.layers.Dense(4, activation='softmax')(opt)
            self._ad_model = tf.keras.Model(ipt, opt, name="Xception_ADModel")
            self._ad_model.summary()

        else:
            print("Note: no selected base_model in the ADModelBuilder class...")

    @staticmethod
    def calc_confu_mat_for_each_label(model, data, data_label):
        """
        The static method to calculate the metrics of confusion matrix for the table in Experiment II in the project paper

        Args:
             model: The trained AD model
             data: The evaluation or test dataset
             data_label: The evaluation or test dataset labels
        """
        # get pred
        y_pre = list(tf.argmax(model.predict(data), axis=1).numpy())
        y_true = list(tf.argmax(data_label, axis=1).numpy())
        # calc confu_mat for each label
        confu_mat = multilabel_confusion_matrix(y_true, y_pre, labels=[0, 1, 2, 3])
        tn = confu_mat[:, 0, 0]
        tp = confu_mat[:, 1, 1]
        fn = confu_mat[:, 1, 0]
        fp = confu_mat[:, 0, 1]
        print('TN:', tn)
        print('TP:', tp)
        print('FN:', fn)
        print('FP:', fp)
        print('Acc:', (tp + tn) / len(data))
        print('ER(Error rate):', (fp + fn) / len(data))
        print('Recall(TP rate):', tp / (tp + fn))
        print('Specialty(TN rate):', tn / (tn + fp))
        print('Fall Out(FP rate):', fp / (fp + tn))
        print('Miss Rate(FN rate):', fn / (fn + tp))

    @staticmethod
    def plot_confusion_mat(model, data, data_label, title=None, figsize=(6, 6), labels=['ND', 'VMD', 'MD', 'MDTD'],
                           font_size=1.1, count_size=12, color_bar=False):
        """
        The static method to plot the confusion matrix in the project paper.
        """
        # get pred
        y_pre = list(tf.argmax(model.predict(data), axis=1).numpy())
        y_true = list(tf.argmax(data_label, axis=1).numpy())
        # count the pred
        mat_dict = {}
        for inx, pair in enumerate(list(zip(y_pre, y_true))):
            if pair[1] in mat_dict:
                mat_dict[pair[1]][pair[0]] += 1
            else:
                mat_dict[pair[1]] = [0, 0, 0, 0]
                mat_dict[pair[1]][pair[0]] += 1
        # generate the mat
        confu_lst = []
        for key in range(4):
            arr = np.array(mat_dict[key])
            confu_lst.append(arr)
        confu_mat = np.array(confu_lst)
        # plot the mat
        axis = labels
        df_cm = pd.DataFrame(confu_mat, axis, axis)
        plt.figure(figsize=figsize)
        sn.set(font_scale=font_size)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": count_size}, fmt="d", linewidths=.8, cmap="YlGnBu",
                   cbar=color_bar)
        plt.title(title)
        plt.show()
