# Detection of Alzheimer’s Disease Using Graph-Regularized Convolutional Neural Network Based on Structural Similarity Learning of Brain Magnetic Resonance Images
![Hex.pm](https://img.shields.io/hexpm/l/plug?logo=Apache&logoColor=%23ff0000&style=flat-square)

**NOTE**: This work is under review by the IEEE Transactions on Pattern Analysis and Machine Intelligence. If you want to use this work, you must cite it in your work.

## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Data Setup](#data-setup)
  * [MRI Dataset](#mri-dataset)
  * [Data Folder Structure](#data-folder-structure)
* [Experiment Setup](#experiment-setup)
  * [Notebook Structure](#notebook-structure)
  * [Notebook Setup](#notebook-setup)
* [Reproducing Paper Tables](#reproducing-paper-tables)
* [Citing This Work](#citing-this-work)
* [Contact](#contact)

## About The Project 
This is a graduate project for my master degree study, and this project is updated as a final version, but will keep fixing bugs. The code is open for the 
committee to review the paper for the publication. Paper: "Detection of Alzheimer’s Disease Using Graph-Regularized Convolutional Neural Network Based on 
Structural Similarity Learning of Brain Magnetic Resonance Images" by Kuo Yang, Emad A. Mohammed, Behrouz H. Far
#### Built With 
![AD Project Diagrams - head images](https://user-images.githubusercontent.com/55723894/99138079-1c32a700-25fc-11eb-9163-95a1526639e8.jpeg)
* [TensorFlow 2.3.0](https://www.tensorflow.org/)
* [Keras 2.4.0](https://www.tensorflow.org/api_docs/python/tf/keras)
* [Neural Structured Learning 1.3.1](https://www.tensorflow.org/neural_structured_learning)
* [Numpy](https://numpy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)
## Data Setup
**NOTE**: MRI dataset is too big to upload, but all tfrecords files for all labels are uploaded. All notebooks start the data pipeline from original MRIs, but all tfrecords are
processed MRI images. If the notebooks are running from the beginning, all tfrecords files will be generated. The graph data is saved ".tsv" files, but due to the big file size, 
not all graph data are uploaded and all graph data could be generated through the notebooks.   
#### MRI Dataset
The MRI dataset could be downloaded from the Open Access Series OF Imaging Studies (OASIS) and here is the [link](https://www.oasis-brains.org/). The similiar data set could also
be found [here](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images). The dataset contains both train and test folders. Either train or test folder has sub-folders as "MildDemented", "ModerateDemented", "NonDemented" and 
"VeryMildDemented". To match the entry point of data pipeline, the user needs to merge both train and test folders in terms of each sub-folders into one folder. 
#### Data Folder Structure 
The graph_images_balanced (`graph_images_balanced.tar.xz`), graph_images_balanced_II (`graph_images_balanced_II.tar.xz`), graph_images (`graph_images.tar.xz`) folders are 
uploaded to the repo with the specific structure. But MRI image folders are empty, and user needs to fill MRI images after the downloading in the folder.

`graph_images.tar.xz` has all ".tfr" files for Experiment I, but there are not ".tsv" graph files and they could be generated via running the notebook. The folder directory 
structure needs to be set up as following. 

   
    ./graph_images/
    ├── MildDemented                # MRIs from both train and test with label MildDemented
    ├── ModerateDemented            # MRIs from both train and test with label ModerateDemented
    ├── NonDemented                 # MRIs from both train and test with label NonDemented
    ├── VeryMildDemented            # MRIs from both train and test with label VeryMildDemented
    ├── MildDemented.tfr            # TFRecord file for MRIs with label MildDemented
    ├── ModerateDemented.tfr        # TFRecord file for MRIs with label ModerateDemented
    ├── NonDemented.tfr             # TFRecord file for MRIs with label NonDemented
    ├── VeryMildDemented.tfr        # TFRecord file for MRIs with label VeryMildDemented
    
`graph_images_balanced.tar.xz` has all ".tfr" files for Experiment II, but has no ".tsv" files for graph data. The folder directory structure needs to be set up as following. 


    ./graph_images_balanced/
    ├── MildDemented                # Balanced MRIs with label MildDemented
    ├── ModerateDemented            # Balanced MRIs with label ModerateDemented
    ├── NonDemented                 # Balanced MRIs with label NonDemented
    ├── VeryMildDemented            # Balanced MRIs with label VeryMildDemented
    ├── MildDemented.tfr            # TFRecord file for balanced MRIs with label MildDemented
    ├── ModerateDemented.tfr        # TFRecord file for balanced MRIs with label ModerateDemented
    ├── NonDemented.tfr             # TFRecord file for balanced MRIs with label NonDemented
    ├── VeryMildDemented.tfr        # TFRecord file for balanced MRIs with label VeryMildDemented
    ├── AD_graph_aae_kmeans.tsv     # Complete AD Graph Data 
    ├── MildDemented_AAErep.tfr     # TFRecord file for MRI representations with label MildDemented
    ├── ModerateDemented_AAErep.tfr # TFRecord file for MRI representations with label ModerateDemented
    ├── NonDemented_AAErep.tfr      # TFRecord file for MRI representations with label NonDemented
    └── VeryMildDemented_AAErep.tfr # TFRecord file for MRI representations with label VeryMildDemented

`graph_images_balanced_II.tar.xz` has all ".tfr" and ".tsv" files for Experiment III. The folder directory structure needs to be set up as following.


    ./graph_images_balanced_II/
    ├── MildDemented                        # Balanced MRIs with label MildDemented
    ├── ModerateDemented                    # Balanced MRIs with label ModerateDemented
    ├── NonDemented                         # Balanced MRIs with label NonDemented
    ├── VeryMildDemented                    # Balanced MRIs with label VeryMildDemented
    ├── test                    
    │   ├── MildDemented              # 30% MRIs with label MildDemented
    │   ├── MildDemented.tfr          
    │   ├── ModerateDemented          # 30% MRIs with label ModerateDemented     
    │   ├── ModerateDemented.tfr
    │   ├── NonDemented               # 30% MRIs with label NonDemented
    │   ├── NonDemented.tfr
    │   ├── VeryMildDemented          # 30% MRIs with label VeryMildDemented
    │   └── VeryMildDemented.tfr
    ├── train
    │   ├── AD_graph_AAE_KMeans.tsv     # the graph data for all labels 
    │   ├── MildDemented                # 70% MRIs with label MildDemented label
    │   ├── MildDemented_AAErep.tfr     # MildDemented MRI Represents 
    │   ├── MildDemented.tfr          
    │   ├── ModerateDemented            # 70% MRIs with label ModerateDemented label
    │   ├── ModerateDemented_AAErep.tfr # ModerateDemented MRI Represents
    │   ├── ModerateDemented.tfr
    │   ├── NonDemented                 # 70% MRIs with label NonDemented label
    │   ├── NonDemented_AAErep.tfr      # NonDemented MRI Represents
    │   ├── NonDemented.tfr
    │   ├── VeryMildDemented            # 70% MRIs with label VeryMildDemented label
    │   ├── VeryMildDemented_AAErep.tfr # VeryMildDemented MRI Represents
    │   └── VeryMildDemented.tfr

## Experiment Setup
**NOTE**: The paper use the result of experiment I and III to create the tables. But there are three experiment conducted in this project. The experiment I is composed of three 
notebooks testing different dimension reduction algorithms (tSNE, VAE and AAE). The experiment II and III are both to improve the model performance. However, the graph data is 
constructed with both train and test MRI images in experiment II while only train MRI images are used to construct the graph data in experiment III. The major difference is shown
in the following workflow digrams. 

**EXPERIMENT II WORKFLOW**

??????????????????????

**EXPERIMENT III WORKFLOW**

??????????????????????

#### Notebook Structure
The external python files need to be imported for both Experiment I, II and III notebooks.  
    
    
    ./python_files/
    ├── AAE_model.py
    ├── AD_model_builder.py
    ├── CNN_tSNE.py
    ├── graph_data_processing.py
    ├── Hierarchical.py
    ├── Kmeans.py
    ├── nsl_data_processing.py
    └── VAE_model.py
    
    .
    ├── Experiment_I_VAE.ipnyb      # For Experiment I 
    ├── Experiment_I_tSNE.ipnyb     # For Experiment I 
    ├── Experiment_I_AAE.ipnyb      # For Experiment I  
    ├── Experiment_II.ipnyb         # For Experiment II
    └── Experiment_III.ipnyb         # For Experiment III 

#### Notebook Setup
All available notebooks are tested on the Google Colaboratory (Recommended), and they could also run on the Jupyter notebook. For either platforms, the root 
path has to be specified based on the data directory and the label list must be keep as the notebook. 
```python
# For experiment I
label_list = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
root_path = '[user_define_path]/graph_images/'
```
```python
# For experiment II
label_list = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
root_path = '[user_define_path]/graph_images_balanced/'
```  
```python
# For experiment III 
label_list = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
root_path = '[user_define_path]/graph_images_balanced_II/' 
# Redefine root path after the train and test data is seperated
root_path = '[user_define_path]/graph_images_balanced_II/train/'                     # root_path for train data folder
root_path_test = '[user_define_path]/graph_images_balanced_II/test/'                 # root_path for test data folder                 
``` 
The notebook starts with the MRI image dataset from renaming images, generating ".tfr" files and parsing ".tfr" files. And all ".tfr" files pf each label is 
uploaded into the repo for both Experiment I, II and III, so user don't have to implement the initial transformation for ".tfr files" and following code needs to
be commented out in the notebook
```python
"""rename images in the graph_image folder"""
GraphDataProcess.rename_images(label_list, root_path) # needs to comment out 
"""Generate image tfr files"""
path_list = [f'{root_path}{label}/' for label in label_list]
tfr_list = [f'{root_path}{label}.tfr' for label in label_list]
GraphDataProcess.generate_tfr_raw(path_list=path_list, tfr_list=tfr_list)  # needs to comment out  
```
## Reproducing Paper Tables
There are 3 experiment conducted in this project. However, the experimental tables in the paper refers to the experiment I and experiment III. The experiment II is the extra 
experiment I implement to examine the effect of different graph construction on the model performance. The major workflow difference has been showed in the previous section.  
All the metrics are collected from the notebook to construct all tables. For experiment I and II, all training process and evaluation are printed in the 
three notebooks, and the evaluation might be slightly different after running the notebook, but it shouldn't have big differences.
## Citing This Work
```
@article{
  title={Precise Detection of Alzheimer's Disease With Graph-Regularized Learning},
  author={Kuo Yang, Emad Mohammed, Behrouz H. Far},
  affiliation={Deparment of Mechanical Engineering, Deparment of Software Engineering, Lakehead University, Deparment of Software Engineering, Calgary University}
  year={2020}
}
```
## Contact
* K. Yang: kyang3@lakeheadu.ca
* E. Mohammed: emohamme@lakeheadu.ca
* Behrouz H. Far: ?????
