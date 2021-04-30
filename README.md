# Detection of Alzheimer’s Disease Using Graph-Regularized Convolutional Neural Network Based on Structural Similarity Learning of Brain Magnetic Resonance Images
![Hex.pm](https://img.shields.io/hexpm/l/plug?logo=Apache&logoColor=%23ff0000&style=flat-square)

## About The Project 
This is a graduate project for my master degree study, and this project is updated as a final version. The code is open for the 
committee to review the paper for the publication. Paper: "Detection of Alzheimer’s Disease Using Graph-Regularized Convolutional Neural Network Based on 
Structural Similarity Learning of Brain Magnetic Resonance Images" by Kuo Yang, Emad A. Mohammed, Behrouz H. Far

#### Built With 
![AD Project Diagrams - head images](https://user-images.githubusercontent.com/55723894/99138079-1c32a700-25fc-11eb-9163-95a1526639e8.jpeg)
* [TensorFlow 2.3.0](https://www.tensorflow.org/)
* [Keras 2.4.0](https://www.tensorflow.org/api_docs/python/tf/keras)
* [Neural Structured Learning 1.3.1](https://www.tensorflow.org/neural_structured_learning)
* [Numpy](https://numpy.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)

## MRI Data
The MRI dataset could be downloaded from the Open Access Series OF Imaging Studies (OASIS) and here is the [link](https://www.oasis-brains.org/). The similiar data set could also
be found [here](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images). The dataset contains both train and test folders. Either train or test folder has sub-folders as "MildDemented", "ModerateDemented", "NonDemented" and 
"VeryMildDemented". To match the entry point of data pipeline, the user needs to merge both train and test folders in terms of each sub-folders into one folder.

## Experiment Setup
**NOTE**: The paper use the result of experiment I and II to create the tables. The experiment I is composed of three notebooks testing different dimension reduction algorithms 
(tSNE, VAE and AAE). The experiment II is to improve the model performance with multiple pre-trained model and model tuning. 

#### Notebook Content
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
    ├── Experiment_I_VAE.ipnyb                         # For Experiment I 
    ├── Experiment_I_tSNE.ipnyb                        # For Experiment I 
    ├── Experiment_I_AAE.ipnyb                         # For Experiment I  
    ├── Experiment_II_data_preprocessing.ipnyb         # For Experiment II
    ├── Experiment_II_generate_graph_data.ipnyb        # For Experiment II
    └── Experiment_II_train_model.ipnyb                # For Experiment III 

#### Notebook Setup
All available notebooks are tested on the Google Colaboratory (Recommended), and they could also run on the Jupyter notebook. For either platforms, the root 
path has to be specified based on the data directory, and the label list must be keep as the notebook. 

## Citing This Work
```
@article{
  title={Precise Detection of Alzheimer's Disease With Graph-Regularized Learning},
  author={Kuo Yang, Emad Mohammed, Behrouz H. Far},
  affiliation={Deparment of Mechanical Engineering, Deparment of Software Engineering, Lakehead University,  Department of Electrical and Computer Engineering
, Calgary University}
  year={2020}
}
```
## Contact
* K. Yang: kyang3@lakeheadu.ca
* E. Mohammed: emohamme@lakeheadu.ca
* Behrouz H. Far: far@ucalgary.ca
