
# Pixelicious

Pixelicious is a computer vision project designed for food image classification using state-of-the-art neural networks. Leveraging cutting-edge deep learning techniques, Pixelicious can accurately categorize diverse food items into 101 distinct classes. From appetizing desserts to savory main courses, Pixelicious provides a delightful solution for automating food recognition, offering a seamless and visually appetizing experience

# Introduction

The inception of Pixelicious was spurred by the ambitious goal of surpassing the benchmarks set by the renowned DeepFood paper. This computer vision endeavor harnesses the power of neural networks to adeptly categorize food images into an extensive array of 101 distinct classes. Developed during the Zero to Mastery Tensorflow course, this project served as an immersive journey into the intricacies of deep learning. Throughout its creation, I delved into diverse techniques, from acquiring preprocessed data using the TensorFlow-datasets library to the art of leveraging pre-trained models through transfer learning. The project unveiled the nuanced process of fine-tuning these models for enhanced performance. Additionally, I gained insights into the pivotal role of data augmentation, witnessing its profound impact on fortifying model generalization capabilities.

# Data

The Food-101 dataset, a pivotal resource in both the DeepFood and Food-101 papers, stands as a comprehensive benchmark for food image classification tasks. Comprising over 100,000 images spanning 101 food categories, the dataset captures a diverse array of culinary delights, ranging from appetizers to desserts. Each class boasts a substantial number of high-resolution images, providing a rich and varied collection for training and evaluation purposes. Notably, the Food-101 dataset serves as a critical asset for assessing the performance of deep learning models in the domain of food recognition. Its meticulous curation and expansive scope make it a cornerstone in the exploration of computer vision applications within the realm of gastronomy. Both the DeepFood and Food-101 papers leverage this dataset to showcase the capabilities and advancements of their respective approaches in the challenging task of food image classification

# Model

The model used is EfficientNetB0

* **Input Layer**: Confirms input tensor shape: (224, 224, 3).

* **EfficientNetB0 Base Model**: Utilizes the EfficientNetB0 architecture through the Keras API and also implemented transfer learning with pre-trained weights from the ImageNet dataset and 
Fine-tuned weights and biases to enhance adaptability to specific data.

* **GlobalAveragePooling2D Layer**:Computes the average of all numbers in the preceding layer and condenses the output into a (1, 3) tensor also Addresses the abundance of numbers in Convolutional Neural Networks (CNNs) to expedite predictions.

* **Dense Layer**: Incorporates 1 neuron per class. Hence in total 101 neurons since 101 classes were present

* **Activation Layer**: Utilizes softmax activation for multi-class classification and Separates activation layer from the Dense layer for compatibility with TensorFlow and Keras mixed-precision feature

* **Model Configuration**: Employs categorical crossentropy as the loss function for model evaluation and Utilizes the Adam optimizer for weight and parameter updates. Achieved approximately 77% accuracy on the test set.

# Contents Of Repository

* **models**: Contains models made for the project and classification. feature_extraction_model_with_EfficientNetB0.h5 is the model used for feature extraction. The fine-tuned model is Fine_tuned_EfficientNetB0_model.h5
* **scripts**: contains python script of helper function used in this project.
* **notebooks**: Contains jupyter notebooks used for this project. The notebook contains analysis of the dataset and the modelling experiments performed.
* **requirements.txt**:This file contains information about the conda environment used to create this project.

