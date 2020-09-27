# Image-Aesthetic-Assesment

Abstract:
Automated Image Aesthetic Assessment has been challenging to implement due to varied perceptions of people. This paper aims to tackle the matter and achieve better accuracy by adopting a deep learning neural network approach to perform image aesthetic classification. This research work presents a deep convolutional neural network framework that programmatically extracts high and low ranking features of an image and differentiates the dataset for analyzing areas of concern. Our model performs Image recognition using TensorFlow and Keras. A high-level network is employed to train and classify images. Additionally, the proposed model employs color contrast, depth of field, and rule of thirds to further improve the aesthetic performance of the model. This also uses GrabCut algorithm for interactive foreground extraction using OpenCV (Open Source Computer Vision Library). Our dataset, comprising 6000 images, is compiled from a range of sources online(Pinterest, Google, Flickr, Kaggle, Flickr) to make it as diverse as possible. Our experiments demonstrate that compared to traditional handcrafted models our Deep Convolutional Neural Network model yields significantly better categorization correctness (accuracy) of 73.27%. Thus, the Deep Learning Model helps exclusively to boost the performance of Aesthetic Assessment.

DATASET

Most of the research linked to this topic, evaluates their respective models particularly on individual collections of data that are difficult to access. To contribute towards the research fraternity, an expandable, usable and openly available dataset is constructed. Using various heterogeneous sources like Kaggle, Google, Pinterest, Shutterstock, Unsplash, etc, over 6000 images are gathered based on the three high- level attributes. In each of the above categories, there are 2000 images, The train to test ratio
used here is 70:30, which prevents overfitting of the model as well as provides better accuracy. For preprocessing, the images of image aesthetic dataset (IAD) were converted to
grayscale, resized to 128*128 and the dataset was standardized by scaling.

APPROACH
Previously various models have been developed based on the handcrafted method of image classification. To increase the accuracy and efficiency, a Deep Neural Network is proposed
as given below. 

1) Image Repository: Image Repository is the database consisting of train and test images with the split ratio of 70:30.
2) Deep Convolutional Neural Network: The DCNN is im-plemented which uses the various rules of photography to assess the image aesthetics.
3) Classifier: Classifies the images into Appealing and Not Appealing.

The Research Work for the above topic can be referred on following link:
https://ieeexplore.ieee.org/document/9156003


Following Repostory Consists of:
Dataset: 3800 Images + 13 New Images(Test Input)
CNN_Train : Training of the Neural Network
CNN_ Test: Testing of the Neural Network
