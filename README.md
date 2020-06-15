# Facial-Key-Point-Detection

## Project Overview

In this project, you’ll combine your knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. Your completed code should be able to look at any image, detect faces, and predict the locations of facial keypoints on each face; examples of these keypoints are displayed below

![GitHub Logo](/images/keypoint_eg1.png)
Format: ![Alt Text](url)
![GitHub Logo](/images/keypoint_eg2.png)
Format: ![Alt Text](url)

The project will be broken up into a few main parts in three Python notebooks and a file : 

Notebook 1 : Loading and Visualizing the Facial Keypoint Data

Notebook 2 : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

Notebook 3 : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

File : model.py

## Requirements
Please see the requirements.txt file. To ensure you're up to date, run:

pip install -r requirements.txt

## Getting the Dataset 
Data Set used is [Youtube Face Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/)
It is a dataset that contains 3,425 face videos designed for studying the problem of unconstrained face recognition in videos. These videos have been fed through processing steps and turned into sets of image frames containing one face and the associated keypoints.
