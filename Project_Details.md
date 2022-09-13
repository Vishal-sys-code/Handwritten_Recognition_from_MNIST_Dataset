# <b>HANDWRITTEN DIGIT RECOGNITION</b>

This is a project to learn handwritten digit recognition on the MNIST dataset.<br>

Basically, we will perform classification on various handwritten texts and judge whether they are valid digits or not using the MNIST dataset. <br>

**What is MNIST?** <br>

* MNIST stands for <b><i>Modified National Institute of Standards and Technology</i></b> dataset.

* It is a set of 70,000 small images of digits handwritten by high school students and employees of the US causes Bureau.

* All images are labeled with the respective digit they represent.

* MNIST is the hello world of the machine learning. Everytime a ML engineer makes a new algorithm for classification, they would always first check it's performance on the MNIST dataset.

* There are 70k images and each images has 28*28 = 784 features.

* Each image is 28*28 pixels and each features simply represents one-pixel intensity from 0 to 255. If the intensity is 0, it means that the pixel is white and if it is 255, it means it is black.

<b><i>Representation of the handwritten digits in the MNIST dataset</i></b>

<a href=""><img src="https://drive.google.com/uc?export=view&id=1nqkp7jTAuBsHGt-WFbzf0dnGwX6Bpdnu" 
style="width: 450px; max-width: 100%; height: auto" title="Click to enlarge picture"></a>

<br>

We will use **Python interactive notebook (ipynb)** to make this ML model. <br>

<hr>

## **INSTALLATIONS**

```sh
pip install numpy
```
```sh
pip install opencv-python
```
```sh
pip install matplotlib
```
```sh
pip install tensorflow
```
<hr>

## **IMPORT STATEMENTS REQUIREMENTS**

```python
import os

# import os means that it is a module to interact with the underlying operating system.
```

```python
import cv2

# cv2 is the module under the OpenCV(Open Computer Vision) it is used all sort of image and video analysis like: facial recognition, liscence plate reading, Optical character recognition etc..
```

```python
import numpy as np

#NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices.
```

```python
import tensorflow as tf

#TensorFlow is an open source framework developed by Google researchers to run machine learning, deep learning and other statistical and predictive analytics workloads.
```

```python
import matplotlib.pyplot as plt

#Pyplot is an API (Application Programming Interface) for Python's matplotlib that effectively makes matplotlib a viable open source alternative to MATLAB. Matplotlib is a library for data visualization, typically in the form of plots, graphs and charts.
```
<hr>