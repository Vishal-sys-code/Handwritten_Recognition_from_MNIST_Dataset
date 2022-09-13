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

## **First Procedure**

*  First of all, we need to import all the libraries required ans also we will use **```fetch_openml```** from the **```sklearn.datasets library```**.

* Create a variable mnist, and store in it the ***mnsit_784*** dataset from the ***featch_openml*** And you can further print and see the contents of this ***mnist*** dataset. You can see its keys, its data, its corresponding labels, and more.

* Code snippet

```python
# fetching dataset
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

mnist = fetch_openml('mnist_784')
```

<hr>

## **Second Procedure**

* Create array variables ```x``` and ```y```. Store in them the data and the targets respectively of the mnist As we discussed above, we have ```784 (28x28) pixels``` of features, and these are now stored in ```x```. Variable ```y``` has the corresponding digit that the picture resembled by ```x``` contains.

* You can try to see that picture ```x``` using ***matplotlib*** Since the pixels here are stacked together in a 1D array ```x``` for memory issues, you’ll have to reshape it back to ```26x26```. Create a variable ```some_digit``` and fill it with any random digit array from the dataset. Reshape it and store it in another variable ```some_digit_image```.

* You can now simply use the command below to get that image plotted. We have used the ```imshow``` attribute of **pyplot** and have fed the ```some_digit_image``` 2D array of pixel data into it.

* Code Snippet

```python
x, y = mnist['data'], mnist['target']

some_digit = x.to_numpy()[36001]
some_digit_image = some_digit.reshape(28, 28)  # let's reshape to plot it

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,interpolation='nearest')
plt.axis("off")
plt.show()
```
<hr>

## **Third Procedure**

* And since the data, we stored corresponded to the 36000th element, the image we were shown was:

<a href=""><img src="https://drive.google.com/uc?export=view&id=1m0Iq0rzqrHSZjnhsm7sfwfJVI9pnD1c6" 
style="width: 450px; max-width: 100%; height: auto" title="Click to enlarge picture"></a>

* And if you just change the selected number from 36000 to 36001, the image we get is:

<a href=""><img src="https://drive.google.com/uc?export=view&id=1H8PS4BedhRVQoYtoPs1s4mt4pcAsgLEH" 
style="width: 450px; max-width: 100%; height: auto" title="Click to enlarge picture"></a>

* You can turn the axes off, using the axis property of pyplot. And now we can see what digits these images resemble. You must have already observed that by now, the first image was a 9, and the second image was a 2. Verify it by printing the labels stored in y.

* And for our convenience, MNIST dataset is already split into training and testing data. The first 60000 are training data, and the last 10000 are testing data. Create two array variables ```x_train``` and ```x_test``` and store in them, the training and the testing data respectively. Similarly, create another two array variables ```y_train``` and ```y_test``` and store in them, the training and the testing labels/targets respectively. And once you split your data, you must consider shuffling the dataset.

* Now, since there are 10 different labels from 0 to 9, and we want to do binary classification, we would replace our target from 0 to 9 to true and false, if the target is a 2 or not. Create two array variables ```y_train_2``` and ```y_test_2``` and store in them the true false value of ```y_train``` and ```y_test``` if the label is a 2 or not.

* Code Snippet

```python
x_train, x_test = x[:60000], x[6000:70000]
y_train, y_test = y[:60000], y[6000:70000]

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train.[shuffle_index], y_train.[shuffle_index]

# Creating a 2-detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == '2')
y_test_2 = (y_test == '2')
```

<hr>

## **Final Procedure(Train the Model)**

* Now, import another package from **sklearn**.```linear_model``` called Logistic Regression and load it into a classifier variable ```clf```. This creates our classifier. Use the fit attribute of the classifier model to feed our features and labels into it and the predict attribute to predict the labels based on some features. And after our classifier gets trained, we’ll test if it predicts true for the image 36001, we showed to you above.

* Code Snippet

```python
# Train a logistic regression classifier
clf = LogisticRegression(tol=0.1)
clf.fit(x_train, y_train_2)
example = clf.predict([some_digit])
print(example)
```

* And the output we received was [True].

* Hence the feature falls into the category of digit 2. So, this is how we build and train a logistic regression on ```MNIST``` dataset, and we did show you how we predict the labels using it.

<hr>

## **Cross Validating**

* As we studied earlier, cross-validation increases the efficiency of the model. We would ```cross-validate``` our model, and retrieve the accuracy of the model. Don’t forget to import ```cross_val_score``` from **sklearn.model_selection**. Create a variable a and store in it the accuracy as measured by our cross validator, when we passed our model, and the training data. Check its mean for the overall accuracy of the model.

```python
# Cross Validation
a = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")
print(a.mean())
```
<hr>

## **Result**

Our output was 0.9781333333333334 which is quite high. Having an accuracy of 97.8% is indeed a great deal.  But I would like to give you all your homework before we end this lecture. Create a classifier that will classify a digit always as not 2. Now, there is a spoiler. Since more than 90% of the digits here are not 2, and even if you classify all the digits as not 2, you would get an accuracy of 90% at least. This was a catch. Is accuracy always a great metric to define a good classifier? NO.

<hr>