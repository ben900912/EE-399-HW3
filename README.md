# EE-399-HW3
``Author: Ben Li``
``Date: 4/20/2023``
``Course: SP 2023 EE399``
![nagesh-pca-1](https://user-images.githubusercontent.com/121909443/233534607-c52ccbeb-6446-4d01-a549-4d270fe54301.gif)

## Abstract
The assignment involves an analysis of MNIST dataset, which is a public database of handwritten digits. The dataset is pre-processed and split into training and testing sets. We need to perform some analysis which includes SVD of the digit images, determining the number of modes necessary for good image reconstruction, and interpreting the the U, Σ, and V matrices.

Once the data is projected into PCA space, classifiers are built to identify individual digits in the training set. Linear discriminant analysis (LDA) is used to classify pairs of digits and triples of digits. The accuracy of separation is calculated for the hardest and easiest pairs of digits to separate.

<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/121909443/233534705-421f3854-0e79-4f9b-a455-4fe353f7b3c2.png"> 
    <em>Figure 1: The figure illustrates a 3-D feature space is split into two 1-D feature spaces, and later, if found to be correlated, the number of features can be reduced even further.</em>
</p>

The performance of LDA, support vector machines (SVM), and decision trees is compared on the hardest and easiest pairs of digits to separate. Finally, the classification performance of all three methods is evaluated on the training and test sets.

## Introduction and Overview
The ``MNIST`` dataset consists of 70000 images of digits, with each image being a 28x28 pixel grayscale image. The MNIST dataset has been widely used as a benchmark dataset for machine learning algorithms.

In this analysis, we will perform an analysis of the MNIST dataset. We will start by performing an SVD analysis of the digit images, and then we will build a classifier to identify individual digits in the training set. We will use linear classifiers such as LDA, SVM, and decision trees to classify the digits.

The analysis consist of four different parts
1. SVD analysis of the MNIST dataset: We will reshape each image into a column vector and perform an SVD analysis to determine the rank of the digit space.

2  Building a classifier to identify individual digits in the training set: We will use linear classifiers such as LDA, SVM, and decision trees to classify the digits. We will start by picking two digits and building a linear classifier to identify them, then we will pick three digits and build a linear classifier to identify them.

3. Determining the easiest and hardest digits to separate: We will use LDA to determine the easiest and hardest digits to separate.

4. Comparing the performance of LDA, SVM, and decision trees: We will compare the performance of these classifiers on the easiest and hardest digits to separate.

## Theoretical Background
There are several important concept that is being used in this assignment. 
First, the MNIST database is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets.

We also uses Principal Component Analysis (PCA) which is commonly used techinque in order to reduce the dimension in data analysis.  It works by finding the linear combinations of the original features that capture the most variation in the data. The resulting principal components are orthogonal and sorted in decreasing order of importance

Next, Linear Discriminant Analysis (LDA) is also used for dimensionality reduction. It works by finding the linear combinations of the original features that maximize the seperation between classes. 

Support Vector Machines (SVMs) are a class of algorithms that are commonly used for classification problems. SVMs work by finding the hyperplane that maximally separates the classes in the feature space. Decision trees are another commonly used algorithm for classification problems. Decision trees work by recursively splitting the feature space into smaller regions based on the values of the features.

## Algorithm Implementation and Development 
Here are some analysis that we performs and my implementation of that
> Do an SVD analysis of the digit images. You will need to reshape each image into a column vector

We need to laod the MNIST dataset and reshape each images. In this case, X should be a matrix with dimensions 784 x 70000, where each column corresponds to a reshaped image of a handwritten digit in grayscale. You can create this matrix using the following code:

```python
import numpy as np
from sklearn.datasets import fetch_openml

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist.data.T, mnist.target.astype(int)

# Rescale the pixel values to be between 0 and 1
X /= 255.0

# Transpose X so that each column corresponds to a digit image
X = X.T
```

Then, we need reshape each image into a column vector to create the data matrix. Here, we can also perform SVD on the data matrix 
```python
import numpy as np
from keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape each image into a column vector
X_train = x_train.reshape((-1, 28*28)).T
X_test = x_test.reshape((-1, 28*28)).T

# Perform SVD on the data matrix
U, s, Vt = np.linalg.svd(X_train, full_matrices=False)
```

> On a 3D plot, project onto three selected V-modes (columns) colored by their digit label. For
example, columns 2,3, and 5.

Over here, we plot the singular values on log scale to visualize their decay. 
```python
import matplotlib.pyplot as plt

# Plot singular value spectrum
plt.semilogy(s)
plt.xlabel('Singular value index')
plt.ylabel('Singular value')
plt.show()
```

The next task is to build a classifier for digit identification, we first need to split our data into training and test sets. We will use the first 60,000 images for training and the remaining 10,000 for testing. We will also use PCA to reduce the dimensionality of the data before applying the classifiers.



**1. Two Digit LDA Classifier**

In order to build a classifier to identify 0 and 1. We use the first two principal components of the PCA transformed data and apply LDA to classify the images.
```python
# Use PCA to reduce dimensionality of data
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Build LDA classifier
lda = LDA()
lda.fit(X_train_pca, y_train)
```

**2. Three digit LDA classifier**

Next we build a classifier to identify the digits 0, 1, and 2. We will use the first three principal components of the PCA transformed data and apply LDA to classify the images.

```python
# Use PCA to reduce dimensionality of data
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Build LDA classifier
lda = LDA()
lda.fit(X_train_pca, y_train)
```

**3. Most difficult Pair of Digit**

To determine which two digits are most difficult to separate, we can use LDA to classify each pair of digits and compare the accuracy. We find that the most difficult pair to separate is 4 and 9.

```python
# Use PCA to reduce dimensionality of data
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Build LDA classifier for digit pair 4 and 9
X_train_49 = X_train_pca[(y_train==4) | (y_train==9)]
y_train_49 = y_train[(y_train==4) | (y_train==9)]
X_test_49 = X_test_pca[(y_test==4) | (y_test==9)]
y_test_49 = y_test[(y_test==4) | (y_test==9)]

lda_49 = LDA()
lda_49.fit(X_train_49, y_train_49)
y_pred_49 = lda_49.predict(X
```

## Computational Results
> What does the singular value spectrum look like and how many modes are necessary for good
image reconstruction? (i.e. what is the rank r of the digit space?)

![image](https://user-images.githubusercontent.com/121909443/233538737-b87495d5-71c7-4cd6-a53f-5bbcb4f7abaf.png)

From the plot, we observe that the singular values decay rapidly, with the first few values dominating the spectrum. To determine the rank r of the digit space, we need to select the number of modes necessary for good image reconstruction. We can use the elbow method to determine the number of modes, which corresponds to the point where the slope of the curve changes significantly.

From the plot, we observe that the slope changes significantly around index 30, indicating that we can choose r=30 modes for good image reconstruction.

> What is the interpretation of the U, Σ, and V matrices?
The SVD decomposes the data matrix X_train into three matrices U, Σ, and V such that $$X_train=UΣV^T$$. The matrix U contains the left singular vectors, which represent the principal components of the data. The matrix V contains the right singular vectors, which represent the basis functions that reconstruct the data. The matrix Σ contains the singular values, which represent the importance of each component.

The following is the 3D plot of the first three V modes. We color the points by their bigit label to visualize the seperation of the digits in PCA space. 

![image](https://user-images.githubusercontent.com/121909443/233762885-76d93d3d-7778-44e6-a9aa-606a95675e92.png)

In building our own classifier to identify individual digits in the training set, we found that 4 and 9 are the most difficult to seperate. It has an accuracy of around 53%


![image](https://user-images.githubusercontent.com/121909443/233763016-3e99cdd5-a869-4bdf-badd-5437047b2670.png)

We found that 0 and 1 is the easiest to seperate. It has a accuaracy of 99%


![image](https://user-images.githubusercontent.com/121909443/233763035-6748ecba-6be5-4b7f-9b26-b7d232177226.png)


**Comparing SVM and decision tree**
For the MNIST dataset, both SVM and decision tree done a good job. However, the performance of these classifier pay depends on the specific pair of digits being classified. For example, like what we did before, some pairs of numbers are more difficult to pair than others. Noted that currently, convolutional neural networks (CNN) have since supassed both models in terms of the accuracy on the MNIST dataset. Despite that SVM and decision tree is a really good baseline for comparison and may still be suitable for certain applications. 

**Comparing performance between LDA SVM and decision tree**
In general, all three classifiers performed well on the training set with accuracy above 99%. However, the accuracy dropped when applied to the test set, with LDA being the least affected, followed by SVM and then decision tree. This may indicate overfitting on the training set.

Overall, it seems that LDA and SVM classifiers performed similarly in most cases, while the decision tree classifier performed slightly worse. However, the choice of classifier may depend on the specific application and the trade-offs between accuracy, interpretability, and computational complexity.


## Summary and Conclusions
In this assignment we have explored and analyze the MNIST dataset. We used PCA to reduce the dimensionality of the data and also built classifiers using LDA, SVM, and decision trees to identify individual digits in the training dataset. 

In conclusion, LDA is best to reasonably identify and classify two and three digits while SVM and decision trees preformed similarly to LDA in seperating all ten digits. Also, the digits that were most difficult to seperate are 4 and 9. The easiest to seperate is 0 and 1. LDA performed the worst on the difficult pair, with an accuracy of only 54.9% on the test set, while SVM and decision trees performed better with accuracies of 80.6% and 84.1%, respectively. On the easy pair, LDA performed the best with an accuracy of 99.1% on the test set, while SVM and decision trees also performed well with accuracies of 98.9% and 97.8%, respectively.

All in all, we can see that PCA is good at reducing dimensionality and the performance of different classifiers. Our findings suggest that different classifiers may perform differently depending on the pair of the digit being seperated, and it is important to evaluate their performance on both training and test sets to assess their generalization ability. 
