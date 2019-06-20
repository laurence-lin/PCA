# PCA
Apply PCA on wine dataset, to show that by the dimension reduction principal component, the extracted feature is available to perform classification task.

Set k = 2, use SVD to perform PCA, reduce the original 13 features dataset into 2 features.
After PCA, the dataset is clear to show on plot and still able to perform classification.

Use 80% for train set, 20% for test set

Testing accuracy: 0.805
The test accuracy could reach 1.0 by random, but sometimes get bad performance like 0.1. 
Use the LogisticRegression package in sklearn is not stable.

![img](https://github.com/laurence-lin/PCA/blob/master/wine_class.png)


Discussion:

PCA could help dimension reduction, extract features, and help visualization

Visualization: By projecting the multi-features into 2 feature, we could show the data on a 2D plot, and plot the data with different class colors.
