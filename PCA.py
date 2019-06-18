import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

'''
PCA: to proof that using reduced dimension feature could also accomplish the classification task
'''

file = pd.read_csv('wine.data')
data = file.values
y_data = data[:, 0]
x_data = data[:, 1:]

shuffle = np.random.permutation(x_data.shape[0])
x_data = x_data[shuffle]
y_data = y_data[shuffle]

total = x_data.shape[0]
train_end = int(total * 0.8)
x_train = x_data[0:train_end, :]
y_train = y_data[0:train_end]
x_test = x_data[train_end:, :]
y_test = y_data[train_end:]    

print('Train data:', x_train.shape, 'Test data:', x_test.shape)

def sigmoid(x):
    return 1./(1 + np.exp(-x))

# Normalization
stdsc = StandardScaler()
x_train = stdsc.fit_transform(x_train)
x_test = stdsc.fit_transform(x_test)

# !. Compute covariance matrix for matrix X
cov_x = np.cov(x_train.T) 
cov_test = np.cov(x_test.T)

# 2. Do SVD to get eigenvalue & eigenvector of Covariance Matrix
u, s, v = np.linalg.svd(cov_x)
u1, s1, v1 = np.linalg.svd(cov_test)

k = 2 # select 2 principal component, which is the largest eigenvalue == largest variance after transformation
principal_component = u[:, 0:k]
pc_2 = u1[:, 0:k]

# 3. Do trainsformation
transform_x = np.matmul(x_train, principal_component)
test_reduce = np.matmul(x_test, pc_2)

class1 = []
class2 = []
class3 = []
for i in range(transform_x.shape[0]):
    if y_train[i] == 1:
        class1.append(transform_x[i])
    elif y_train[i] == 2:
        class2.append(transform_x[i])
    elif y_train[i] == 3:
        class3.append(transform_x[i])

class1 = np.array(class1)
class2 = np.array(class2)
class3 = np.array(class3)

'''Show the result: use logistic regression'''
# Use logistic regression to see if the principal component allows us to do classification by fewer dimension data
classifier = LogisticRegression()
classifier = classifier.fit(transform_x, y_train)
predict_class = classifier.predict(test_reduce)
correct_list = (predict_class == y_test).astype(int)
correct_rate = sum(correct_list)/len(correct_list) # test performance

# Plot decision boundary line
# Create meshgrid to plot the decesion boundary line
xmin = min(transform_x[:, 0])
xmax = max(transform_x[:, 0])
ymin = min(transform_x[:, 1])
ymax = max(transform_x[:, 1])
#print(len(np.arange(xmin, xmax, 0.01)))
#print(len(np.arange(ymin, ymax, 0.01)))
axis_len = len(transform_x) # number of grid points
resolution_x = (xmax - xmin)/axis_len
resolution_y = (ymax - ymin)/axis_len
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.005), np.arange(ymin, ymax, 0.005))
# plot coutour: create axis points xx, yy and input to the classifiers as principal component [x1, x2], we can show the boundary line

prob = np.c_[xx.ravel(), yy.ravel()] # output probability boundary line: [[xx1, yy1], [xx2, yy2], ... ] use axis point as model input
boundary = classifier.predict(prob).reshape(xx.shape) # reshape to locate the output on each grid point

print('Test accuracy:', correct_rate)

plt.title('Wine dataset')
plt.scatter(class1[:, 0], class1[:, 1], label = 'class1')
plt.scatter(class2[:, 0], class2[:, 1], label = 'class2')
plt.scatter(class3[:, 0], class3[:, 1], label = 'class3')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.contour(xx, yy, boundary)
plt.legend(loc = 'upper right')


plt.show()








