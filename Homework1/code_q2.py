
import numpy as np
import numpy.random
from scipy.spatial import distance
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


#Fetching and loading DATA
mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target'].astype(int)

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]
#################################################################
#Qustion 2 part A
def k_nn(images, lables, query_image, k):
    dists = distance.cdist(images, [query_image], 'euclidean').flatten()
    #print(dists)
    k_closest_indxs = np.argpartition(dists, k)[:k]
    k_closest_labels = lables[ k_closest_indxs]
    most_com_label = np.bincount(k_closest_labels).argmax()
    return most_com_label
#################################################################
#For Sections 2-3 (take n = 1000)
images_1000 = train[:1000]
labels_1000 = train_labels[:1000]
#################################################################
#Qustion 2 part B

preds = []
for test_image in test:
    preds.append(k_nn(images_1000, labels_1000, test_image, 10))
# Calculate accuracy
accuracy = accuracy_score(test_labels, preds)
print(f'Accuracy: {accuracy * 100}%')
#OUTPUT: 86.4%

# Expected accuracy from a completely random predictor will be 10% because the random predicor will predict an image to 1 of the 10 groups
#################################################################



#################################################################
#Qustion 2 part C
def k_nn_with_precomputed_distances(distances, labels, k):
    #Couldn't do this qustion with the orginal k_nn
    #the k_nn needed to be modifed to calculate in more effiacnt way to get a "normal" run time
    #this version ~recieves~ the distances precalculated that makes the code way more faster.
    #the distances are calculated by cdist funcion (scipy) which (according to the internet) faster in a bit from np.linalg.norm
    k_closest_indxs = np.argpartition(distances, k)[:k]
    k_closest_labels = labels[k_closest_indxs]
    most_com_label = np.bincount(k_closest_labels).argmax()
    return most_com_label

distances_matrix = distance.cdist(test, images_1000, 'euclidean')
acc_list = []

for k in range(1, 101):
    preds = []
    for i in range(len(test)):
        preds.append(k_nn_with_precomputed_distances(distances_matrix[i], labels_1000, k))
    accuracy = accuracy_score(test_labels, preds)
    acc_list.append(accuracy)
plt.plot(range(1, 101), acc_list)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Prediction accuracy as a function of k')
plt.grid(True)
plt.show()
#################################################################
#Qustion 2 part D

#Note: the run time in colab was 6m 40s , unfortunatly could'nt make it more efficient
accuracies = []

for n in range(100, 5001, 100):
    images_n = train[:n]
    labels_n = train_labels[:n]
    preds = []
    for test_image in test:
        preds.append(k_nn(images_n,labels_n, test_image, 1))
    accuracy = accuracy_score(test_labels, preds)
    accuracies.append(accuracy)

plt.plot(range(100, 5001, 100), accuracies)
plt.xlabel('Training set size (n)')
plt.ylabel('Accuracy')
plt.title('Prediction accuracy as a function of training set size')
plt.grid(True)
plt.show()
#################################################################