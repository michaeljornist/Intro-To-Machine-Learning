import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *

# Loading Data
np.random.seed(0)  # For reproducibility






#Solution B:
def sol_b():

    n_train = 10000
    n_test = 5000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)


    # Training configuration
    epochs = 30
    batch_size = 10
    # Network configuration
    layer_dims = [784, 40, 10] #1 hidden layer of size 40.

    train_loss = {}
    train_accuracy ={}
    test_accuracy = {}
    for learning_Rate in [0.001,0.01,0.1,1,10]:
        network = Network(layer_dims)
        parms, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc= network.train(x_train, y_train, epochs, batch_size, learning_Rate, x_test=x_test, y_test=y_test)
        
        train_loss[learning_Rate] = epoch_train_cost
        train_accuracy[learning_Rate] = epoch_train_acc
        test_accuracy[learning_Rate] = epoch_test_acc

    for learning_Rate in [0.001,0.01,0.1,1,10]:
        plt.plot(train_loss[learning_Rate] , label=f'Rate={learning_Rate}')
    plt.xlabel('Epochs')
    plt.ylabel('training loss')
    plt.legend()
    plt.show()

    for learning_Rate in [0.001,0.01,0.1,1,10]:
        plt.plot(train_accuracy[learning_Rate] , label=f'Rate={learning_Rate}')
    plt.xlabel('Epochs')
    plt.ylabel('train accuracy')
    plt.legend()
    plt.show()

    for learning_Rate in [0.001,0.01,0.1,1,10]:
        plt.plot(test_accuracy[learning_Rate] , label=f'Rate={learning_Rate}')
    plt.xlabel('Epochs')
    plt.ylabel('test accuracy')
    plt.legend()
    plt.show()

def sol_c():
    n_train = 50000
    n_test = 10000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

    # Training configuration
    epochs = 30
    batch_size = 10
    learning_rate = 0.1

    # Network configuration
    layer_dims = [784, 40, 10]

    network = Network(layer_dims)
    parms, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = network.train(x_train, y_train, epochs, batch_size, learning_rate, x_test, y_test)
    print(f'Final Test Accuracy: {epoch_test_acc[-1]}')

def sol_d():
    n_train = 50000
    n_test = 10000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

    epochs = 30
    batch_size = 10
    learning_rate = 0.1

    layer_dims = [784, 10]
    network = Network(layer_dims)

    parms, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc= network.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

    plt.figure(figsize=(12, 6))
    plt.plot(range(epochs), epoch_train_acc, label='Training Accuracy')
    plt.plot(range(epochs), epoch_test_acc, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Test Accuracy vs Epochs')
    plt.show()

    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        weight_image = parms['W1'][i].reshape(28, 28)
        axes[i].imshow(weight_image, cmap='viridis', interpolation='nearest')
        axes[i].set_title(f'Class {i}')
        axes[i].axis('off')
    plt.show()



def sol_e():
    n_train = 50000
    n_test = 10000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

    # Training configuration
    epochs = 30
    batch_size = 300
    learning_rate = 0.5

    # Network configuration
    layer_dims = [784, 40, 10]

    network = Network(layer_dims)
    test_acc = network.train(x_train, y_train, epochs, batch_size, learning_rate, x_test, y_test)[4]
    print(f'Final Test Accuracy: {test_acc[-1]}')


if __name__ == "__main__":
    sol_b()
    #sol_c()
    #sol_d()
    #sol_e()
