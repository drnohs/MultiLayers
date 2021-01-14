"""
2021.1.14
Compare Time for Two layers Net & Multi Layers Net
Backpropagation, SGD
"""
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.two_layer_net import TwoLayerNet
from common.multi_layer_net import MultiLayerNet
import time

# Dataset
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


iters_num = 10000

train_size = x_train.shape[0] #60,000
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

train2_loss_list = []
train2_acc_list = []
test2_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1) #600

# TWO LAYERS
# 784 - 100 - 10

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

print("Two Layers:784-100-10")
print("Training Data set:",train_size)
print("Batch size:",batch_size)
print("Training Data set:",train_size)

sTime=time.time()

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # Learning Process
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("#{} : {} epoch ".format(i,i*batch_size/train_size), end="| ")
        print("train acc : {:.4f} , test acc : {:.4f}".format(train_acc,test_acc))

eTime=time.time()
print("{:.2f}".format((eTime-sTime)*1000000),end=" ")
print("seconds elapsed")
print("")

# MULTI LAYERS
# 784 - 100 - 100 - 100 - 100 - 10

network2 = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],output_size=10)

print("Multi Layers : 784-100-100-100-100-10")
print("Training Data set:",train_size)
print("Batch size:",batch_size)
print("Training Data set:",train_size)

sTime=time.time()

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network2.gradient(x_batch, t_batch)
    for key in network2.params:
        network2.params[key] -= learning_rate * grad[key]

    loss = network2.loss(x_batch, t_batch)
    train2_loss_list.append(loss)

    if i % iter_per_epoch == 0:


        train2_acc = network2.accuracy(x_train, t_train)
        test2_acc = network2.accuracy(x_test, t_test)
        train2_acc_list.append(train2_acc)
        test2_acc_list.append(test2_acc)
        print("#{} : {} epoch ".format(i, i * batch_size / train_size), end="| ")
        print("train acc : {:.4f} , test acc : {:.4f}".format(train2_acc, test2_acc))

eTime=time.time()
print("{:.2f}".format((eTime-sTime)*1000000),end=" ")
print("seconds elapsed")

# Graph
#markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, test_acc_list, label='Two Layers Acc')
#plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.plot(x, test2_acc_list, label='Multi Layer Acc')
plt.xlabel("epochs")
plt.ylabel("accuracy")
#plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()