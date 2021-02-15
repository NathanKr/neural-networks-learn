import os
from os.path import join 
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from nn_learn_back_propagation_engine import Network
from utils import sigmoid , dsigmoid_to_dval

current_dir = os.path.abspath(".")
data_dir = join(current_dir, 'data')
file_name = join(data_dir,"ex3data1.mat")
mat_dict = sio.loadmat(file_name)
# print("mat_dict.keys() : ",mat_dict.keys())

X = mat_dict["X"]
# print(f"X.shape : {X.shape}")
y = mat_dict["y"]
m = y.size
# print(f"y.shape : {y.shape}")
Y = np.zeros((m,10))

# fix Y for logistic regression
for row,y_sample in enumerate(y):
    if y_sample == 10:
        # digit 0 is marked as 10 in y
        Y[row,0]=1 
    else:
        # digit 1-9 are marked as is y
        Y[row,y_sample]=1                 




def plot_image(ax , sample):
    image = X[sample].reshape(20,20)
    rotated_img = ndimage.rotate(image,0)
    ax.set_title(f'image of X[{sample}] , y[{sample}][0] : {y[sample][0]} ')
    ax.imshow(rotated_img, cmap='gray')


def plots():
    _ , axs = plt.subplots(2,2)
    # pick a sample to plot
    plot_image(axs[0,1],4300)

    sample = 10
    plot_image(axs[0,0],sample)

    axs[1,0].set_title(f'X[{sample}]')
    axs[1,0].grid()
    axs[1,0].plot(X[sample],'o')

    axs[1,1].set_title('y')
    axs[1,1].plot(y,'o')

    plt.show()

def learn_nn():
    net = Network([400, 40 , 10],sigmoid , dsigmoid_to_dval)
    net.print_shapes()
    mini_batch = [(x_sample.reshape(x_sample.size,1),y_sample.reshape(y_sample.size,1)) for x_sample , y_sample in zip(X,Y)]
    epochs = 40
    net.train(mini_batch,epochs,0.01)
    print('*********************************************')
    net.print_shapes()
    count_correct=0
    for x_sample , y_sample_fixed in zip(X,Y):
        h = net.feedforward(x_sample)
        i_max = np.argmax(h) # index of max probability
        if y_sample_fixed[i_max] == 1:
           count_correct += 1 

    print(f"percentage of correct estimations : {100*count_correct/m}")


# plots()    
learn_nn()

