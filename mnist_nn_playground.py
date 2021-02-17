import os
from os.path import join 
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from Network import Network
from utils import sigmoid , dsigmoid_to_dval , make_results_reproducible , make_results_random

make_results_reproducible()


current_dir = os.path.abspath(".")
data_dir = join(current_dir, 'data')
file_name = join(data_dir,"ex3data1.mat")
mat_dict = sio.loadmat(file_name)
# print("mat_dict.keys() : ",mat_dict.keys())

X = mat_dict["X"]
# print(f"X.shape : {X.shape}")
y = mat_dict["y"]

# make order random so test is ok because mnist is arrange
# such that each 500 samples are the same
indices = np.arange(len(y))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]


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




def plot_image(ax , sample,_X,_y):
    image = _X[sample].reshape(20,20)
    rotated_img = ndimage.rotate(image,0)
    ax.set_title(f'image of X[{sample}] , y[{sample}][0] : {_y[sample][0]} ')
    ax.imshow(rotated_img, cmap='gray')


def plots(_X,_y):
    _ , axs = plt.subplots(2,2)
    # pick a sample to plot
    plot_image(axs[0,1],4300,_X,_y)

    sample = 10
    plot_image(axs[0,0],sample,_X,_y)

    axs[1,0].set_title(f'X[{sample}]')
    axs[1,0].grid()
    axs[1,0].plot(_X[sample],'o')

    axs[1,1].set_title('y')
    axs[1,1].plot(_y,'o')

    plt.show()

def compute_success_percentage(net,_X,_Y):
    count_correct=0
    for x_sample , y_sample_fixed in zip(_X,_Y):
        h = net.feedforward(x_sample)
        i_max = np.argmax(h) # index of max probability
        if y_sample_fixed[i_max] == 1:
           count_correct += 1 

    return  100*count_correct/len(_Y)

def learn_nn(_X,_Y):
    net = Network([400, 30 , 10],sigmoid , dsigmoid_to_dval)
    epochs = 20
    test_samples_percentage = 20
    test_samples = int(m * (test_samples_percentage / 100))
    traning_samples = m - test_samples
    training_data = [(x_sample.reshape(x_sample.size,1),y_sample.reshape(y_sample.size,1)) for x_sample , y_sample in zip(_X[:traning_samples,:],_Y[:traning_samples,:])]
    mini_batch_size = 1
    learning_rate = 1 
    net.SGD(training_data, epochs, mini_batch_size, learning_rate)
    correct_test_percentage = compute_success_percentage(net,_X[-test_samples:,:],_Y[-test_samples:,:])
    correct_training_percentage = compute_success_percentage(net,_X[:traning_samples,:],_Y[:traning_samples,:])
    return (correct_test_percentage , correct_training_percentage)
    
def learning_curves_engine(samples_vec):
    # correct_trainings = []
    # correct_tests = []
    
    # for samples in samples_vec:
    #     correct_test_percentage , correct_training_percentage = learn_nn(X[:samples,:],Y[:samples,:])
    #     correct_trainings.append(100 - correct_training_percentage)
    #     correct_tests.append(100 - correct_test_percentage)


    correct_trainings = range(0,len(samples_vec),1)
    correct_tests = range(len(samples_vec),0,-1)
    return (correct_trainings , correct_tests)


    

def learning_curves():
    make_results_random() # it is a must 

    results = []
    loops = 10
    start=500
    stop=m
    step = 500
    samples_vec = range(start,stop,step)
    np_correct_trainings = np.array([])
    np_correct_tests = np.array([])

    _ , (ax1, ax2) = plt.subplots(2)

    for _ in range(loops):
        correct_trainings , correct_tests = learning_curves_engine(samples_vec)
        results.append((correct_trainings,correct_tests))
        ax1.plot(samples_vec,correct_tests)
        # plt.grid()
        ax2.plot(samples_vec,correct_trainings)

    # c_training , c_tests = results[0]
    # plt.title("error percentage . training - x , test - o")
    # plt.xlabel("dataset size")
    # ax1.plot(samples_vec,c_tests,'o')
    # plt.grid()
    # ax2.plot(samples_vec,c_training,'x')
    plt.show()

    make_results_reproducible() # outside of this function i want reproducible



def learn():
    correct_test_percentage , correct_training_percentage = learn_nn(X,Y)
    print(f"percentage of correct estimations test : {correct_test_percentage}")
    print(f"percentage of correct estimations training : {correct_training_percentage}")


# plots(X,Y)    
# learn()
learning_curves()

