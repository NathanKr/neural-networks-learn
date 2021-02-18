from PIL import Image
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
    ax.set_title(f'image of X[{sample}] , y[{sample}][0] : {_y[sample][0]} ')
    ax.imshow(image, cmap='gray')


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
    error_indecis = []
    i_sample=0
    for x_sample , y_sample_fixed in zip(_X,_Y):
        h = net.feedforward(x_sample)
        i_max = np.argmax(h) # index of max probability
        if y_sample_fixed[i_max] == 1:
           count_correct += 1 
        else:
           error_indecis.append(i_sample)                       

        i_sample += 1

    return  (100*count_correct/len(_Y) , error_indecis)

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
    (correct_test_percentage , error_test_indices) = \
        compute_success_percentage(net,_X[-test_samples:,:],_Y[-test_samples:,:])
    (correct_training_percentage , error_training_indices) = \
        compute_success_percentage(net,_X[:traning_samples,:],_Y[:traning_samples,:])
    return ((correct_test_percentage,error_test_indices) , \
            (correct_training_percentage,error_training_indices))
    
def learning_curves_engine(samples_vec):
    correct_trainings = []
    correct_tests = []
    
    for samples in samples_vec:
        ((correct_test_percentage ,_),(correct_training_percentage, _)) = \
            learn_nn(X[:samples,:],Y[:samples,:])
        correct_trainings.append(100 - correct_training_percentage)
        correct_tests.append(100 - correct_test_percentage)

    return (correct_trainings , correct_tests)


def learning_curves():
    make_results_random() # it is a must 

    loops_for_mean = 5
    
    samples_vec = [50 , 75, 100 , 200 , 500, 1000, 2000,5000]
    np_correct_trainings = np.array([])
    np_correct_tests = np.array([])

    _ , (ax1, ax2 , ax3) = plt.subplots(3)

    for i in range(loops_for_mean):
        print(f"\n********* loop : {i+1} ***************\n")
        correct_trainings , correct_tests = learning_curves_engine(samples_vec)
        np_correct_trainings = np.append(np_correct_trainings,correct_trainings)
        np_correct_tests = np.append(np_correct_tests,correct_tests)
        ax1.plot(samples_vec,correct_tests)
        ax1.set_title("test error [%]")
        ax2.plot(samples_vec,correct_trainings)
        ax2.set_title("traing error [%]")


    np_correct_trainings = np_correct_trainings.reshape((loops_for_mean,len(samples_vec)))
    np_correct_tests = np_correct_tests.reshape((loops_for_mean,len(samples_vec)))
    ax3.plot(samples_vec,np_correct_trainings.mean(axis=0),'x')
    ax3.plot(samples_vec,np_correct_tests.mean(axis=0),'o')
    ax3.set_title("mean error [%] . training - x , test - o")
    plt.tight_layout()
    plt.show()

    make_results_reproducible() # outside of this function i want reproducible



def learn(show_error_images=False):
    _ , (ax1,ax2) = plt.subplots(2,1)
    ((correct_test_percentage,error_test_indices) , \
            (correct_training_percentage,error_training_indices)) = learn_nn(X,Y)
    print(f"percentage of correct estimations test : {correct_test_percentage}")
    print(f"percentage of correct estimations training : {correct_training_percentage}")
    if show_error_images:
        plot_images(ax1 , error_training_indices,X,y)
        ax1.set_title("training error images")
        # ax1xaxis.set_visible(False)
        plot_images(ax2 , error_test_indices,X,y)
        ax2.set_title("test error images")
        # ax2.xaxis.set_visible(False)
        plt.show()


def plot_images(ax , samples,_X,_y):
    images_in_row = 10
    images = []
    sample = 0

    while sample < len(samples):
        images_row = []
        for _ in range(images_in_row):
            images_row.append(_X[sample].reshape(20,20))
            sample += 1
            if sample < len(samples):
                break
        
        merged_images_horizontal = np.concatenate(images_row,axis=1)   
        images.append(merged_images_horizontal)

    merged_images = np.concatenate(images,axis=0)
    ax.imshow(merged_images, cmap='gray')


_ , ax1 = plt.subplots(1)
plot_images(ax1,[0],X,y)
plt.show()

# plots(X,Y)    
# learn(True)
# learning_curves()

