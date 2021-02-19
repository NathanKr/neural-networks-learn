import numpy as np
import random


def sigmoid(val):
    return 1/(1+np.exp(-val))

def dsigmoid_to_dval(val):   
    sig = sigmoid(val)
    return sig * (1 - sig)        


def make_results_reproducible():
    random.seed(12345678)
    np.random.seed(12345678)

def make_results_random():
    random.seed()
    np.random.seed()

def plot_images(ax ,images_in_row,image_height, image_width,samples,_X,_y):
    """
        function is working ONLY if len(samples) % images_in_row is zero
    
    """
    images = []
    sample = 0

    while sample < len(samples):
        images_row = []
        for _ in range(images_in_row):
            images_row.append(_X[sample].reshape(image_height,image_width))
            sample += 1
            if sample == len(samples):
                break
        
        merged_images_horizontal = np.concatenate(images_row,axis=1)  # append horizontaly
        images.append(merged_images_horizontal)

    merged_row_images_vertically = np.concatenate(images,axis=0) # append vertically
    ax.imshow(merged_row_images_vertically, cmap='gray')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)