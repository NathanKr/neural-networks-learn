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
