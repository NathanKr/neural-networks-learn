import numpy as np

# this code is based on https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py

class Network:
    def __init__(self,sizes,activation_function,dactivation_function_to_dval):
        self.num_layers = len(sizes)
        # e.g. [3, 4, 2] i.e. 3 nurons in first(input) layer, 4 in second and 2 on last (output) layer
        self.sizes = sizes 
        # we will get for [3,4,2] a biases of [4x1 , 2x1]
        self.biases = [np.random.randn(row, 1) for row in sizes[1:]]
        # we will get for [3,4,2] a weights of [4x3 , 2x4] (because we have transpose on BP2)
        self.weights = [np.random.randn(row, col)
                        for col, row in zip(sizes[:-1], sizes[1:])]
        self.activation_function = activation_function
        self.dactivation_function_to_dval = dactivation_function_to_dval


    def backprop(self, x, y):
        """[summary]
        "Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights

        Args:
            x ([number]): input sample
            y ([number]): actual output sample
        """

        # ******* initialization
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # same size as biases
        nabla_w = [np.zeros(w.shape) for w in self.weights] # same size as weights

        # ******* feeforward pass
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)

        #  *******  backward pass
        # this first delta is the delta on the last layer L 
        # this is BP1 http://neuralnetworksanddeeplearning.com/chap2.html
        delta = self.cost_derivative(activations[-1], y) * \
            self.dactivation_function_to_dval(zs[-1])
        nabla_b[-1] = delta # BP3
        nabla_w[-1] = np.dot(delta, activations[-2].T) # BP4 , replaced transpose() with .T

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.dactivation_function_to_dval(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # BP2 , replaced transpose() with .T
            nabla_b[-l] = delta # BP3
            nabla_w[-l] = np.dot(delta, activations[-l-1].T) # BP4 , replaced transpose() with .T

        return (nabla_b, nabla_w)

   
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives of C_x with respect to
           the output activations.
           we assume here cost (1/2)*(output_activations-y)^2 

        Args:
            output_activations (number): this is actually the hypothesis h of this sample
            y ([type]): actual output sample

        Returns:
            [type]: [description]
        """
        return (output_activations-y)

    def update_mini_batch(self, mini_batch, alfa):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "alfa"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(alfa/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(alfa/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]


    def train(self,mini_batch,epochs,alfa):
        for _ in range(epochs):
            self.update_mini_batch( mini_batch,  alfa)


    def feedforward(self, x): 
        """[summary]

        Args:
            x ([number]): input layer sample

        Returns:
            [number]: output of the network - this is actualy h sample
        """

        if x.ndim > 1:
            a = x # set the first layer
        else:
            # dim is 1
            a = x.reshape(x.size,1)

        for b, w in zip(self.biases, self.weights):
            a = self.activation_function(np.dot(w, a)+b)
        
        h = a
        return h # hypothesis ,output

    def print_list(self,l,var_name):
        i=0
        while i < len(l):
            print(f"{var_name}[{i}].shape : {l[i].shape}")
            i += 1
        print(f"{var_name}\n{l}")        

    def print_shapes(self):
        self.print_list(self.biases,"biases")
        self.print_list(self.weights,"weights")
            