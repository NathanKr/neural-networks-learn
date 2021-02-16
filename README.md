<h2>Motivation</h2>
My experience has taught me that the best way to know a new subject is to code it yourself so here i have python code to do neural network learning. 
I must say that 90% of the code here was not written by me , in this case the best thing that i can do is fully understand every line of code that i am using from another person code

<h2>Points of interest</h2>
<ul>
<li>the code is based on the excellent source of Michael Nielsen <a href="http://neuralnetworksanddeeplearning.com/chap2.html">book</a> and <a href="https://github.com/mnielsen/neural-networks-and-deep-learning">code</a> with about 10% changes by me</li>
</ul>

<h2>Content</h2>
<table>
    <tr>
    <td>mnist_nn_playground.py</td>
    <td>
    <ul>
    <li>the data set is from Andrew Ng machine learning course @ Coursera, 5000 samples , each sample input is 20x20 and output sample is scalar</li>
    <li>ploting the 20x20 images it looks rotated not clear why. i do not know how it will affect learning , i think it will not</li>
    <li>looking at the output vector y is the the following coding : 0 has code 10 all other digit has code same as digit</li>
    <li>it is solved with neural network : layer 1 must have 400 neuron because each sample is 20x20 . using sigmoid activation function last layer must have 10 nurons each representing a digit</li>
    <li>using 10 nurons on last layer we need to update y accordingly to be 5000x10</li>
    </ul>
    </td>
  </tr>
  
</table>
