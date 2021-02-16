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
    <li>hyper tuning variable used are : # nurons on the hidden layer , batch_size , learning_rate, epochs. i started with #nurons 10 , batch_size 5000 , learning_rate 0.01 , epochs : 40 and looked at the percentage of correct output estimated with feedforward compared to the actual one. i got better and better result by reducing batch_size by half until i got to batch_size of 1. then started to use number of neurons 10,20,30,40,50,60 and saw best results for 30. meanwhile i saw that reducing the epochs to 10 does not deteriate result. and lastly reduce learning_rate by 2 until suprisingly ig 96.36% success for learning_rate of 1 !!! thus the final stup is : 30 neurons in the hidden layer,batch_size of 1 , learning_rate of 1 , ephocs=10</li>
    <li>actually after i got 96.36% i used ephocs=20 and got even better results of 97.68% this make me wonder if there is a known procedure \ best practices for hyper tuning</li>
    </ul>
    </td>
  </tr>
  
</table>
