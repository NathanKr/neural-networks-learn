<h2>Motivation</h2>
My experience has taught me that the best way to learn a new subject is to code it yourself so here i have python code to do neural network learning using backpropagation and SGD. 
I must say that 90% of the code here was not written by me , in this case the best thing that i can do is fully understand every line of code that i am using from another person code

<h2>Points of interest</h2>
<ul>
<li>the code is based on the excellent source of Michael Nielsen <a href="http://neuralnetworksanddeeplearning.com/chap2.html">book</a> and <a href="https://github.com/mnielsen/neural-networks-and-deep-learning">code</a> with about 10% changes by me</li>
<li>for 5000 data set samples i get training score of 97.68% check <a href="https://github.com/NathanKr/neural-networks-learn/releases/tag/1.1">here</a>. However, it is meaning less without the score on the test data set. for training samples of 4950 and test of 50 the test score is 92% , for 4800 and test of 200 the test score is 89% , for 4700 and 300 the test score is 83 , for 4600 and 400 the test score is 66.75% , for 4550 and 450 the test score is 0% !!!! -----> i was able to solve this <a href="https://github.com/NathanKr/neural-networks-learn/releases/tag/1.2">here</a> the reason is the mnist dataset : first 500 data set sample are for digit 0 , next 500 is digit 1 and so on .Thus training 4500 and test on last 500 try to test on digit 9 which was not trained. the solution is to shuffle the data set at the begining</li>
<li>learning curves (learning_curves.png , <a href="https://github.com/NathanKr/neural-networks-learn/releases/tag/1.3">tag</a>): 1. data set of 200 sampling was enough to get an error of ~ 5% 2. prediction of test and training is almost the same which is very good - no variance problem thus no need for regularization 3. avarging was needed on the learning curve because it matter which part of the data set is used</li>
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
    <li>hyper tuning variable used are : # nurons on the hidden layer , batch_size , learning_rate, epochs. i started with #nurons 10 , batch_size 5000 , learning_rate 0.01 , epochs : 40 and looked at the percentage of correct output estimated with feedforward compared to the actual one. i got better and better result by reducing batch_size by half until i got to batch_size of 1. then started to use number of neurons 10,20,30,40,50,60 and saw best results for 30. meanwhile i saw that reducing the epochs to 10 does not deteriate result. and lastly reduce learning_rate by 2 until suprisingly got 96.36% success for learning_rate of 1 on training set !!! thus the final stup is : 30 neurons in the hidden layer,batch_size of 1 , learning_rate of 1 , ephocs=10</li>
    <li>actually after i got 96.36% i used ephocs=20 and got even better results of 97.68% this make me wonder if there is a known procedure \ best practices for hyper tuning</li>
    <li>deviding the data set for training and test i was also able to get also good results : 4500 training \ 500 test -> 92% , 4000 training \ 1000 test 90.7%  </li>
    </ul>
    </td>
  </tr>
</table>

<h2>Need more investigation</h2>
<ul>
<li>look for more structure \ step by step way to do hyper tuning. can i do mini machine learning model for them ??</li>
<li>i am getting very good results for training - 97.68%. it is interesting to understand is there a common denominator between the images that were not classified correctly</li>
</ul>
