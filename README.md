# Deep-Learning-Tutorial      
This repository contains Deep Learning tutorial projects.

## Important terms in Deep Learning        

## What is deep learning?          
Deep learning is one of the area of machine learning which focuses on deep artificial neural network.     
It has application in various field computer vision,Speech recognition and Natural Language Processing.     

## How deep networ is better than artificial neural network/shallow network?         
Deep network have several hidden layer , so they are able to extract batter feature.       

## What is cost function?        
Cost function is a measure of the accuracy of the neural network with respect to given training sample and expected output.   
It can be calculated as mean square error function on system output and desired output.     

## What is gradient Descent?         
Gradient descent is an optimization algorith,which is used to learn the value of training parameters that minimizes the cost function.We compute the gradient descent of cost function for given parameter and update the parameter.     

## What is backpropagation?       
Backpropagation is training algorithm used for multilayer neural network.It is used gradient calculation to reduce the cost function to minimize error function to get update the parameter for a better network.     
It follows below steps :       
1. forward propagation of training data to get the output.      
2.Using target value and output value , error derivative can be computed with respect to output activation.        
3.Then we back propagate for computing derivative of the error , with respect to output activation on previous and continue this for all the hidden layer.         
4.Using previously calculated derivative for output and all hidden layer we calculate error derivative with respect to weight.       
5. And then we update the weight.        
  
## What are the three varients of Gradient descent: batch,stochastic and mini-batch?       
1.Stochastic Gradient Descent:     
We use only single training example for calculation of gradient and update the parameter.      
2.Batch Gradient Descent :          
We calculate the gradient for whole datset and perform the update at each iteration.      
3.Mini-batch Gradient Descent :          
It is similar to Stochastic Gradient Descent , here instead of single training example, mini-batch of sample is used.      

What are the benefits of mini-batch gradient descent?           
- This is more efficient compare to stochastic gradient descent.       
- Generalization for finding the flat minima.        
- Mini batches allows help to approximate the gradient of entire training set which help us to avoid local minima.        

## What is data normalization?         
Data normalization is used during backpropagation.Main motive behind data normalization is to reduce or eliminate data redundancy. Here we rescale value to fit into specific range to achieve better convergence.         

## What is weight initialization in neural network?        
Good weight initialization help in giving a quicker convergence and a better overall error.         
Biases can be generally initialized to zero.Weight should be close to zero without being too small.          

## What is an auto-encoder?          
It is an autonomous Machine Learning algorithm that usages backpropagation algorithm , where target value is set to be equal to the input provided.It has a hidden layer that describe the code used to represet the input.         
1. It is unsupervised machine learning algorithm, similar to principal component analysis.       
2. It minimizes the same objective function as principal component analysis.        
3. The neural network taget output is it's input.      

## What is Boltzmann Machine?        
It is used to optimize the solution of a problem.The work is basically to optimize the weight and quantity for the given problem.       
- It uses recurrent structure.     
- It consist of stochastic neurons, which consist one of two possible state 0 or 1.         
- The neuron in this either in adaptive (free state ) or clamped (frozen state).        

## What is the role of activation function?             
Activation function is used to introduce non-linearity into the neural network, helping it to learn more complex function.
Without which it will be able to learn linear function which is linear combination of the input data.


