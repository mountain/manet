# multiply-accumulate-network

Multiply–accumulate network is multiply–accumulate-only and activation-free and can be used as a building block for deep
neural networks.

Multiply–accumulate operation is a generalization of the perceptron and the linear neuron. But surprisingly,
it can be the only mechanism to construct deep neural networks. The non-linearity can be eliminated, 
we can easily see the point that exponential functions are already included by pure multiplications,
so we can achieve a wide range of non-linearity by using different combinations of multiplications add additions.

It also owns a beautiful mathematical background from the field of arithmetic expression geometry.

## Demo

We demonstrate the performance of multiply–accumulate network on MNIST dataset.

We gradually replace the non-linear activation functions with mac operations, and the performance is almost the same.
And sometimes, the performance is even better. Please check mnist0.py to mnist3.py for details.


