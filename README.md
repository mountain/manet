# manet

"Manet" is a library for MAC networks.

MAC networks are networks consisted only by [multiply–accumulate operations](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation), and they are activation-free and can be used as building blocks for deep neural networks.

Multiply–accumulate operation is a generalization of the perceptron and the linear neuron. But surprisingly,
it can be the only mechanism to construct deep neural networks. The non-linearity can be eliminated,  we can easily see the point that:
* exponential functions is pure multiplicative
* a wide range of non-linear functions can be constructed by using different combinations of multiplications add additions.

It also owns a beautiful mathematical background from the field of arithmetic expression geometry.

## Demo

We demonstrate the performance of multiply–accumulate network on MNIST dataset.

We gradually replace the non-linear activation functions with MAC operations, and the performance is almost the same and sometimes is even better.
Please check mnist0.py to mnist3.py for details.

```bash
python -m demo.train -m mnist0
python -m demo.train -m mnist1
python -m demo.train -m mnist2
python -m demo.train -m mnist3
```

Note: mnist3.py is buggy and conv replacement is still ongoing.
