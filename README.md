## The Repo

This repo implements a basic Bigram language model from scratch with vanilla RUST.

This is a multi-layer perceptron.

This code backpropagates every parameter without any external library.

This model generates the most probable word after a given sequence of words.

## Env vars

```
    FILE_PATH - Address to a text file containing the text that will be used to train the model
    CONTEXT_LENGTH - The size of the context window
    DIMENSIONS - The number of channels
    NEURONS - The number of neurons at the hidden layers
    MAX_STEPS - The amount of training loops
    BATCH_SIZE - The size of the batches used at training
    LEARNING_RATE - The learning rate that adjust the parameters. 
```

## References:
- Kaiming initialization 
    
    - https://arxiv.org/pdf/1502.01852v1.pdf

- Batch normalization

    - https://arxiv.org/pdf/1502.03167.pdf

- Layer normalization

    - https://arxiv.org/pdf/1607.06450