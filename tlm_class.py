import torch
import torch.nn as nn

# Attention: batch_size determines how many samples we will use to train the neural network in one iteration.

class TLM(nn.Module):
    """
    This class defines the whole **Tiny Language Model (TLM)**.

    We are creating a four-layer-based neural network:
    
        vocab_size -> 120 -> 60 -> vocab_size

    The first and last layers both have vocab_size neurons.
    The second and third layers are hidden layers with 120 and 60 neurons.

    Powered by PyTorch :)
    """

    def __init__(self, vocab_size, context_size, embedding_dim):
        """
        Method to initialize the neural network.

        _Differences between vocab_size, context_size and embedding_dim:_
        - _vocab_size_: number of words in the vocabulary (number of unique words in the dataset)
        - _context_size_: number of words in the context window (number of words used to predict the next word)
        - _embedding_dim_: number of dimensions of the word embeddings (the size of the dense vector that represents each word)
        """

        super().__init__()  

        self.vocab_size = vocab_size
        self.context_size = context_size 
        self.embedding_dim = embedding_dim
        
        # Layers of the neural network
        self.layer1 = nn.Linear(embedding_dim * context_size, 120)  # our input is a vector with embedding_dim * context_size elements, 
                                                                    # since we are using a context window of size context_size      
        self.layer2 = nn.Linear(120, 60)            
        self.layer3 = nn.Linear(60, vocab_size)

        # Torch methods for activation and loss function
        self.relu = nn.ReLU()
        self.lossFunction = nn.CrossEntropyLoss()

        # Embedding layer to convert word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        """
        Input:
            x: tensor with word indices.
            Shape: (batch_size, context_size)

        Output:
            logits.
            Shape: (batch_size, vocab_size)
        """

        embedded = self.embedding(x)

        # embedded shape:
        # (batch_size, context_size, embedding_dim)

        embedded = embedded.flatten(start_dim=1)

        # embedded shape:
        # (batch_size, context_size * embedding_dim)

        z1 = self.layer1(embedded)
        a1 = self.relu(z1)

        z2 = self.layer2(a1)
        a2 = self.relu(z2)

        z3 = self.layer3(a2)    # logits

        return z3
    
    def prediction(self, input):
        """
        Input: vector with word indices.
        Output: predicted word index.
        """

        with torch.no_grad():                                       # we are not saving any calculations from here, this is just a prediction

            logits = self.forward(input)                            # we get z_j^(L)
            logitsPostSoftmax = torch.softmax(logits, dim=-1)       # we softmax the logits
            prediction = torch.argmax(logitsPostSoftmax, dim=-1)    # then obtain the maximum of the logits
                                                                    # notice: dim=-1 is cause we are softmaxing the last layer                                                        
        return prediction
    
    def cost(self, logits, target):
        """
        Input: logits and target (in order to obtain de loss). (No softmax in this input)
        Output: real number (of course possitive).
        """

        return self.lossFunction(logits, target)

    def backpropagation(self, cost, optimizer):
        """
        This method applies backpropagation and gradient descent.

        cost.backward() computes the gradients of the loss function.
        optimizer.step() updates the weights and biases of the neural network.
        """

        optimizer.zero_grad()       # we need to set the gradients to zero before starting to do backpropragation 
                                    # because PyTorch accumulates the gradients on subsequent backward passes.
        cost.backward()             # this is the backpropagation
        optimizer.step()            # this is the gradient's descend

    def trainStep(self, input, target, epochs, optimizer):
        """
        This method trains the neural network with one input and one target for a number of epochs. 
        It is a simple implementation of the training process.

        This is just for educational purposes, to understand how the training process works.
        """

        total_cost = 0

        for e in range(epochs):
            logits = self(input)                     # we get the logits (z_j^(L)) of the neural network for the input x
                                                     # is the same as logits = self.forward(input)
            cost = self.cost(logits, target)         # we get the cost (loss)
            self.backpropagation(cost, optimizer)    # we apply backpropagation and gradient descent

            total_cost += cost.item()

        return total_cost / epochs
    
    def accuracy(self, input, target, batch_size=32):
        """
        This method calculates the accuracy of the neural network.

        Input:
            input: tensor with word indices.
                Shape: (number_of_examples, context_size)

            target: tensor with the correct next word indices.
                    Shape: (number_of_examples)

        Output:
            accuracy percentage.
        """

        was_training = self.training
        self.eval()

        total_predictions = 0
        correct_predictions = 0

        with torch.no_grad():

            for start in range(0, input.shape[0], batch_size):

                batch_input = input[start:start + batch_size]
                batch_target = target[start:start + batch_size]

                logits = self(batch_input)

                predictions = torch.argmax(logits, dim=-1)

                correct_predictions += (predictions == batch_target).sum().item()
                total_predictions += batch_target.shape[0]

        if was_training:
            self.train()

        accuracy = correct_predictions / total_predictions

        return accuracy * 100