# JUST FOR TESTING. UNUPDATED. IGNORE.

import tlm_class
import torch
import datasets_tlm_matematicas     # custom dataset: list of sentences with mathematical expressions in Spanish
    

if __name__ == '__main__':
    """
    This is the main file of the project. It will be used to test the neural network.
    Let's keep in mind that we are working with tensors, rather than np.matrix or np.array. This is because PyTorch is a library for deep learning,
    and it is optimized for working with tensors :)
    """

    vocab_size = 343
    context_size = 5
    embedding_dim = 32

    TLM = tlm_class.TLM(vocab_size, context_size, embedding_dim) 
    optimizer = torch.optim.Adam(TLM.parameters(), lr=0.001)  

    # Example context:
    # word indices [12, 37, 81, 22]
    x = torch.tensor([[12, 37, 81, 22]], dtype=torch.long)

    # Correct next word index
    target = torch.tensor([95], dtype=torch.long)

    final_cost = TLM.trainStep(x, target, epochs=100, optimizer=optimizer)

    prediction = TLM.prediction(x)

    print("Final cost:", final_cost)
    print("Prediction:", prediction)  
    