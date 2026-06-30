<div align="center">

![Tiny Language Model banner](assets/tlm_banner.png)

# Tiny Language Model (TLM)

A small educational language model built in Python with PyTorch to understand how text can be transformed into numerical representations and used to predict the next word in a sequence.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Network-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![Tkinter](https://img.shields.io/badge/Interface-Tkinter-1f2937?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## Overview

**Tiny Language Model (TLM)** is a personal educational project focused on building a simple word-level language model from the ground up using PyTorch.

The goal of the project is not to create a production-ready large language model, but to understand the core ideas behind modern NLP systems: tokenization, vocabularies, context windows, embeddings, logits, softmax, loss functions and next-word prediction.

Given a short context such as:

```text
la derivada de
```

the model tries to predict the most likely next word according to the mathematical text dataset used during training.

> **Note**  
> This project is educational. It is intentionally small and interpretable, designed to make the internal structure of a language model understandable.

---

## Main Features

- Word-level text preprocessing and tokenization.
- Vocabulary construction with special tokens: `<pad>`, `<unk>` and `<eos>`.
- Context-window based next-word prediction.
- Trainable word embeddings.
- Feedforward neural network implemented with PyTorch.
- Cross-entropy loss for classification over the vocabulary.
- Model training with Adam optimizer.
- Accuracy calculation over the training examples.
- Graphical interface built with Tkinter.
- Model, vocabulary and training configuration persistence.
- Custom Spanish mathematical datasets.

---

## Model Architecture

The implemented model follows this pipeline:

```text
Raw text
   ↓
Tokenization
   ↓
Vocabulary indexing
   ↓
Context window
   ↓
Embedding layer
   ↓
Flattened vector
   ↓
Feedforward neural network
   ↓
Logits
   ↓
Softmax / Argmax
   ↓
Predicted next word
```

For the current trained configuration:

```text
context_size = 5
embedding_dim = 32
vocab_size = 345
```

Each input consists of 5 previous words. Each word is converted into a 32-dimensional embedding vector:

```text
5 words × 32 dimensions = 160 input values
```

The neural network architecture is:

```text
160 → 120 → 60 → 345
```

Where:

- `160` is the flattened embedding representation of the context.
- `120` and `60` are hidden layers with ReLU activation.
- `345` is the output size, one logit for each word in the vocabulary.

---

## Architecture Diagram

```mermaid
flowchart LR
    A[Spanish mathematical text] --> B[Tokenizer]
    B --> C[Word indices]
    C --> D[Context window: 5 words]
    D --> E[Embedding layer: 32D per word]
    E --> F[Flatten: 5 × 32 = 160]
    F --> G[Linear layer: 160 → 120]
    G --> H[ReLU]
    H --> I[Linear layer: 120 → 60]
    I --> J[ReLU]
    J --> K[Linear layer: 60 → vocab_size]
    K --> L[Logits]
    L --> M[Predicted next word]
```

---

## Training Configuration

The current saved training configuration is:

| Parameter | Value |
|---|---:|
| Vocabulary size | 345 |
| Context size | 5 |
| Embedding dimension | 32 |
| Epochs | 100 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Training examples | 5,153 |
| Final cost | 0.6368 |
| Final accuracy | 79.86% |

The reported accuracy is a strict **top-1 next-word accuracy**. In natural language tasks, this is a demanding metric because several words may be reasonable continuations for the same context.

---

## Repository Structure

```text
TLM/
├── interface.py                         # Graphical interface and training workflow
├── tlm_class.py                         # Tiny Language Model class
├── main_testing.py                      # Small testing script
├── datasets_tlm_matematicas/            # Spanish mathematical text datasets
│   ├── matematicas_algebra.txt
│   ├── matematicas_aritmetica.txt
│   ├── matematicas_calculo.txt
│   ├── matematicas_lineal_matrices.txt
│   ├── matematicas_probabilidad_estadistica.txt
│   ├── matematicas_preguntas_respuestas.txt
│   └── matematicas_dataset_completo.txt
├── tlm_model.pth                        # Saved trained model
├── vocab.json                           # Saved vocabulary
├── training_config.json                 # Saved training configuration
├── docs/
│   └── Memoria_TLM.pdf                  # Full academic report
├── assets/
│   └── tlm_banner.png                   # README banner
└── README.md
```

---

## How It Works

### 1. Text normalization

The input text is converted to lowercase, accents are removed and unnecessary characters are filtered out.

Example:

```text
La derivada de una función
```

becomes:

```text
la derivada de una funcion
```

### 2. Tokenization

The normalized text is split into words:

```text
["la", "derivada", "de", "una", "funcion"]
```

### 3. Vocabulary construction

Each unique word receives an integer index. The model also uses three special tokens:

| Token | Meaning |
|---|---|
| `<pad>` | Padding token used to complete short contexts |
| `<unk>` | Unknown word token |
| `<eos>` | End-of-sentence token |

### 4. Context creation

The model receives a fixed-size context and learns to predict the following word.

Example:

```text
Context: la derivada de una
Target: funcion
```

### 5. Embeddings

Each word index is transformed into a dense vector. This allows the model to learn internal numerical representations of words.

### 6. Prediction

The network outputs one score, or logit, for each word in the vocabulary. The word with the highest probability is selected as the prediction.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/TLM.git
cd TLM
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required dependency:

```bash
pip install torch
```

---

## Running the Project

Launch the graphical interface:

```bash
python interface.py
```

From the interface, you can:

1. Select a dataset.
2. Configure the vocabulary size, context size and embedding dimension.
3. Train the model.
4. Save the model and vocabulary.
5. Enter a short context.
6. Predict the next word.

---

## Example Contexts

You can test the model with short mathematical prompts such as:

```text
la derivada de
```

```text
una matriz identidad
```

```text
la probabilidad de
```

```text
el limite de
```

The quality of the prediction depends on the size and content of the training dataset.

---

## Educational Purpose

This project was built to understand how a language model works internally at a small scale.

Instead of using a pre-trained transformer or a high-level NLP framework, the project implements the main components step by step:

- text preprocessing,
- word indexing,
- context windows,
- embeddings,
- neural network forward propagation,
- loss computation,
- backpropagation through PyTorch,
- gradient descent using an optimizer,
- next-word prediction.

It represents a conceptual transition from a classical neural network for numerical/image data to a neural network capable of processing language.

---

## Limitations

This model is intentionally simple. It does not include:

- attention mechanisms,
- transformers,
- subword tokenization,
- large-scale training,
- long-term memory,
- semantic reasoning,
- autoregressive multi-word text generation.

The model predicts one word at a time using a fixed-size context window.

---

## Possible Improvements

Future improvements could include:

- Top-k accuracy evaluation.
- Multi-word autoregressive text generation.
- Train/validation split.
- Better dataset cleaning.
- Larger and more diverse corpus.
- Temperature-based sampling.
- Attention mechanisms.
- Transformer-based version.
- Exporting training curves.
- Improving the graphical interface.

---

## Academic Report

A complete academic explanation of the project is included in:

```text
docs/Memoria_TLM.pdf
```

The report explains the main ideas behind the project, including embeddings, context windows, the network architecture, the training process and future improvements.

---

## License

This project is licensed under the MIT License.

---

<div align="center">

Built as a personal educational project to understand the foundations of language models.

</div>
