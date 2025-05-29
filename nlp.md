# Introduction to NLP

Natural Language Processing (NLP) uses artifical intelligence to
understand and generate text in human languages. NLP is already
ubiquitious in everyday life, whether it be with voice-powered
assistants and online chatbots. What's exciting about NLP is its
potential to extend into the realm of sentiment analysis---that is,
its ability to analyse and recognise emotions, feelings, sarcasm,
mood, and other insights from largely unstructured text data. This, in
turn, constitutes a remarkable leap in data analysis.

## NLTK

Natural Language Toolkit (NLTK) is a Python library designed for
processing of English text. Here are some key concepts related to text
processing.
- Tokenising: breaking up text into its words
- Stop words: repeated words that don't add much to meaning and are
usually featured out (e.g., 'in', 'as', 'of')
- Stemming: reducing words to their root
- Lematizing: reducing words to their core meaning (e.g., best -> good)
- Chuncking: breaking up text into phrases---often the meaning of
individual words doesn't bear much relation to the phrase that it's
in

This week I began experimenting with the NLTK library, writing a simple script
to tokenize and lemmatize a text file before printing out the most common words,
both before and after taking out stop words.
A plot of the frequency distribution is also generated and saved to a file.

[Code](W1/textstats.py)

# Week 2 - PyTorch
The basics of using PyTorch to create a neural network is as follows.

- Import training and testing data using the `Dataset` class and wrap it up with
  `Dataloader` to allow it to be fed to the model in batches.
- Define a class that inherits from ``nn.Module``; this class should have a
  `forward` function that passes the inputs through the model. This function can
explicitly apply linear transformations to the data, or the class can be
initialised with an instance of the `Sequential` class that makes it easy to
define a sequential series of layers for the data to be fed through.
Use `nn.linear` to create a module that applies a linear transformation to the
input data.
- Choose a loss and paramter optimisation function.
- To train the model, enumerate the training data dataloader (which will come in
  batches of the batch size), calculate `cost` using the loss function and then
use `cost.backward()` to calculate the gradient with respect to each parameter
in the model. The `step` method of the optimiser then updates the paramters. Use
the `zero_grad` method of the optimiser to reset the gradient---otherwise it
accumulates.
- To test the model, use `torch.no_grad`  to ensure no gradients are computed.
- Train and test the model over a number of epochs.

As a very simple example to get started, I generated values of the sine function
at many inputs and implemented my own `Dataset` class to load the data from a
CSV file. Then I trained a simple model to calculate the sine of its single
input.

[Code](W2/experiment.py)

(The model does well---but only on data within the range of its input! Passing
$100\pi$ to the model, for example, returns rubbish.
