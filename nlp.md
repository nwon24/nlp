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

# Week 3 - Vectorisation and classification algorithms
To pass text to a classification algorithm or model, it must be vectorised. This
is because an algorithm can operate on text. Algorithms for turning text into
numerical input are called vectorisers. 

A simple vectoriser is called Bag of Words, where each word in the text is
associated with its frequency count in the text. While simple, this approach
results in common words such as prepositions being weighted higher than other
words that may be more important to the meaning of the document.

A more complicated approach is called term frequency-inverse document frequency,
or TF-IDF. This algorithm is applied to a collection of texts (corpus). The term
frequency, TF, of a particular word in a particular document is $f/n$, where
$f$ is the number of times the word appears in the document and $n$ is the
number of words in the document. The inverse document frequency, or IDF, is
$\log(N/F)$, where $N$ is the number of documents and $F$ is the number of
documents in which the word appears. The TF-IDF number for that word is the
product of its TF and IDF scores.

This means that words common to all the documents are given little weighting,
which is appropriate because words like 'the' appear in virtually everything and
don't add much meaning to a document. Words that appear frequently in one
document but not the others will be given a higher weighting, and this is
appropriate because it's much more likely that word is significant to the
meaning of that document.

The TF-IDF vectorisers found in tookits such as `sklearn` implement smoothing
and other normalisation to change the TF-IDF scores slightly from the simplistic
model described above, but the idea is the same.

Once a corpus has been vectorised, it can be fed into a classification
algorithm. One such algorithm is called logistic regression, which is similar to
linear regression except that the output is often binary (categorial), not
continous. This is useful, for example, in classifying whether an email is spam
or not, or whether a movie review is positive or negative.

The idea is to take an input vector, say $x$, and then apply a linear
transformation to it in the form $z=w\cdot x+b$, where $\cdot$ represents the
dot product. However, this $z$ could be any real number, and therefore we then
pass it to the function $$\sigma(z)=\frac{1}{1+e^{-z}}$$ which spits out a
number between $0$ and $1$. This gives us a probability that the input belongs
to one of the two specified classes. We can then apply a decision boundary to
determine whether the probability is high enough for us to conclude that the
input belongs to that category.

The weights and biases that go into the linear transformation of the input are
given by stochastic gradient descent.

Another classification algorithm is called Support Vector Machine (SVM). In this
method inputs are represented as points in $n$-dimensional space, and the idea
is to find the hyperplane that best separates the two groups of data. This is done
by maximising the distance between the hyperplane to its closest points.

[TF-IDF Pipeline using NLTK](W3/vectorise.ipynb)

[SVM and Logistic Regression using sklearn](W3/classifiers.ipynb)

This week I used both of these classifiers on some reviews on Amazon. 
The corpus was split into two categories: positive and negative reviews. Where
the models' predictions were different to the actual label, I have saved those
reviews to separate files for analysis.

[SVM Model Results](W3/SVM_results.md)

[Logistic Regression Model Results](W3/log_results.md)

Some of the predictions seem to be plain wrong; but others seem to be wrong when
there is some subtlety or nuance to the review's overall opinion. For example, in
one review, a book was praised as an overall `good read' but most of it was spent
criticising the style of the author.

# Week 4 - Neural networks for text classification

This week I moved on from baseline models to neural networks. 

Convolutional Neural Networks have the same basic structure as other neural
networks: the input, a series of hidden layers in which transformations are made
to the data via weights and biases, and then the output layer. 

A convolutional neural network is a special type of neural network in which some
of the layers are convolutional layers. This means instead of connecting every
neuron in the previous layer to every neuron in the next (fully connected
layer), the next layer is obtained by sliding a 'window' called the kernel or
filter over the input layer; this window consists of weights and biases that are
applied to the part of the input over which the kernel covers. 
Thus the next layer consists of the outputs of these convolutions. 
The idea is that different filters can recognise different features of the
input. The CNN is built up of several convolutional layers, each with multiple
filters, each recognising different aspects of the input. 

To avoid overfitting, a technique called pooling is used, in which the size of
the output of a convolutional layer is reduced by taking the average or maximum
of a group of outputs, increasing computational efficiency in the process.
A CNN often ends with fully connected layers, which take the output of the
previous convolutional and pooling layers to produce the final output of the
network.

CNNs are often used for image recognition--two dimensional data. In this case I
have built a CNN for my one dimensional input data, which is just text. The text
is vectorised using TF-IDF, and the input to the network consists of a tensor of
TF-IDF values. The final output layer of the CNN has only one neuron, as the
task is classifying whether the text is positive or negative: the closer to 1,
the higher the chance of the text being positive.

[CNN text classifier](W4/cnn.ipynb)

With only a single convolutional and fully connected layer, the validation test
results reach about the same level of accuracy as the baseline model (around
86%). Increasing the number of epochs or the number or layers only resulted in
the validation accuracy decreasing due to overfitting. To increase the accuracy
of the model, it is likely more data is needed.

(Strangely, the model gets stuck using PyTorch but not Keras... Even with
PyTorch as the backend for Keras.)

## Strange issue

The PyTorch network doesn't seem to be learning at all---the cost is always around 0.7
or thereabouts, and no changes to any of the hyperparameters seems to affect it.
This is true with both the CNN and the simple fully connected network. However, when
the network is implemented using Keras and run on the same input data (TF-IDF
vectorisation of the dataset), the model does seem to learn and the validation accuracy
reaches around 85% (roughly on par with the baseline classifiers). And using PyTorch
as the Keras backend gives the same result...

A head scratcher, for now.
[CNN text classifier with Amazon reviews](W4/amazon_review_polarity_csv/cnn_Amazon.ipynb)

After fiddling around with the learning rate, the manual implementation seems to work
with a learning rate of $10^{-3}$. Mysteriously, sometimes the cost won't go
anywhere and remains stuck at around 0.7, other times it will jump around but
decrease overall, resulting in validation test accuracy of around 83%.

# Week 5 - RNNs and LSTMs

Normal neural networks assume that each piece of data in the input is independent
of the other pieces of data. Unfortunately, this assumption is not correct for
textual data, because often the meaning of a word or phrase depends on the words
that have come before it. To a human this idea of context is intuitive; but
traditional neural networks are unable to capture this link.

Recurrent Neural Networks (RNNs) are a kind of neural network that aims to solve
this problem. As such, they are often used on streams of textual data to predict
the next word at the end of a phrase or sentence. 

RNNs work by keeping a 'memory' of the input data that has been received up
until a particular time; RNNs work in time steps, with each time step involving
the processing of one piece of the input (e.g., the next word). The processing
of each piece of input is done by 'cells.' The idea behind a RNN is that each
cell accepts the next piece of input and produces an output called the 'hidden
state' that is then fed into the next cell, taking into account its memory of
what has occured in the previous time steps. As with CNNs, the output of the
recurrent layers is fed into a dense layer that spits out the output required by
the network.

How these cells actually work relates to the particular type of RNN being used.
One type of RNN is called Long Short-Term Memory (LSTM) and is the network I
tried to implement this week for my text classification problem.

## Embeddings

One thing not done in the previous weeks' foray into text classification using NNs
was the inclusion of an embedding layer. In this case it was necessary because
the input to the LSTM needs to be sequential. In earlier weeks the input
consisted of sparse arrays of vectorised texts (using TF-IDF), where each word
always occupied the same index in the array. However, this completely disregards
the sequential aspect of text processing, as well as the relationship that might
arise between words in the context of a particular review. Word embeddings are a
way to solve this problem.

First I used a vectoriser to create a dictionary of unique words in the corpus,
assigning to each word an integer or index. Then I converted each piece of text
into an array of indices to represent the words, padding them to be the same
length as the longest review. This could be passsed directly into a network, but
no success was achieved. :(

Then I found out about embeddings, which are a way of converting those arrays of
indices into higher dimensional vectors, whose entries are initialised randomly.
As the network learns, the entries in those vectors are fine-tuned (learned) to
reflect more accurately the semantic relationships between the words in the
text. So essentially the embedding layer is just another layer in the network,
which takes as input a list of indices (or this can be thought of as a large
matrix of zeroes with a single 1 indicating which word is represented) and
applied a linear transformation to it via the embedding matrix to output a
vector for each word in the text. This output is then fed into the LSTM layer.

## How LSTMs work

An LSTM layer consists of a series of LSTM cells, each of which process each piece
of input data for for each time step. As input is fed through the LSTM,
something called the cell state is maintained, which acts like the memory of the
LSTM. Each cell produces an output called the hidden state, which is then fed
into the next cell as part of its input.

So each LSTM cell takes three pieces of input: $h_{t-1}$, the previous hidden
state, $c_{t-1}$, the previous cell state, and $x_t$, the current input. These
three pieces of data are fed through three 'gates':

- forget gate: this decides which information in the cell state is relevant (and
  ''forgets'' the rest);
- input gate: this decides which information in the input ($x_t$) is relevant
  and then uses it to produce a new cell state;
- output gate: this determines the hidden state to be output from the cell,
  given the new cell state.

### The forget gate

To ease notation, denote by $L(x)$ a linear transformation of the vector $x$
with a set of weights and a bias. Pointwise multiplication is denoted by
$\otimes$ and pointwise addition is denoted by $\oplus$.

The computation for this gate is $$f_t=\sigma(L(x_t)+L(h_{t-1})).$$
The values in $f_i$ are between $0$ and $1$, and so multiplying this with $c_t$
pointwise effectively ''forgets'' some of its values and ''remembers'' others. 

### The input gate

We now need to decide how to update the cell state. The change to the cell state
is given by $$\Delta c_t=\tanh(L(h_{t-1})+L(x_t)).$$
In other words, we compute the change to the cell state via a linear
transformation of the two inputs, $H_{t-1}$ and $x_t$; but we also want to make
sure that the irrelevant parts of the input are forgotten. This is achieved by
computing $$i_t=\sigma(L(h_{t-1})+L(x_t));$$
that is, another sigmoid activate linear layer, but with a different set of
weights and biases to the forget gate. 
Note that $\tanh$ is used because the output is between $-1$ and $1$; this is
because we might want to decrease some of the values in the cell state as well
as add them.

Thus, to compute the new cell state, we first forget some of the old cell state,
decide which parts of $\Delta c_t$ we want to keep, and then pointwise add the
two vectors together, viz. $$c_t=c_{t-1}\otimes f_t\oplus\Delta c_t\otimes i_t.$$

### The output gate

We want to now output some part of the cell state---but only the part most
relevant to the input and previous hidden state. So we compute, again,
$$o_t=\sigma(L(h_{t-1})+L(x_t))$$ and then spit out the new hidden state
$$h_t=o_t\otimes\tanh(c_t).$$ (Here $\tanh$ is used to normalise the hidden
states.)

And that's it! This new hidden state is then fed into the next LSTM cell with
the next input, and this along with the current cell state goes through all the
same operations again. 

## Project update

Finally got the homebaked LSTM to work---accuracy approximately 82%. I needed to
flatten the output of the LSTM differently so that all timesteps was passed to the
dense layer; with this fix somehow the model started to actually train.

[Text classifier using LSTM in PyTorch and Keras](W5/LSTM.ipynb)
