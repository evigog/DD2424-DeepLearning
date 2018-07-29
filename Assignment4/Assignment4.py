import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy


def DataPrep():
    #read book contents
    book_fname = 'goblet_book.txt';
    with open(book_fname) as f:
        book_data = f.read()

    #get unique characters of book contents
    book_chars = list(set(book_data))

    vocab_size = len(book_chars)

    #create dictionaries of the unique characters
    ind_to_char = {idx: key for idx, key in enumerate(book_chars)}
    char_to_ind = {key: idx for idx, key in enumerate(book_chars)}

    return (book_data, book_chars, vocab_size, ind_to_char, char_to_ind)

def CharToOneHot(char, char_to_ind, letter_length):
    one_hot = np.zeros((letter_length, 1))
    one_hot[char_to_ind[char]] = 1
    return one_hot

def OneHotToChar(one_hot, ind_to_char):
    idx = one_hot.tolist().index(1)
    char = ind_to_char[idx]
    return char

def SoftMax(z):
    """
    returns softmax for each column of the input matrix
    """
    e = np.exp(z - np.max(z))
    return((e / e.sum(axis = 0)))

class RNNgrads:
    def __init__(self, theRNN, seq_length):
        # biases
        self.b = np.zeros(np.shape(theRNN.b))
        self.c = np.zeros(np.shape(theRNN.c))

        # weights
        self.U = np.zeros(np.shape(theRNN.U))
        self.W = np.zeros(np.shape(theRNN.W))
        self.V = np.zeros(np.shape(theRNN.V))

        #intermediary
        self.a = np.zeros((theRNN.m, seq_length))
        self.h = np.zeros((theRNN.m, seq_length + 1))  # because we also store the initial inner state

    def Compute(self, theRNN, seq_length, X, Y, a, h, p):
        grad_o = p-Y #for all timesteps

        #these gradients can be calculated straighforwardly
        for t in range(seq_length):
            self.c += grad_o[:, t].reshape(-1, 1)
            self.V += np.dot(grad_o[:, t].reshape(-1, 1), h[:, t + 1].reshape(1, -1))

        #for last ht - as h has one more column than the others
        self.h[:, -1] = np.dot(grad_o[:, -1].reshape(1, -1), theRNN.V)

        #for all previous timesteps
        for t in range(seq_length-1, -1, -1):
            self.a[:, t] = np.dot(self.h[:, t+1].reshape(1, -1), np.diag(1 - np.power(np.tanh(a[:, t]), 2)))
            self.h[:, t] = np.dot(grad_o[:, t].reshape(1, -1), theRNN.V) + np.dot(self.a[:, t].reshape(1, -1), theRNN.W)

            self.b += self.a[:, t].reshape(-1,1)
            self.U += np.dot(self.a[:, t].reshape(-1,1), X[:,t].reshape(1, -1))
            self.W += np.dot(self.a[:, t].reshape(-1,1), self.h[:, t].reshape(1,-1))


class RNN:
    def __init__(self, letter_length, hidden_length):

        # dimensionality of RNN's hidden state
        self.m = hidden_length

        # input and output digit length - equal to one-hot representation of a letter
        self.K = letter_length

        # biases
        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))

        # weights
        self.U = np.random.normal(0, 0.01, (self.m, self.K))
        self.W = np.random.normal(0, 0.01, (self.m, self.m))
        self.V = np.random.normal(0, 0.01, (self.K, self.m))

    def BasicForwPass(self, hprev, xt):
        at = np.dot(self.W, hprev) + np.dot(self.U, xt) + self.b
        ht = np.tanh(at)
        ot = np.dot(self.V, ht) + self.c
        pt = SoftMax(ot)

        return (at,ht,ot,pt)

    def SynthText(self, x0, h0, seq_length):
        txt = np.zeros((self.K, seq_length))

        hprev = h0
        xt = x0

        for t in range(seq_length):
            at, ht, ot, pt = self.BasicForwPass(hprev, xt)

            #update hprev
            hprev = ht

            #given pt select next input xt
            cp = np.cumsum(pt)
            a = random.uniform(0, 1)
            char_idx = np.where(cp - a > 0)[0][0] #double indexing because np.where returns a tuple with a single element
            #convert character index to one hot representation
            xt = np.zeros((self.K, 1))
            xt[char_idx] = 1

            txt[:, t] = xt.flatten()

        return txt

    def ForwardPass(self, X, Y, h0, seq_length):
        Loss = 0
        a = np.zeros((self.m, seq_length))
        h = np.zeros((self.m, seq_length+1)) #because we also store the initial inner state
        p = np.zeros((self.K, seq_length))

        h[:,0] = h0.flatten()
        hprev = h0
        for t in range(seq_length):
            yt = Y[:, t].reshape(-1, 1)
            xt = X[:,t].reshape(-1,1)
            at, ht, ot, pt = self.BasicForwPass(hprev, xt)
            Loss = Loss - np.log(np.dot(yt.transpose(), pt))

            hprev = ht

            a[:, t] = at.flatten()
            h[:, t+1] = ht.flatten()
            p[:, t] = pt.flatten()

        return Loss, a, h, p



def Main():
    #RNN sequence length
    seq_length = 25# given
    #dimensionality of hidden state
    hidden_length = 100# given

    # learning rate
    eta = 0.1  # default

    book_data, book_chars, vocab_size, ind_to_char, char_to_ind = DataPrep()
    # vocabulary size equal to one-hot representation length of a letter
    letter_length = vocab_size
    theRNN = RNN(letter_length, hidden_length)

    x0 = CharToOneHot(".", char_to_ind, letter_length)
    h0 = np.zeros((theRNN.m, 1))
    txt = theRNN.SynthText(x0, h0, seq_length)

    readable_text = ""
    for i in range(seq_length):
        readable_text += OneHotToChar(txt[:,i], ind_to_char)
    print(readable_text)

    X_chars = book_data[0: seq_length]
    X = np.zeros((theRNN.K, seq_length))
    Y_chars = book_data[1: seq_length + 1]
    Y = np.zeros((theRNN.K, seq_length))

    for i in range(seq_length):
        X[:, i] = CharToOneHot(X_chars[i], char_to_ind, letter_length).flatten()
        Y[:, i] = CharToOneHot(Y_chars[i], char_to_ind, letter_length).flatten()

    Loss, a, h, p = theRNN.ForwardPass(X, Y, h0, seq_length)
    print(Loss)

    theGrads = RNNgrads(theRNN, seq_length)
    theGrads.Compute(theRNN, seq_length, X, Y, a, h, p)

Main()


