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

def PlotLoss(smooth_losses):
    steps = []
    for i,loss in enumerate(smooth_losses):
        steps.append((i+1)*100)

    plt.plot(steps, smooth_losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    # plt.legend()
    plt.title("Loss development over two epochs")#, using eta " + str(eta) + " and lambda " + str(lamda), y=1.03)
    # plt.show()
    filename = "loss.png"
    plt.savefig(filename)
    print("saved file", filename)

class NUMgrads:
    def __init__(self, theRNN):
        # biases
        self.b = np.zeros(np.shape(theRNN.b))
        self.c = np.zeros(np.shape(theRNN.c))

        # weights
        self.U = np.zeros(np.shape(theRNN.U))
        self.W = np.zeros(np.shape(theRNN.W))
        self.V = np.zeros(np.shape(theRNN.V))

    def ComputeGrad(self, fieldname, theRNN, seq_length, X, Y, h):
        grad = getattr(self, fieldname) #already initialized to zeros
        h0 = np.zeros((theRNN.m, 1))

        #only if grad == 2D
        if len(grad.shape) == 2:
            for i in range(grad.shape[0]):
                for j in range(grad.shape[1]):
                    RNN_try = deepcopy(theRNN)
                    attr_try = getattr(RNN_try, fieldname)
                    attr_try[i, j] -= h
                    Loss1, dummy1, dummy2, dummy3 = RNN_try.ForwardPass(X, Y, h0, seq_length)
                    RNN_try = deepcopy(theRNN)
                    attr_try = getattr(RNN_try, fieldname)
                    attr_try[i, j] += h
                    Loss2, dummy1, dummy2, dummy3 = RNN_try.ForwardPass(X, Y, h0, seq_length)
                    grad[i, j] = (Loss2 - Loss1) / (2 * h)
        #else if grad == 1D
        else:
            for i in range(grad.shape[0]):
                RNN_try = deepcopy(theRNN)
                attr_try = getattr(RNN_try, fieldname)
                attr_try[i] -= h
                Loss1, dummy1, dummy2, dummy3 = RNN_try.ForwardPass(X, Y, h0, seq_length)
                RNN_try = deepcopy(theRNN)
                attr_try = getattr(RNN_try, fieldname)
                attr_try[i] += h
                Loss2, dummy1, dummy2, dummy3 = RNN_try.ForwardPass(X, Y, h0, seq_length)
                grad[i] = (Loss2 - Loss1) / (2 * h)

    #given code in matlab
    # function grad = ComputeGradNum(X, Y, f, RNN, h) #f einai to sygkekrimeno fieldname #h einai
    #     n = numel(RNN.(f));
    #     grad = zeros(size(RNN.(f)));
    #     hprev = zeros(size(RNN.W, 1), 1);
    #     for i=1:nRNN_try = RNN;
    #         RNN_try.(f)(i) = RNN.(f)(i) - h;
    #         l1 = ComputeLoss(X, Y, RNN_try, hprev);
    #         RNN_try.(f)(i) = RNN.(f)(i) + h;
    #         l2 = ComputeLoss(X, Y, RNN_try, hprev);
    #         grad(i) = (l2 - l1) / (2 * h)


    def Compute(self, theRNN, seq_length, X, Y, h):
        fieldnames = list(self.__dict__.keys())
        for i in range(len(fieldnames)):
            self.ComputeGrad(fieldnames[i], theRNN, seq_length, X, Y, h)

    def Compare(self, theGrads):
        fieldnames = list(self.__dict__.keys())
        for f in range(len(fieldnames)):
            num_grad = getattr(self, fieldnames[f])
            grad = getattr(theGrads, fieldnames[f])

            print(fieldnames[f])
            # only if grad == 2D
            if len(grad.shape) == 2:
                for i in range(np.shape(grad)[0]):
                    for j in range(np.shape(grad)[1]):
                        diff = abs(grad[i, j] - num_grad[i, j]) / max(1e-6, (abs(grad[i, j]) + abs(num_grad[i, j])))
                        if (diff > 1e-5):
                            print(i, j, grad[i, j], num_grad[i, j], diff)
            #else if grad == 1D
            else:
                for i in range(np.size(grad)):
                    diff = abs(grad[i] - num_grad[i]) / max(1e-6, (abs(grad[i]) + abs(num_grad[i])))
                    if (diff > 1e-5):
                        print(i, grad[i], num_grad[i], diff)




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
        self.h = np.zeros((theRNN.m, seq_length))

    def Compute(self, theRNN, seq_length, X, Y, a, h, p):
        grad_o = p-Y #for all timesteps

        #these gradients can be calculated straighforwardly
        for t in range(seq_length):
            self.c += grad_o[:, t].reshape(-1, 1)
            self.V += np.dot(grad_o[:, t].reshape(-1, 1), h[:, t + 1].reshape(1, -1))

        #for last ht - as h has one more column than the others
        self.h[:, -1] = np.dot(grad_o[:, -1], theRNN.V)
        self.a[:, -1] = np.dot(self.h[:, -1], np.diag(1 - np.power(np.tanh(a[:, -1]), 2)))

        #for all previous timesteps
        for t in range(seq_length-2, -1, -1):
            self.h[:, t] = np.dot(grad_o[:, t], theRNN.V) + np.dot(self.a[:, t+1], theRNN.W)
            self.a[:, t] = np.dot(self.h[:, t], np.diag(1 - np.power(np.tanh(a[:, t]), 2)))

        for t in range(seq_length):
            self.b += self.a[:, t].reshape(-1, 1)
            self.U += np.dot(self.a[:, t].reshape(-1, 1), X[:, t].reshape(1, -1))
            self.W += np.dot(self.a[:, t].reshape(-1, 1), h[:, t].reshape(1, -1))

    def Clip(self):
        fieldnames = list(self.__dict__.keys())
        for f in range(len(fieldnames)):
            if fieldnames[f] != 'a' and fieldnames[f] != 'h':
                grad = getattr(self, fieldnames[f])
                grad = np.maximum(np.minimum(grad, 5), -5)
                setattr(self, fieldnames[f], grad)
        pass



class RNN:
    def __init__(self, letter_length, hidden_length, **kwargs):
        #first, default constructor
        if len(kwargs) == 0:

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

            #adagrad ms
            self.m_b = np.zeros((self.m, 1))
            self.m_c = np.zeros((self.K, 1))

            self.m_U = np.zeros((self.m, self.K))
            self.m_W = np.zeros((self.m, self.m))
            self.m_V = np.zeros((self.K, self.m))


        #constructor used for deep copying
        else:
            # dimensionality of RNN's hidden state
            self.m = hidden_length

            # input and output digit length - equal to one-hot representation of a letter
            self.K = letter_length

            # biases
            self.b = kwargs['b']
            self.c = kwargs['c']

            # weights
            self.U = kwargs['U']
            self.W = kwargs['W']
            self.V = kwargs['V']

            #adagrad ms
            self.m_b = kwargs['m_b']
            self.m_c = kwargs['m_c']

            self.m_U = kwargs['m_U']
            self.m_W = kwargs['m_W']
            self.m_V = kwargs['m_V']



    # this is copied (but modified) from stackoverflow
    def __deepcopy__(self, memo):  # memo is a dict of id's to copies
        id_self = id(self)  # memoization avoids unnecessary recursion
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                deepcopy(self.K, memo),
                deepcopy(self.m, memo),
                b=deepcopy(self.b, memo),
                c=deepcopy(self.c, memo),
                U=deepcopy(self.U, memo),
                W=deepcopy(self.W, memo),
                V=deepcopy(self.V, memo),
                m_b=deepcopy(self.m_b, memo),
                m_c=deepcopy(self.m_c, memo),
                m_U=deepcopy(self.m_U, memo),
                m_W=deepcopy(self.m_W, memo),
                m_V=deepcopy(self.m_V, memo))
            memo[id_self] = _copy
        return _copy


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

    def AdaGrad(self, theGrads, eta, e):
        fieldnames = list(theGrads.__dict__.keys())
        for f in range(len(fieldnames)):
            if fieldnames[f] != 'a' and fieldnames[f] != 'h':
                weight_matrix = getattr(self, fieldnames[f])
                grad = getattr(theGrads, fieldnames[f])
                m = getattr(self, "m_"+fieldnames[f])

                m = m + np.power(grad, 2)
                weight_matrix = weight_matrix - (eta * grad)/np.sqrt(m + e)

                setattr(self, fieldnames[f], weight_matrix)
                setattr(self, "m_"+fieldnames[f], m)

        pass


def Main():
    #RNN sequence length
    seq_length = 25# given
    #dimensionality of hidden state
    hidden_length = 100# given for default training #5#to check grads #

    # learning rate
    eta = 0.1  # default

    book_data, book_chars, vocab_size, ind_to_char, char_to_ind = DataPrep()
    # vocabulary size equal to one-hot representation length of a letter
    letter_length = vocab_size
    theRNN = RNN(letter_length, hidden_length)


    # #synthesize readable text
    # x0 = CharToOneHot(".", char_to_ind, letter_length)
    # h0 = np.zeros((theRNN.m, 1))
    # txt = theRNN.SynthText(x0, h0, seq_length)
    #
    # readable_text = ""
    # for i in range(seq_length):
    #     readable_text += OneHotToChar(txt[:,i], ind_to_char)
    # print(readable_text)
    # ############################

    # X_chars = book_data[0: seq_length]
    # X = np.zeros((theRNN.K, seq_length))
    # Y_chars = book_data[1: seq_length + 1]
    # Y = np.zeros((theRNN.K, seq_length))
    #
    # for i in range(seq_length):
    #     X[:, i] = CharToOneHot(X_chars[i], char_to_ind, letter_length).flatten()
    #     Y[:, i] = CharToOneHot(Y_chars[i], char_to_ind, letter_length).flatten()
    #
    # h0 = np.zeros((theRNN.m, 1))
    # Loss, a, h, p = theRNN.ForwardPass(X, Y, h0, seq_length)
    #
    # theGrads = RNNgrads(theRNN, seq_length)
    # theGrads.Compute(theRNN, seq_length, X, Y, a, h, p)

    # #check grads
    # theNumGrads = NUMgrads(theRNN)
    # theNumGrads.Compute(theRNN, seq_length, X, Y, h=1e-4)
    # theNumGrads.Compare(theGrads)
    # ############################

    # theGrads.Clip()

    #high level trainining loop
    epoch = 0
    hprev = np.zeros((theRNN.m, 1)) #initializing hidden state

    #keep track of smooth loss in order to plot that later
    smooth_losses = []

    while epoch < 2:
        e = 0  # where we are in the book
        step = 0 # step in epoch
        while e+seq_length+1 < len(book_data):
            X_chars = book_data[e: e+seq_length]
            X = np.zeros((theRNN.K, seq_length))
            Y_chars = book_data[e+1: e+seq_length+1]
            Y = np.zeros((theRNN.K, seq_length))

            for i in range(seq_length):
                X[:, i] = CharToOneHot(X_chars[i], char_to_ind, letter_length).flatten()
                Y[:, i] = CharToOneHot(Y_chars[i], char_to_ind, letter_length).flatten()

            Loss, a, h, p = theRNN.ForwardPass(X, Y, hprev, seq_length)
            theGrads = RNNgrads(theRNN, seq_length)
            theGrads.Compute(theRNN, seq_length, X, Y, a, h, p)
            theGrads.Clip()

            theRNN.AdaGrad(theGrads, eta, 1e-8)


            hprev = h[:, -1].reshape(-1, 1)
            e += seq_length

            if step == 0 and epoch == 0:
                smooth_loss = Loss
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * Loss
                if step % 100 == 0:
                    print("Epoch: ", epoch, " Step: ", step, " Smooth loss: ", smooth_loss)
                    smooth_losses.append(int(smooth_loss))


            step += 1

        epoch += 1

    PlotLoss(smooth_losses)

Main()


