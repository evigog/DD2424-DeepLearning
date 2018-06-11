import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import random
import os


class NonInteger(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors

def ComputeGradsNumSlow(X, Y, W, b, lamda, h):

    grad_W = np.zeros(np.shape(W))
    grad_b = np.zeros(np.shape(b))
    for i in range(np.shape(b)[0]):
        b_try = np.copy(b)
        b_try[i] = b_try[i] - h
        c1 = ComputeCost(X, Y, W, b_try, lamda)
        b_try = np.copy(b)
        b_try[i] = b_try[i] + h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = np.divide((c2-c1),(2*h))

    for i in range(np.shape(W)[0]):
        for j in range(np.shape(W)[1]):
            W_try = np.copy(W)
            W_try[i, j] = W_try[i, j] - h
            c1 = ComputeCost(X, Y, W_try, b, lamda)
            W_try = np.copy(W)
            W_try[i, j] = W_try[i, j] + h
            c2 = ComputeCost(X, Y, W_try, b, lamda)
            grad_W[i, j] = np.divide((c2-c1),(2*h))

        print("continuing...")

    return (grad_W, grad_b)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def LoadBatch(filename, numoflabels):
    parentdir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    filename = parentdir + "/Assignment1/cifar-10-batches-py/" + filename
    dict = unpickle(filename)

    #images
    X = (dict[b'data']/255.0).transpose()
    #labels
    y = np.array(dict[b'labels'])
    #one hot representation of labels
    Y = np.zeros((numoflabels, X.shape[1]))
    for i, elem in enumerate(y):
        Y[elem, i] = 1

    return (X, Y, y)

def ToZeroMean(Xtrain, Xval, Xtest):
    mean_X = np.mean(Xtrain, axis=0).reshape(1,-1)
    Xtrain = Xtrain - mean_X
    Xval = Xval - mean_X
    Xtest = Xtest - mean_X
    return Xtrain, Xval, Xtest


def GetDimensions(X):
    #dim of each image
    d = np.shape(X)[0]
    #num of images
    N = np.shape(X)[1]

    return (d, N)


def InitParams(K1, K2, d):
    #same seed every time for testing purposes
    # np.random.seed(0)
    W1 = np.random.normal(0, 0.001, (K1, d))
    W2 = np.random.normal(0, 0.001, (K2, K1))
    b1 = np.zeros((K1, 1))
    b2 = np.zeros((K2, 1))
    return (W1, W2, b1, b2)

def SoftMax(z):
    """
    returns softmax for each column of the input matrix
    """
    e = np.exp(z - np.max(z))
    return((e / e.sum(axis = 0)))

def ReLU(s):
    below_zero_indices = s < 0
    s[below_zero_indices] = 0
    return s

def EvaluateClassifier(X, W1, b1, W2, b2):
    """
    apply forward pass and return
    the output probabilities of the classifier
    """
    s1 = np.matmul(W1, X) + b1
    h = ReLU(s1)
    s2 = np.matmul(W2, h) + b2
    P = SoftMax(s2)

    return P, h, s1

def ComputeCost(X, Y, W, b, lamda):
    """
    X: each column of it corresponds to an image and the whole matrix has size dx n
    Y: each column of Y (Kx n) is the one-hot ground truth label for the corresponding column of X
    W: weights
    b: bias
    lamda: legularization parameter
    Ν: number of images
    :return: a scalar corresponding to the sum of the loss of the network's predictions for the images in X relative to
            the ground truth labels and the regularization term on W
    """
    d, N = GetDimensions(X)
    regularization = lamda * np.sum(np.power(W, 2))

    P, h, s1 = EvaluateClassifier(X, W, b)
    cross_entropy_loss = 0 - np.log(np.sum(np.prod((np.array(Y), P), axis=0), axis=0))


    J = (1/N) * np.sum(cross_entropy_loss) + regularization
    return J

def ComputeAccuracy(X, y, W, b):
    """
    X: each column of X corresponds to an image and X has size dn.
    y: Y is the vector of ground truth labels of length n.
    W:
    b:
    acc: acc is a scalar value containing the accuracy.
    """
    P, h, s1 = EvaluateClassifier(X, W, b)
    predicitons = np.argmax(P, axis=0)
    correct = np.count_nonzero((y-predicitons) == 0)#because of the condition it actually counts the zeros
    all = np.size(predicitons)
    if all == 0:
        raise(ZeroDivisionError("division by zero in ComputeAccuracy"))
    acc = correct/all

    return acc

def IndXPositive(x):
    above_zero_indices = x > 0
    below_zero_indices = x <= 0
    x[above_zero_indices] = 1
    x[below_zero_indices] = 0

    return x


def ComputeGradients(X, Y, W1, b1, W2, b2, lamda):
    """
    :param X: each column of X corresponds to an image and it has size d xn
    :param Y: each column of Y (Kx n) is the one-hot ground truth label for the corresponding column of X
    :param W:
    :param lamda:
    :return: grad_W is the gradient matrix of the cost J relative to W and has size K xd - same as W
             grad_b is the gradient vector of the cost J relative to b and has size K x1 - same as b
    """

    grad_W1 = np.zeros(np.shape(W1))
    grad_b1 = np.zeros(np.shape(b1))
    grad_W2 = np.zeros(np.shape(W2))
    grad_b2 = np.zeros(np.shape(b2))

    # each column of P contains the probability for each label for the image in the corresponding column of X.
    # P has size Kx n
    # h is the hidden layer output of the network during the forward pass
    # h has size K1 x N
    P, h, s1 = EvaluateClassifier(X, W1, b1, W2, b2)

    N = np.shape(X)[1]
    for i in range(N):
        Yi = Y[:, i].reshape((-1, 1))
        Pi = P[:, i].reshape((-1, 1))
        Xi = X[:, i].reshape((-1,1))
        hi = h[:, i].reshape((-1, 1))
        si = s1[:, i]

        g = - np.transpose(Yi - Pi)
        grad_b2 = grad_b2 + np.transpose(g)
        grad_W2 = grad_W2 +  np.matmul(np.transpose(g), np.transpose(hi))

        g = np.matmul(g, W2)
        g = np.matmul(g, np.diag(IndXPositive(si))) #???? -> gia to matmul kai to transpose

        grad_b1 = grad_b1 + np.transpose(g)
        grad_W1 = grad_W1 + np.matmul(np.transpose(g), np.transpose(Xi)) #???? -> gia to matmul kai to transpose


    grad_b1 = np.divide(grad_b1, N)
    grad_W1 = np.divide(grad_W1, N) + 2 * lamda * W1

    grad_b2 = np.divide(grad_b2, N)
    grad_W2 = np.divide(grad_W2, N) + 2 * lamda * W2

    return (grad_W1, grad_b1, grad_W2, grad_b2)

def CompareGrads(WA, bA, WB, bB):
    print("********\nW\n********")
    for i in range(np.shape(WA)[0]):
        for j in range(np.shape(WA)[1]):
            diff = abs(WA[i, j] - WB[i, j]) / max(abs(WA[i, j]), abs(WB[i, j]))
            if (diff > 1e-7):
                print(i, j, WA[i, j], WB[i, j], diff)

    print("********\nb\n********")
    for i in range(np.size(bA)):
        diff = abs(bA[i] - bB[i]) / max(1e-6, abs(bA[i]) + abs(bB[i]))
        if (diff > 1e-7):
            print(i, bA[i], bB[i], diff)

def CheckGrads(X, Y, W, b, lamda, how_many):
    randomdatapoints = random.sample(range(0, np.shape(X)[1]), how_many)

    X = X[:, randomdatapoints]
    Y = Y[:, randomdatapoints]

    gW, gb = ComputeGradients(X, Y, W, b, lamda)
    gWnumSl, gbnumSl = ComputeGradsNumSlow(X, Y, W, b, lamda, 1e-6)
    CompareGrads(gWnumSl, gbnumSl, gW, gb)


def MiniBatchGD(X, Y, GDparams, W, b, lamda):
    """
    :param X: all the training images
    :param Y: the labels for the training images
    :param GDparams: an object containing the parameter values n_batch, eta and n_epochs
    :param W: and :param b: the initial values for the network's parameters
    :param lamda: regularization factor
    :return: Wstar, bstar final trained network parameters
    """
    n_batch = GDparams["n_batch"]
    eta = GDparams["eta"]
    n_epochs = GDparams["epochs"]

    #all datapoints
    N = np.shape(X)[1]

    if N % n_batch != 0:
        raise (NonInteger("non integer number of datapoints per step",1))
    steps_per_epoch = int(N / n_batch)

    allW = [W]
    allb = [b]

    for ep in range(n_epochs):
        for st in range(steps_per_epoch):
            batch_start = st * n_batch
            batch_end = batch_start + n_batch
            Xbatch = X[:, batch_start:batch_end]
            Ybatch = Y[:, batch_start:batch_end]
            grad_W, grad_b = ComputeGradients(Xbatch, Ybatch, W, b, lamda)
            W = W - eta * grad_W
            b = b - eta * grad_b

        allW.append(W)
        allb.append(b)

        print("continuing...")


    return (W, b, allW, allb)

def PlotLoss(Xtrain, Ytrain, Xval, Yval, allW, allb, lamda, eta):
    train_cost = []
    val_cost = []

    #calculate costs
    for i, W in enumerate(allW):
        b = allb[i]

        cost = ComputeCost(Xtrain, Ytrain, W, b, lamda)
        train_cost.append(cost)
        cost = ComputeCost(Xval, Yval, W, b, lamda)
        val_cost.append(cost)

    #plot
    idx = np.arange(len(train_cost))
    plt.plot(idx, train_cost, label="Training")
    plt.plot(idx, val_cost, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([0, 10, 20, 30, 40])
    plt.title("Loss development over epochs, using eta " + str(eta) + " and lambda " + str(lamda))
    plt.show()

def VisualizeW(W, eta, lamda, epochs, batch):
    fig, ax = plt.subplots(4, int(np.shape(W)[0]/2))
    fig.suptitle("Learnt weights using eta " + str(eta) + ", lambda " + str(lamda) + ", " + str(epochs) + " epochs and batch of " + str(batch))
    for i in range(np.shape(W)[0]):
        im = np.reshape(W[i, :], (32, 32, 3), order="F")
        im = (im - np.amin(im)) / (np.amax(im) - np.amin(im))
        im = np.transpose(im, (1,0,2))
        row = int(i//(np.shape(W)[0]/2))
        col = int(i%(np.shape(W)[0]/2))
        ax[row, col].imshow(im, interpolation="none")
        ax[row, col].set_yticklabels([])
        ax[row, col].set_xticklabels([])
    plt.show()

def Main():
    try:
        #mode = "check" for gradient checking
        #       "default" for default training
        mode = "default"

        #constants
        # lamda = regularization parameter
        lamda = 0
        # eta = learning rate
        eta = 0.01
        n_batch = 100
        epochs = 40

        # K =num of labels
        K1 = 50
        K2 = 10

        # N = num of input examples
        # d = dimension of each input example
        # X = images (d x N)
        # Y = one-hot labels (K x N)
        # y = labels (N)
        Xtrain,Ytrain,ytrain = LoadBatch("data_batch_1", K2)
        Xval, Yval, yval = LoadBatch("data_batch_2", K2)
        Xtest, Ytest, ytest = LoadBatch("test_batch", K2)

        Xtrain, Xval, Xtest = ToZeroMean(Xtrain, Xval, Xtest)


        # d = dim of each image
        # N = num of images
        d, N = GetDimensions(Xtrain)

        # W1 = weights K1 x d
        # b1 = bias K1 x 1
        # W2 = weights K2 x K1
        # b2 = bias K2 x 1
        W1, W2, b1, b2 = InitParams(K1, K2, d)
        ComputeGradients(Xtrain, Ytrain, W1, b1, W2, b2, lamda)
        pass

        # if mode == "check":
        #     # check
        #     CheckGrads(Xtrain, Ytrain, W, b, lamda, 3)
        # else:
        #     #train
        #     Wstar, bstar, allW, allb = MiniBatchGD(Xtrain, Ytrain, {"eta": eta, "n_batch":n_batch, "epochs": epochs}, W, b, lamda)
        #     #plot loss on training and validation dataset
        #     PlotLoss(Xtrain, Ytrain, Xval, Yval, allW, allb, lamda, eta)
        #     #calculate accuracy on training dataset
        #     test_accuracy = ComputeAccuracy(Xtest, ytest, Wstar, bstar)
        #     print(test_accuracy)
        #     #visualize weights
        #     VisualizeW(Wstar, eta, lamda, epochs, n_batch)

        print("done")

    except ZeroDivisionError as err:
        print(err.args)




Main()