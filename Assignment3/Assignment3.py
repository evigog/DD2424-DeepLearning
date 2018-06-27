import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import random
import os
from copy import deepcopy


class NonInteger(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors

def ComputeGradsNum(X, Y, W, b, lamb, h):
    #initialize grads
    grad_W = []
    grad_b = []

    for i in range(len(W)):
        Wnew = np.zeros(np.shape(W[i]))
        grad_W.append(Wnew)
        bnew = np.zeros(np.shape(b[i]))
        grad_b.append(bnew)

    c = ComputeCost(X, Y, W, b, lamb)

    for k in range(len(W)):
        for i in range(len(b[k])):
            b_try = deepcopy(b)
            b_try[k][i] += h
            c2 = ComputeCost(X, Y, W, b_try, lamb)
            grad_b[k][i] = (c2 - c) / h

        for i in range(W[k].shape[0]):
            for j in range(W[k].shape[1]):
                W_try = deepcopy(W)
                W_try[k][i, j] += h
                c2 = ComputeCost(X, Y, W_try, b, lamb)
                grad_W[k][i, j] = (c2 - c) / h

    return grad_W, grad_b

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

    neededmeans = np.shape(Xval)[1]
    Xval = Xval - mean_X[:, :neededmeans]

    neededmeans = np.shape(Xtest)[1]
    Xtest = Xtest - mean_X[:, :neededmeans]
    return Xtrain, Xval, Xtest


def GetDimensions(X):
    #dim of each image
    d = np.shape(X)[0]
    #num of images
    N = np.shape(X)[1]

    return (d, N)


def InitParams(nodes):
    #same seed every time for testing purposes
    np.random.seed(123)
    W = []
    b = []
    for i, nodesnum in enumerate(nodes):
        if i == 0:
            continue
        Wi = np.random.normal(0, 0.001, (nodes[i], nodes[i-1]))
        W.append(Wi)
        bi = np.zeros((nodes[i], 1))
        b.append(bi)
    return (W, b)

def SoftMax(z):
    """
    returns softmax for each column of the input matrix
    """
    e = np.exp(z - np.max(z))
    return((e / e.sum(axis = 0)))

def ReLU(s):
    s = np.maximum(s, 0)
    return s

def EvaluateClassifier(X, W, b):
    """
    apply forward pass and return
    the output probabilities of the classifier
    """
    # W = [W1, W2]
    # b = [b1, b2]
    h = [X]
    s = []

    for i in range(len(W)):
        si = np.dot(W[i], h[-1]) + b[i]
        s.append(si)
        hi = ReLU(si)
        h.append(hi)

    #in the last layer apply softmax
    P = SoftMax(s[-1])

    #detach X from h list (was appended just to help generalisation)
    h.pop(0)

    # s1 = np.dot(W1, X) + b1
    # h = ReLU(s1)
    # s2 = np.dot(W2, h) + b2
    # P = SoftMax(s2)

    return P, h, s

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
    layers = len(W)

    regularization = 0
    for l in range(layers):
        regularization += lamda * np.sum(np.power(W[l], 2))

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
    predictions = np.argmax(P, axis=0)
    correct = np.count_nonzero((y-predictions) == 0)#because of the condition it actually counts the zeros
    all = np.size(predictions)
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


def ComputeGradients(X, Y, W, b, lamda):
    """
    :param X: each column of X corresponds to an image and it has size d xn
    :param Y: each column of Y (Kx n) is the one-hot ground truth label for the corresponding column of X
    :param W:
    :param lamda:
    :return: grad_W is the gradient matrix of the cost J relative to W and has size K xd - same as W
             grad_b is the gradient vector of the cost J relative to b and has size K x1 - same as b
    """

    #initialize gradients
    layers = len(W)

    grad_W = []
    grad_b = []
    for i in range(layers):
        grad_W.append(np.zeros(np.shape(W[i])))
        grad_b.append(np.zeros(np.shape(b[i])))

    # each column of P contains the probability for each label for the image in the corresponding column of X.
    # P has size Kx n
    # h is the hidden layer output of the network during the forward pass
    # h has size K1 x N
    P, h, s = EvaluateClassifier(X, W, b)

    N = np.shape(X)[1]
    for i in range(N):
        #last layer
        Yi = Y[:, i].reshape((-1, 1))
        Pi = P[:, i].reshape((-1, 1))
        hi = h[-2][:, i].reshape((-1, 1))

        g = Pi - Yi

        grad_b[-1] = grad_b[-1] + g
        grad_W[-1] = grad_W[-1] + np.dot(g, np.transpose(hi))

        #for the next layers
        #progressing from second-to-last to first
        for l in range(layers-2, -1, -1):

            si = s[l][:, i]
            if l == 0:
                hi = X[:, i].reshape((-1, 1))
            else:
                hi = h[l-1][:, i].reshape((-1, 1))


            #propagate error backwards
            g = np.dot(np.transpose(W[l+1]), g)
            g = np.dot(np.diag(IndXPositive(si)), g)

            grad_b[l] = grad_b[l] + g
            grad_W[l] = grad_W[l] + np.dot(g, np.transpose(hi)) #???? h or s ???

    for l in range(layers):
        grad_b[l] = np.divide(grad_b[l], N)
        grad_W[l] = np.divide(grad_W[l], N) + 2 * lamda * W[l]

    return (grad_W, grad_b)

def CompareGrads(WA, bA, WB, bB):
    print("********\nW\n********")
    for i in range(np.shape(WA)[0]):
        for j in range(np.shape(WA)[1]):
            diff = abs(WA[i, j] - WB[i, j]) / max(1e-6, (abs(WA[i, j])+ abs(WB[i, j])))
            if (diff > 1e-4):
                print(i, j, WA[i, j], WB[i, j], diff)

    print("********\nb\n********")
    for i in range(np.size(bA)):
        diff = abs(bA[i] - bB[i]) / max(1e-6, (abs(bA[i]) + abs(bB[i])))
        if (diff > 1e-4):
            print(i, bA[i], bB[i], diff)

def CheckGrads(X, Y, W, b, lamda, how_many):
    randomdatapoints = [11,12]#random.sample(range(0, np.shape(X)[1]), how_many)

    X = X[10:15, randomdatapoints]
    Y = Y[:, randomdatapoints]
    W[0] = W[0][:, 10:15]

    gW, gb = ComputeGradients(X, Y, W, b, lamda)
    gWnumSl, gbnumSl = ComputeGradsNum(X, Y, W, b, lamda, 1e-5)
    for l in range(len(gW)):
        CompareGrads(gWnumSl[l], gbnumSl[l], gW[l], gb[l])


def MiniBatchGD(X, Y, GDparams):
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
    lamda = GDparams["lamda"]
    rho = GDparams["rho"]
    lr_decay = GDparams["lr_decay"]

    W = GDparams["W"]
    b = GDparams["b"]

    layers = len(W)

    #all datapoints
    N = np.shape(X)[1]

    if N % n_batch != 0:
        raise (NonInteger("non integer number of datapoints per step",1))
    steps_per_epoch = int(N / n_batch)


    #initialize list of lists to keep track of all Ws and bs - in each step
    #so allW[0][0] -> initial Ws for layer 0
    #   allW[1][0] -> initial Ws for layer 1
    allW = [[] for l in range(layers)]
    allb = [[] for l in range(layers)]
    for l in range(layers):
        allW[l].append(W[l])
        allb[l].append(b[l])


    #keeping track of momentum
    momW = []
    momb = []
    for l in range(layers):
        momW.append(np.zeros(np.shape(W[l])))
        momb.append(np.zeros(np.shape(b[l])))

    for ep in range(n_epochs):
        for st in range(steps_per_epoch):
            batch_start = st * n_batch
            batch_end = batch_start + n_batch
            Xbatch = X[:, batch_start:batch_end]
            Ybatch = Y[:, batch_start:batch_end]
            grad_W, grad_b = ComputeGradients(Xbatch, Ybatch, W, b, lamda)

            #applying momentum to the update
            for l in range(layers):
                momW[l] = rho * momW[l] + eta * grad_W[l]
                W[l] = W[l] - momW[l]

                momb[l] = rho * momb[l] + eta * grad_b[l]
                b[l] = b[l] - momb[l]

        eta = lr_decay * eta

        for l in range(layers):
            allW[l].append(W[l])
            allb[l].append(b[l])

        print("continuing...")

    return (W, b, allW, allb)

def PlotLoss(Xtrain, Ytrain, Xval, Yval, allW, allb, lamda, eta, mode):
    train_cost = []
    val_cost = []

    layers = len(allW)

    #calculate costs
    for i in range(len(allW[0])):
        W = [allW[l][i] for l in range(layers)]
        b = [allb[l][i] for l in range(layers)]
        # W = [allW[0][i], allW[1][i]]
        # b = [allb[0][i], allb[1][i]]

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
    plt.title("Loss development over epochs, using eta " + str(eta) + " and lambda " + str(lamda))
    # plt.show()
    filename = mode + "eta" + str(eta) + "lamda" + str(lamda) + ".png"
    plt.savefig(filename)
    print("saved file", filename)
    plt.clf()

    return (train_cost[-1], val_cost[-1])

def OpenAllData(num_of_labels):
    X, Y, y = LoadBatch("data_batch_1", num_of_labels)
    X2, Y2, y2 = LoadBatch("data_batch_2", num_of_labels)
    X3, Y3, y3 = LoadBatch("data_batch_3", num_of_labels)
    X4, Y4, y4 = LoadBatch("data_batch_4", num_of_labels)
    X5, Y5, y5 = LoadBatch("data_batch_5", num_of_labels)

    X = np.concatenate((X, X2, X3, X4, X5), axis=1)
    Y = np.concatenate((Y, Y2, Y3, Y4, Y5), axis=1)
    y = np.concatenate((y, y2, y3, y4, y5))

    return X, Y, y

def OpenData(num_of_labels, mode):
    if mode == "search" or mode == "check" or mode == "sanitycheck":
        #open one batch
        Xtrain, Ytrain, ytrain = LoadBatch("data_batch_1", num_of_labels)
        Xval, Yval, yval = LoadBatch("data_batch_2", num_of_labels)
        Xtest, Ytest, ytest = LoadBatch("test_batch", num_of_labels)

        if mode == "sanitycheck":
            Xtrain, Ytrain, ytrain = Xtrain[:, :100], Ytrain[:, :100], ytrain[:100]
            Xval, Yval, yval = Xval[:, :100], Yval[:, :100], yval[:100]
            Xtest, Ytest, ytest = Xtest[:, :100], Ytest[:, :100], ytest[:100]

    else:
        # open all data batches
        X, Y, y = OpenAllData(num_of_labels)

        Xtrain = X[:, 0:-1000]
        Ytrain = Y[:, 0:-1000]
        ytrain = y[0:-1000]

        Xval = X[:, -1000:]
        Yval = Y[:, -1000:]
        yval = y[-1000:]

        Xtest, Ytest, ytest = LoadBatch("test_batch", num_of_labels)

    return (Xtrain, Ytrain, ytrain, Xval, Yval, yval, Xtest, Ytest, ytest)


def Main():
    try:
        #mode = "check" for gradient checking
        #       "sanitycheck" to try overfitting 100 datapoints
        #       "search" for searching the best hyperparameters
        #       "default" for default training
        mode = "def"#"sanitycheck"#"default"#"sanitycheck"

        #constants
        # lamda = regularization parameter
        lamda = 5e-4 #0 = noregularization
        # eta = learning rate
        eta = 0.0159
        n_batch = 100
        epochs = 30
        rho = 0.9
        lr_decay = 0.95 #0 = nodecay

        labels = 10

        Xtrain, Ytrain, ytrain, Xval, Yval, yval, Xtest, Ytest, ytest = OpenData(labels, mode)
        Xtrain, Xval, Xtest = ToZeroMean(Xtrain, Xval, Xtest)

        # d = dim of each image
        # N = num of images
        d, N = GetDimensions(Xtrain)

        # nodes in layers
        # nodes[0] -> nodes in input
        # nodes[1] -> nodes in hidden layer
        # nodes[2] -> nodes in next layer
        # nodes = [d, 50, 30, 10]
        nodes = [d, 50, 10]

        # W1 = weights K1 x d
        # b1 = bias K1 x 1
        # W2 = weights K2 x K1
        # b2 = bias K2 x 1
        W, b = InitParams(nodes)


        if mode == "check":
            # check
            CheckGrads(Xtrain, Ytrain, W, b, lamda, 2)
        elif mode == "sanitycheck":
            #try to overfit 100 datapoints
            Xtrain, Xval, Xtest = ToZeroMean(Xtrain, Xval, Xtest)

            Wstar, bstar, allW, allb = MiniBatchGD(Xtrain, Ytrain,
                                                   {"eta": 0.05, "n_batch": 10, "epochs": 200, "lamda": 0,
                                                    "rho": rho, "lr_decay": lr_decay, "W": W, "b": b})
            # plot loss on training and validation dataset
            PlotLoss(Xtrain, Ytrain, Xval, Yval, allW, allb, lamda, eta, mode)
            # calculate accuracy on training dataset
            train_accuracy = ComputeAccuracy(Xtrain, ytrain, Wstar, bstar)
            print("train accuracy:", train_accuracy)

        elif mode == "search":
            #search for best parameters
            Xtrain, Xval, Xtest = ToZeroMean(Xtrain, Xval, Xtest)

            #uniformly log-initialize etas in the range found with coarse search
            #different experiment initializations

            # es = np.random.uniform(-3, -1, 15)
            # ls = np.random.uniform(-7, 3, 10)

            # es = np.random.uniform(-3, -2, 15)
            # ls = np.random.uniform(-8, -2, 10)

            # es = np.random.uniform(0.013, 0.026, 15)
            # es = np.sort(es)
            # ls = np.random.uniform(-6, -1, 10)
            # ls = np.sort(ls)

            # es = [0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021]
            # ls = [-8, -7, -6, -5, -4, -3]

            es = [0.0155, 0.0157, 0.0159, 0.0161, 0.0163, 0.0165]
            ls = [1e-4, 5e-4, 1e-5, 5e-5]
            # es = [0.0155, 0.0156, 0.0157, 0.0158, 0.0159, 0.016, 0.0161, 0.0162, 0.0163, 0.0164, 0.0165]
            # ls = [5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
            # es = [0.0166, 0.0167, 0.0168, 0.0169, 0.017, 0.0171, 0.0172, 0.0173, 0.0174, 0.0175, 0.0176, 0.0177]
            # ls = [5e-5, 7e-5]
            for thee in es:
                # eta = 5 * (10 ** thee)
                eta = thee
                for thel in ls:
                    # lamda = 10 ** thel
                    lamda = thel

                    W, b = InitParams(nodes)

                    Wstar, bstar, allW, allb = MiniBatchGD(Xtrain, Ytrain,
                                                           {"eta": eta, "n_batch":n_batch, "epochs": epochs, "lamda": lamda,
                                                            "rho": rho, "lr_decay": lr_decay, "W": W, "b": b})
                    #calculate accuracy on test dataset
                    valid_accuracy = ComputeAccuracy(Xval, yval, Wstar, bstar)
                    #save in file
                    file = open("thefineresttraincosts.txt", "a")
                    file.write("\n" + "eta: " + str(eta) + "    lamda: " + str(lamda) +
                               "    validation accuracy: " + str(valid_accuracy))
                    file.close()
        else:
            #default training
            Wstar, bstar, allW, allb = MiniBatchGD(Xtrain, Ytrain,
                                                   {"eta": eta, "n_batch": n_batch, "epochs": epochs, "lamda": lamda,
                                                    "rho": rho, "lr_decay": lr_decay, "W": W, "b": b})

            # plot loss on training and validation dataset
            PlotLoss(Xtrain, Ytrain, Xval, Yval, allW, allb, lamda, eta, mode)
            # calculate accuracy on test dataset
            testacc = ComputeAccuracy(Xtest, ytest, Wstar, bstar)
            print("TEST ACCURACY:", testacc)

        print("done")

    except ZeroDivisionError as err:
        print(err.args)




Main()