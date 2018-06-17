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

def ComputeGradsNumSlow(X, Y, W1, b1, W2, b2, lamb, h):
    W = [W1, W2]
    b = [b1, b2]

    #initialize grads
    grad_W = []
    grad_b = []

    for i in range(len(W)):
        Wnew = np.zeros(np.shape(W[i]))
        grad_W.append(Wnew)
        bnew = np.zeros(np.shape(b[i]))
        grad_b.append(bnew)

    c = ComputeCost(X, Y, W[0], b[0], W[1], b[1], lamb)

    for k in range(len(W)):
        for i in range(len(b[k])):
            b_try = deepcopy(b)
            b_try[k][i] += h
            c2 = ComputeCost(X, Y, W[0], b_try[0], W[1], b_try[1], lamb)
            grad_b[k][i] = (c2 - c) / h

        for i in range(W[k].shape[0]):
            for j in range(W[k].shape[1]):
                W_try = deepcopy(W)
                W_try[k][i, j] += h
                c2 = ComputeCost(X, Y, W_try[0], b[0], W_try[1], b[1], lamb)
                grad_W[k][i, j] = (c2 - c) / h

    return grad_W[0], grad_b[0], grad_W[1], grad_b[1]

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


def InitParams(K1, K2, d):
    #same seed every time for testing purposes
    # np.random.seed(123)
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
    s = np.maximum(s, 0)
    return s

def EvaluateClassifier(X, W1, b1, W2, b2):
    """
    apply forward pass and return
    the output probabilities of the classifier
    """
    s1 = np.dot(W1, X) + b1
    h = ReLU(s1)
    s2 = np.dot(W2, h) + b2
    P = SoftMax(s2)

    return P, h, s1

def ComputeCost(X, Y, W1, b1, W2, b2, lamda):
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
    regularization = lamda * (np.sum(np.power(W1, 2)) + np.sum(np.power(W2, 2)))

    P, h, s1 = EvaluateClassifier(X, W1, b1, W2, b2)
    cross_entropy_loss = 0 - np.log(np.sum(np.prod((np.array(Y), P), axis=0), axis=0))

    J = (1/N) * np.sum(cross_entropy_loss) + regularization
    return J


def ComputeAccuracy(X, y, W1, b1, W2, b2):
    """
    X: each column of X corresponds to an image and X has size dn.
    y: Y is the vector of ground truth labels of length n.
    W:
    b:
    acc: acc is a scalar value containing the accuracy.
    """
    P, h, s1 = EvaluateClassifier(X, W1, b1, W2, b2)
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
        Xi = X[:, i].reshape((-1, 1))
        hi = h[:, i].reshape((-1, 1))
        si = s1[:, i]

        g = Pi - Yi
        grad_b2 = grad_b2 + g
        grad_W2 = grad_W2 + np.dot(g, np.transpose(hi))

        #propagate error backwards
        g = np.dot(np.transpose(W2), g)
        g = np.dot(np.diag(IndXPositive(si)), g)

        grad_b1 = grad_b1 + g
        grad_W1 = grad_W1 + np.dot(g, np.transpose(Xi)) #???? -> gia to matmul kai to transpose


    grad_b1 = np.divide(grad_b1, N)
    grad_W1 = np.divide(grad_W1, N) + 2 * lamda * W1

    grad_b2 = np.divide(grad_b2, N)
    grad_W2 = np.divide(grad_W2, N) + 2 * lamda * W2

    return (grad_W1, grad_b1, grad_W2, grad_b2)

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

def CheckGrads(X, Y, W1, b1, W2, b2, lamda, how_many):
    randomdatapoints = random.sample(range(0, np.shape(X)[1]), how_many)

    X = X[100:300, randomdatapoints]
    Y = Y[:, randomdatapoints]
    W1 = W1[:, 100:300]

    gW1, gb1, gW2, gb2 = ComputeGradients(X, Y, W1, b1, W2, b2, lamda)
    gWnumSl1, gbnumSl1, gWnumSl2, gbnumSl2 = ComputeGradsNumSlow(X, Y, W1, b1, W2, b2, lamda, 1e-5)
    CompareGrads(gWnumSl1, gbnumSl1, gW1, gb1)
    CompareGrads(gWnumSl2, gbnumSl2, gW2, gb2)


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

    W1 = GDparams["W1"]
    b1 = GDparams["b1"]
    W2 = GDparams["W2"]
    b2 = GDparams["b2"]

    #all datapoints
    N = np.shape(X)[1]

    if N % n_batch != 0:
        raise (NonInteger("non integer number of datapoints per step",1))
    steps_per_epoch = int(N / n_batch)

    allW1 = [W1]
    allb1 = [b1]
    allW2 = [W2]
    allb2 = [b2]

    momW1 = np.zeros(np.shape(W1))
    momW2 = np.zeros(np.shape(W2))
    momb1 = np.zeros(np.shape(b1))
    momb2 = np.zeros(np.shape(b2))

    for ep in range(n_epochs):
        for st in range(steps_per_epoch):
            batch_start = st * n_batch
            batch_end = batch_start + n_batch
            Xbatch = X[:, batch_start:batch_end]
            Ybatch = Y[:, batch_start:batch_end]
            grad_W1, grad_b1, grad_W2, grad_b2 = ComputeGradients(Xbatch, Ybatch, W1, b1, W2, b2, lamda)

            #applying momentum to the update
            momW1 = rho * momW1 + eta * grad_W1
            W1 = W1 - momW1

            momb1 = rho * momb1 + eta * grad_b1
            b1 = b1 - momb1

            momW2 = rho * momW2 + eta * grad_W2
            W2 = W2 - momW2

            momb2 = rho * momb2 + eta * grad_b2
            b2 = b2 - eta * grad_b2

        eta = lr_decay * eta

        allW1.append(W1)
        allb1.append(b1)
        allW2.append(W2)
        allb2.append(b2)

        print("continuing...")


    return (W1, b1, W2, b2, allW1, allb1, allW2, allb2)

def PlotLoss(Xtrain, Ytrain, Xval, Yval, allW1, allb1, allW2, allb2, lamda, eta):
    train_cost = []
    val_cost = []

    #calculate costs
    for i, W1 in enumerate(allW1):
        b1 = allb1[i]
        W2 = allW2[i]
        b2 = allb2[i]

        cost = ComputeCost(Xtrain, Ytrain, W1, b1, W2, b2, lamda)
        train_cost.append(cost)
        cost = ComputeCost(Xval, Yval, W1, b1, W2, b2, lamda)
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
    filename = "eta" + str(eta) + "lamda" + str(lamda) + ".png"
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

def Main():
    try:
        #mode = "check" for gradient checking
        #       "sanitycheck" to try overfitting 100 datapoints
        #       "search" for searching the best hyperparameters
        #       "default" for default training
        mode = "default"#"sanitycheck"

        #constants
        # lamda = regularization parameter
        lamda = 5e-4
        # eta = learning rate
        eta = 0.0159
        n_batch = 100
        epochs = 30
        rho = 0.9
        lr_decay = 0.95

        # K =num of labels
        K1 = 50
        K2 = 10

        X, Y, y = OpenAllData(K2)

        Xtrain = X[:, 0:-1000]
        Ytrain = Y[:, 0:-1000]
        ytrain = y[0:-1000]

        Xval = X[:, -1000:]
        Yval = Y[:, -1000:]
        yval = y[-1000:]

        Xtest, Ytest, ytest = LoadBatch("test_batch", K2)

        # d = dim of each image
        # N = num of images
        d, N = GetDimensions(Xtrain)

        # W1 = weights K1 x d
        # b1 = bias K1 x 1
        # W2 = weights K2 x K1
        # b2 = bias K2 x 1
        Xtrain, Xval, Xtest = ToZeroMean(Xtrain, Xval, Xtest)


        if mode == "check":
            # check
            W1, W2, b1, b2 = InitParams(K1, K2, d)
            CheckGrads(Xtrain, Ytrain, W1, b1, W2, b2, lamda, 4)
        elif mode == "sanitycheck":
            #try to overfit 100 datapoints

            Xtrain, Ytrain, ytrain = LoadBatch("data_batch_1", K2)
            Xval, Yval, yval = LoadBatch("data_batch_2", K2)
            Xtest, Ytest, ytest = LoadBatch("test_batch", K2)

            Xtrain, Ytrain, ytrain = Xtrain[:, :100], Ytrain[:, :100], ytrain[:100]
            Xval, Yval, yval = Xval[:, :100], Yval[:, :100], yval[:100]
            Xtest, Ytest, ytest = Xtest[:, :100], Ytest[:, :100], ytest[:100]

            d, N = GetDimensions(Xtrain)

            Xtrain, Xval, Xtest = ToZeroMean(Xtrain, Xval, Xtest)

            W1, W2, b1, b2 = InitParams(K1, K2, d)

            W1star, b1star, W2star, b2star, allW1, allb1, allW2, allb2 = MiniBatchGD(Xtrain, Ytrain,
                                                                                     {"eta": 0.05, "n_batch": 10,
                                                                                      "epochs": 200, "lamda": 0,
                                                                                      "rho": rho, "lr_decay": lr_decay,
                                                                                      "W1": W1, "b1": b1, "W2": W2,
                                                                                      "b2": b2})
            # plot loss on training and validation dataset
            PlotLoss(Xtrain, Ytrain, Xval, Yval, allW1, allb1, allW2, allb2, lamda, eta)
            # calculate accuracy on training dataset
            train_accuracy = ComputeAccuracy(Xtrain, ytrain, W1star, b1star, W2star, b2star)
            print("train accuracy:", train_accuracy)

        elif mode == "search":
            #search for best parameters

            Xtrain, Ytrain, ytrain = LoadBatch("data_batch_1", K2)
            Xval, Yval, yval = LoadBatch("data_batch_2", K2)
            Xtest, Ytest, ytest = LoadBatch("test_batch", K2)

            d, N = GetDimensions(Xtrain)

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

                    W1, W2, b1, b2 = InitParams(K1, K2, d)

                    W1star, b1star, W2star, b2star, allW1, allb1, allW2, allb2 = MiniBatchGD(Xtrain, Ytrain,
                                                                                             {"eta": eta, "n_batch":n_batch,
                                                                                              "epochs": epochs, "lamda": lamda,
                                                                                              "rho": rho, "lr_decay": lr_decay,
                                                                                              "W1": W1, "b1": b1, "W2": W2,
                                                                                              "b2": b2})
                    #calculate accuracy on test dataset
                    valid_accuracy = ComputeAccuracy(Xval, yval, W1star, b1star, W2star, b2star)
                    #save in file
                    file = open("thefineresttraincosts.txt", "a")
                    file.write("\n" + "eta: " + str(eta) + "    lamda: " + str(lamda) +
                               "    validation accuracy: " + str(valid_accuracy))
                    file.close()
        else:
            #default training
            W1, W2, b1, b2 = InitParams(K1, K2, d)

            W1star, b1star, W2star, b2star, allW1, allb1, allW2, allb2 = MiniBatchGD(Xtrain, Ytrain,
                                                                                     {"eta": eta,
                                                                                      "n_batch": n_batch,
                                                                                      "epochs": epochs,
                                                                                      "lamda": lamda,
                                                                                      "rho": rho,
                                                                                      "lr_decay": lr_decay,
                                                                                      "W1": W1, "b1": b1, "W2": W2,
                                                                                      "b2": b2})
            # plot loss on training and validation dataset
            PlotLoss(Xtrain, Ytrain, Xval, Yval, allW1, allb1, allW2, allb2, lamda, eta)
            # calculate accuracy on test dataset
            testacc = ComputeAccuracy(Xtest, ytest, W1star, b1star, W2star, b2star)
            print("TEST ACCURACY:", testacc)

        print("done")

    except ZeroDivisionError as err:
        print(err.args)




Main()