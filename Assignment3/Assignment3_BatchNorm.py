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

def ComputeGradsNum(X, Y, W, b, lamb, epsilon, h, batch_norm_mode):
    #initialize grads
    grad_W = []
    grad_b = []

    for i in range(len(W)):
        Wnew = np.zeros(np.shape(W[i]))
        grad_W.append(Wnew)
        bnew = np.zeros(np.shape(b[i]))
        grad_b.append(bnew)

    c = ComputeCost(X, Y, W, b, lamb, epsilon, batch_norm_mode)

    for k in range(len(W)):
        for i in range(len(b[k])):
            b_try = deepcopy(b)
            b_try[k][i] += h
            c2 = ComputeCost(X, Y, W, b_try, lamb, epsilon, batch_norm_mode)
            grad_b[k][i] = (c2 - c) / h

        for i in range(W[k].shape[0]):
            for j in range(W[k].shape[1]):
                W_try = deepcopy(W)
                W_try[k][i, j] += h
                c2 = ComputeCost(X, Y, W_try, b, lamb, epsilon, batch_norm_mode)
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


def InitParams(nodes, init_mode):
    #same seed every time for testing purposes
    np.random.seed(123)
    W = []
    b = []
    for i, nodesnum in enumerate(nodes):
        if i == 0:
            continue
        if init_mode == "he":
            Wi = np.random.normal(0, (2/nodes[i - 1]), (nodes[i], nodes[i - 1]))
        else:
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

def ComputeCost(X, Y, W, b, lamda, epsilon, batch_norm_mode, **kwargs):
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

    if batch_norm_mode == "no":
        P, h, s1 = EvaluateClassifier(X, W, b)
    else:
        P, scores, bn_scores, bn_relu_scores, mus, vars = ForwardBatchNorm(X, W, b, epsilon, **kwargs)
    cross_entropy_loss = 0 - np.log(np.sum(np.prod((np.array(Y), P), axis=0), axis=0))

    J = (1/N) * np.sum(cross_entropy_loss) + regularization
    return J


def ComputeAccuracy(X, y, W, b, epsilon, batch_norm_mode, **kwargs):
    """
    X: each column of X corresponds to an image and X has size dn.
    y: Y is the vector of ground truth labels of length n.
    W:
    b:
    acc: acc is a scalar value containing the accuracy.
    """
    if batch_norm_mode == "no":
        P, h, s1 = EvaluateClassifier(X, W, b)
    else:
        P, scores, bn_scores, bn_relu_scores, mus, vars = ForwardBatchNorm(X, W, b, epsilon, **kwargs)
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

def ComputeBatchBackGrads(n, g, bn_relu_scores_prev_layer, lamda, W, layer):
    bgrad = (1 / n) * np.sum(g, axis=1)
    bgrad = bgrad.reshape(-1, 1)
    Wgrad = (1 / n) * np.dot(g, bn_relu_scores_prev_layer.T) + 2 * lamda * W[layer]
    return (bgrad, Wgrad)

def PropagateBatchBackGrads(g, W, bn_scores, layer):
    g = np.dot(W[layer].T, g)
    g = np.multiply(g, IndXPositive(bn_scores[layer-1]))
    # did the elementwise multiplication (above)
    # instead of this:
    # for i in range(n):
    #     g[:,i] = np.dot(g[:,i], np.diag(IndXPositive(bn_scores[-2][:,i])))
    return g

def BatchNormBackPass(layer, g, scores, mus, vars, epsilon):
    n = np.shape(g)[1]

    vl = vars[layer]
    mul = mus[layer]
    sl = scores[layer]

    V12 = (vl + epsilon)
    V12 = np.power(V12, (-1 / 2))
    V12 = np.diag(V12)

    V32 = (vl + epsilon)
    V32 = np.power(V32, (-3 / 2))
    V32 = np.diag(V32)


    gradJvar = np.zeros(np.shape(mul))
    for i in range(n):
        gradJvar += np.dot(g[:, i], (np.dot(V32, np.diag(sl[:, i] - mul))))
    gradJvar = -(1/2) * gradJvar

    gradJmu = np.zeros(np.shape(mul))
    for i in range(n):
        gradJmu += np.dot(g[:, i], V12)
    gradJmu = gradJmu * (-1)


    gnew = np.zeros(np.shape(g))
    #for each datapoint
    for i in range(n):
        gnew[:, i] = np.dot(g[:, i], V12) + (2/n) * np.dot(gradJvar, np.diag(sl[:,i]-mul)) + (1/n) * gradJmu

    return gnew

def BackwardBatchNorm(X, Y, P, W, scores, bn_scores, bn_relu_scores, lamda, mus, vars, epsilon):
    n = np.shape(Y)[1]
    layers = len(W)

    bgrads = []
    Wgrads = []

    #last layer
    layer = layers-1
    g = -(Y - P)
    #compute grads
    bgrad, Wgrad = ComputeBatchBackGrads(n, g, bn_relu_scores[layer-1], lamda, W, layer)
    bgrads.insert(0, bgrad)
    Wgrads.insert(0, Wgrad)
    #propagate to prev layer
    g = PropagateBatchBackGrads(g, W, bn_scores, layer)

    #previous layers
    for layer in range(layers - 2, -1, -1):
        g = BatchNormBackPass(layer, g, scores, mus, vars, epsilon)
        # compute grads
        if layer > 0:
            bn_relu_scores_prev_layer = bn_relu_scores[layer-1]
        else:
            bn_relu_scores_prev_layer = X
        bgrad, Wgrad = ComputeBatchBackGrads(n, g, bn_relu_scores_prev_layer, lamda, W, layer)
        bgrads.insert(0, bgrad)
        Wgrads.insert(0, Wgrad)
        # propagate to prev layer
        if layer > 0:
            g = PropagateBatchBackGrads(g, W, bn_scores, layer)

    return (Wgrads, bgrads)


def MuAndVarOfLayer(si):
    # for layer l

    # mu of each node of the layer for all datapoints in batch
    # mu = np.sum(si, axis=1)/datapoint_num | primitive way
    mu = np.mean(si, axis=1)

    # variance of each node of the layer for all datapoints in batch
    # datapoint_num = np.shape(si)[1]
    # sigma2 = 0                         |
    # for i in range(datapoint_num):     | simple way
    #     sigma2 += (si[:, i] - mu) ** 2 |
    # sigma2 /= datapoint_num            |
    # sigma2 = ( np.sum(np.power((si - newmu), 2), axis=1) )/ datapoint_num | clever but primitive
    sigma2 = np.var(si, axis=1)  # | as it should

    return mu, sigma2

def BatchNormalize(sl, mu, var, epsilon):
    slnorm = np.zeros(np.shape(sl))

    a = (var + epsilon)
    a = np.power(a,(-1/2))
    a = np.diag(a)

    #for each datapoint in sl
    for i in range(np.shape(sl)[1]):
        slnorm[:, i] = np.dot(a, (sl[:, i]-mu))

    return slnorm

def ForwardBatchNorm(X, W, b, epsilon, **kwargs):
    scores = [] #unnormalized scores
    bn_scores = [] #normalized scores
    bn_relu_scores = [X] #normalized scores after ReLU
    mus = [] #means of layers
    vars = [] #vars of layers

    if kwargs != {}:
        mus = kwargs["movav_mean"]
        vars = kwargs["movav_var"]

    #for each layer except last:
    for l in range(len(W)):
        sl = np.dot(W[l], bn_relu_scores[-1]) + b[l]
        scores.append(sl)

        if kwargs != {}:
            mu = mus[l]
            var = vars[l]
        else:
            mu, var = MuAndVarOfLayer(sl)
            mus.append(mu)
            vars.append(var)

        slnorm = BatchNormalize(sl, mu, var, epsilon)
        bn_scores.append(slnorm)
        slrelunorm = ReLU(slnorm)
        bn_relu_scores.append(slrelunorm)

    #for last layer (last computed sl):
    P = SoftMax(sl)

    # detach X from bn_relu_scores list (was appended just to help generalisation)
    bn_relu_scores.pop(0)

    return P, scores, bn_scores, bn_relu_scores, mus, vars

def ComputeGradientsBatchNorm(X, Y, W, b, lamda, epsilon):
    """
    :param X: each column of X corresponds to an image and it has size d xn
    :param Y: each column of Y (Kx n) is the one-hot ground truth label for the corresponding column of X
    :param W:
    :param lamda:
    :return: grad_W is the gradient matrix of the cost J relative to W and has size K xd - same as W
             grad_b is the gradient vector of the cost J relative to b and has size K x1 - same as b
    """


    # each column of P contains the probability for each label for the image in the corresponding column of X.
    # P has size Kx n
    # h is the hidden layer output of the network during the forward pass
    # h has size K1 x N

    P, scores, bn_scores, bn_relu_scores, mus, vars = ForwardBatchNorm(X, W, b, epsilon)
    Wgrads, bgrads = BackwardBatchNorm(X, Y, P, W, scores, bn_scores, bn_relu_scores, lamda, np.copy(mus), np.copy(vars), epsilon)


    return Wgrads, bgrads, mus, vars

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

def CheckGrads(X, Y, W, b, lamda, epsilon, how_many, batch_norm_mode):
    randomdatapoints = [11,12]#random.sample(range(0, np.shape(X)[1]), how_many)

    X = X[10:15, randomdatapoints]
    Y = Y[:, randomdatapoints]
    W[0] = W[0][:, 10:15]

    if batch_norm_mode == "no":
        gW, gb = ComputeGradients(X, Y, W, b, lamda)
    else:
        gW, gb, mus, vars = ComputeGradientsBatchNorm(X, Y, W, b, lamda, epsilon)
    gWnumSl, gbnumSl = ComputeGradsNum(X, Y, W, b, lamda, epsilon, 1e-5, batch_norm_mode)
    for l in range(len(gW)):
        CompareGrads(gWnumSl[l], gbnumSl[l], gW[l], gb[l])


def UpdateMovAv(movav_mean, movav_var, mus, vars, alpha, layers):
    for l in range(layers):
        movav_mean[l] = alpha * movav_mean[l] + (1 - alpha) * mus[l]
        movav_var[l] = alpha * movav_var[l] + (1 - alpha) * vars[l]
    return movav_mean, movav_var

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
    epsilon = GDparams["epsilon"]

    W = GDparams["W"]
    b = GDparams["b"]

    batch_norm_mode = GDparams["batch_norm_mode"]

    nodes = GDparams["nodes"]
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

    #initializing moving averages (used in batch normalization)
    movav_mean = []
    movav_var = []
    alpha = 0 # only for the first loop. then I change it to 0.99
    for n in range(1, len(nodes)):
        movav_mean.append(np.zeros(nodes[n]))
        movav_var.append(np.zeros(nodes[n]))

    for ep in range(n_epochs):
        for st in range(steps_per_epoch):
            batch_start = st * n_batch
            batch_end = batch_start + n_batch
            Xbatch = X[:, batch_start:batch_end]
            Ybatch = Y[:, batch_start:batch_end]

            if batch_norm_mode == "no":
                grad_W, grad_b = ComputeGradients(Xbatch, Ybatch, W, b, lamda)
            else:
                grad_W, grad_b, mus, vars = ComputeGradientsBatchNorm(Xbatch, Ybatch, W, b, lamda, epsilon)
                movav_mean, movav_var = UpdateMovAv(movav_mean, movav_var, mus, vars, alpha, layers)
                alpha = 0.99 #in order for the moving average to be updated correctly after the first loop when it is initialized with a = 0

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

        print("epoch "+str(ep)+" continuing...")

    return (W, b, allW, allb, movav_mean, movav_var)

def PlotLoss(Xtrain, Ytrain, Xval, Yval, allW, allb, lamda, eta, epsilon, mode, batch_norm_mode, **kwargs):
    train_cost = []
    val_cost = []

    layers = len(allW)

    #calculate costs
    for i in range(len(allW[0])):
        W = [allW[l][i] for l in range(layers)]
        b = [allb[l][i] for l in range(layers)]
        # W = [allW[0][i], allW[1][i]]
        # b = [allb[0][i], allb[1][i]]

        cost = ComputeCost(Xtrain, Ytrain, W, b, lamda, epsilon, batch_norm_mode, **kwargs)
        train_cost.append(cost)
        cost = ComputeCost(Xval, Yval, W, b, lamda, epsilon, batch_norm_mode, **kwargs)
        val_cost.append(cost)

    #plot
    idx = np.arange(len(train_cost))
    plt.plot(idx, train_cost, label="Training")
    plt.plot(idx, val_cost, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss development over epochs, using eta " + str(eta) + " and lambda " + str(lamda), y=1.03)
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
    if mode == "search" or mode == "check" or mode == "sanitycheck" or mode == "default1":
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
        #       "default1" for default training but using only one batch
        mode = "search"#"default1"#"sanitycheck"

        # "yes" - > implement batch normalization
        # "no"  - > do not implement
        batch_norm_mode = "no"

        # "default" -> default weight initialization
        # "he"      -> He weight initialization
        init_mode = "he"

        #constants
        # lamda = regularization parameter
        lamda = 5e-4#1e-6 #0 = noregularization
        # eta = learning rate
        eta = 0.0159
        n_batch = 100
        epochs = 30
        rho = 0.9
        lr_decay = 0.95 #0 = nodecay
        epsilon = 1e-16 #small constant to prevent divisions by zerp

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
        nodes = [d, 50, 30, 10]
        # nodes = [d, 50, 10]

        # W1 = weights K1 x d
        # b1 = bias K1 x 1
        # W2 = weights K2 x K1
        # b2 = bias K2 x 1
        W, b = InitParams(nodes, init_mode)


        if mode == "check":
            # check
            CheckGrads(Xtrain, Ytrain, W, b, lamda, epsilon, 2, batch_norm_mode)
        elif mode == "sanitycheck":
            #try to overfit 100 datapoints
            Xtrain, Xval, Xtest = ToZeroMean(Xtrain, Xval, Xtest)

            Wstar, bstar, allW, allb, movav_mean, movav_var = MiniBatchGD(Xtrain, Ytrain,
                                                            {"eta": eta, "n_batch": 10, "epochs": 200, "lamda": lamda,
                                                            "rho": rho, "lr_decay": lr_decay, "W": W, "b": b,
                                                            "epsilon": epsilon, "nodes": nodes,
                                                            "batch_norm_mode": batch_norm_mode})
            # plot loss on training and validation dataset
            PlotLoss(Xtrain, Ytrain, Xval, Yval, allW, allb, lamda, eta, epsilon, mode, batch_norm_mode,
                     **{"movav_mean": movav_mean, "movav_var": movav_var})
            # calculate accuracy on training dataset
            train_accuracy = ComputeAccuracy(Xtrain, ytrain, Wstar, bstar, epsilon, batch_norm_mode,
                                            **{"movav_mean":movav_mean, "movav_var": movav_var})
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
            es = [0.05, 0.1, 0.5]
            ls = [5e-4]

            # es = [0.0155, 0.0157, 0.0159, 0.0161, 0.0163, 0.0165]
            # ls = [1e-4, 5e-4, 1e-5, 5e-5]
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

                    W, b = InitParams(nodes, init_mode)

                    Wstar, bstar, allW, allb, movav_mean, movav_var = MiniBatchGD(Xtrain, Ytrain,
                                                                    {"eta": eta, "n_batch":n_batch, "epochs": epochs,
                                                                     "lamda": lamda, "rho": rho, "lr_decay": lr_decay,
                                                                     "W": W, "b": b, "epsilon": epsilon, "nodes": nodes,
                                                                     "batch_norm_mode": batch_norm_mode})

                    PlotLoss(Xtrain, Ytrain, Xval, Yval, allW, allb, lamda, eta, epsilon, mode, batch_norm_mode,
                            **{"movav_mean": movav_mean, "movav_var": movav_var})

                    #calculate accuracy on test dataset
                    valid_accuracy = ComputeAccuracy(Xval, yval, Wstar, bstar, epsilon, batch_norm_mode,
                                                     **{"movav_mean":movav_mean, "movav_var": movav_var})
                    #save in file
                    file = open("coarsecosts.txt", "a")
                    file.write("\n" + "eta: " + str(eta) + "    lamda: " + str(lamda) +
                               "    validation accuracy: " + str(valid_accuracy))
                    file.close()
        else:
            #default training
            Wstar, bstar, allW, allb, movav_mean, movav_var = MiniBatchGD(Xtrain, Ytrain,
                                                            {"eta": eta, "n_batch": n_batch, "epochs": epochs,
                                                             "lamda": lamda, "rho": rho, "lr_decay": lr_decay,
                                                             "W": W, "b": b, "epsilon": epsilon, "nodes": nodes,
                                                             "batch_norm_mode": batch_norm_mode})

            # plot loss on training and validation dataset
            PlotLoss(Xtrain, Ytrain, Xval, Yval, allW, allb, lamda, eta, epsilon, mode, batch_norm_mode,
                     **{"movav_mean":movav_mean, "movav_var": movav_var})
            # calculate accuracy on test dataset
            testacc = ComputeAccuracy(Xtest, ytest, Wstar, bstar, epsilon, batch_norm_mode,
                                      **{"movav_mean":movav_mean, "movav_var": movav_var})
            print("TEST ACCURACY:", testacc)

        print("done")

    except ZeroDivisionError as err:
        print(err.args)




Main()