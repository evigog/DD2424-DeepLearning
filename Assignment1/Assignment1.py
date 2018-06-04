import matplotlib.pyplot as plt
import numpy as np

class NonInteger(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors

def ComputeGradsNumSlow(X, Y, W, b, lamda, h):
    grad_W = np.zeros(np.shape(W))
    grad_b = np.zeros(np.shape(b))
    for i in range(np.size(b)):
        b_try = b
        b_try[i] = b_try[i] - h
        c1 = ComputeCost(X, Y, W, b_try, lamda)
        b_try = b
        b_try[i] = b_try[i] + h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = np.divide((c2-c1),(h))

    for i in range(np.shape(W)[0]):
        for j in range(np.shape(W)[1]):
            W_try = W
            W_try[i,j] = W_try[i,j] - h
            c1 = ComputeCost(X, Y, W_try, b, lamda)
            W_try = W
            W_try[i,j] = W_try[i,j] + h
            c2 = ComputeCost(X, Y, W_try, b, lamda)
            grad_W[i,j] = np.divide((c2-c1),(h))
    return (grad_W, grad_b)

# def ComputeGradsNum(X, Y, W, b, lamda, h):
#     grad_W = np.zeros(np.shape(W))
#     grad_b = np.zeros(np.shape(b))
#
#     c = ComputeCost(X, Y, W, b, lamda)
#
#     for i in range(np.size(b)):
#         b_try = b
#         b_try[i] = b_try[i] - h
#         c1 = ComputeCost(X, Y, W, b_try, lamda)
#         grad_b[i] = np.divide((c-c1),2*h)
#
#     for i in range(np.shape(W)[0]):
#         for j in range(np.shape(W)[1]):
#             W_try = W
#             W_try[i] = W_try[i] - h
#             c1 = ComputeCost(X, Y, W_try, b, lamda)
#             grad_W[i] = np.divide((c-c1),2*h)
#
#     return(grad_W, grad_b)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def LoadBatch(filename, numoflabels):
    filename = "cifar-10-batches-py/" + filename
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

def GetDimensions(X):
    #dim of each image
    d = np.shape(X)[0]
    #num of images
    N = np.shape(X)[1]

    return (d, N)


def InitParams(K, d):
    #make seed the same every time for testing purposes
    np.random.seed(0)
    b = np.random.normal(0, 0.01, (K,1))
    W = np.random.normal(0, 0.01, (K,d))
    return (b,W)

def SoftMax(z):
    """
    returns softmax for each column of the input matrix
    """
    e = np.exp(z - np.max(z))
    return((e / e.sum(axis = 0)))


def EvaluateClassifier(X, W, b):
    """
    the output probabilities of the classifier
    """
    s = np.matmul(W, X) + b
    P = SoftMax(s)

    return P

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

    P = EvaluateClassifier(X, W, b)
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
    P = EvaluateClassifier(X, W, b)
    predicitons = np.argmax(P, axis=0)
    correct = np.count_nonzero((y-predicitons) == 0)#because of the condition it actually counts the zeros
    all = np.size(predicitons)
    if all == 0:
        raise(ZeroDivisionError("division by zero in ComputeAccuracy"))
    acc = correct/all

    return acc

def ComputeGradients(X, Y, W, b, lamda):
    """
    :param X: each column of X corresponds to an image and it has size d xn
    :param Y: each column of Y (Kx n) is the one-hot ground truth label for the corresponding column of X
    :param W:
    :param lamda:
    :return: grad_W is the gradient matrix of the cost J relative to W and has size K xd - same as W
             grad_b is the gradient vector of the cost J relative to b and has size K x1 - same as b
    """

    grad_W = np.zeros(np.shape(W))
    grad_b = np.zeros(np.shape(b))

    # each column of P contains the probability for each label for the image in the corresponding column of X.
    # P has size Kx n
    P = EvaluateClassifier(X, W, b)

    N = np.shape(X)[1]
    for i in range(N):
        g = - (Y[:, i] - P[:, i]).reshape(-1, 1)
        grad_b += g
        grad_W += np.matmul(g, X[:,i].reshape(1, -1))

    grad_b = grad_b/N
    grad_W = grad_W/N + 2 * lamda * W

    return (grad_W, grad_b)

def CheckGrads(WA, bA, WB, bB):
    print("********\nW\n********")
    for i in range(np.shape(WA)[0]):
        for j in range(np.shape(WA)[1]):
            diff = abs(WA[i, j] - WB[i, j]) / max(abs(WA[i, j]), abs(WB[i, j]))
            if (diff > 1e-6):
                print(i, j, WA[i, j], WB[i, j], diff)

    print("********\nb\n********")
    for i in range(np.size(bA)):
        diff = abs(bA[i] - bB[i]) / max(abs(bA[i]), abs(bB[i]))
        if (diff > 1e-6):
            print(i, bA[i], bB[i], diff)

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

    for ep in range(n_epochs):
        for st in range(steps_per_epoch):
            batch_start = st * n_batch
            batch_end = batch_start + n_batch
            Xbatch = X[:, batch_start:batch_end]
            Ybatch = Y[:, batch_start:batch_end]
            grad_W, grad_b = ComputeGradients(Xbatch, Ybatch, W, b, lamda)
            W = W - eta * grad_W
            b = b - eta * grad_b

        print(ComputeCost(X, Y, W, b, lamda))


    return (W, b)

def Main():
    try:

        # K =num of labels
        K = 10

        # N = num of input examples
        # d = dimension of each input example
        # X = images (d x N)
        # Y = one-hot labels (K x N)
        # y = labels (N)
        Xtrain,Ytrain,ytrain = LoadBatch("data_batch_1", K)
        Xval, Yval, yval = LoadBatch("data_batch_2", K)
        Xtest, Ytest, ytest = LoadBatch("test_batch", K)

        # d = dim of each image
        # N = num of images
        d, N = GetDimensions(Xtrain)

        # W = weights K x d
        # b = bias K x 1
        b, W = InitParams(K, d)

        # lamda = regularization parameter
        lamda = 0#.3

        acc = ComputeCost(Xtrain, Ytrain, W, b, lamda)
        print(acc)


        #check
        # gW, gb = ComputeGradients(X, Y, W, b, lamda)
        # gWnumSl, gbnumSl = ComputeGradsNumSlow(X, Y, W, b, lamda, 1e-6)
        # CheckGrads(gWnumSl, gbnumSl, gW, gb)

        MiniBatchGD(Xtrain, Ytrain, {"eta": 0.1, "n_batch":100, "epochs": 20}, W, b, lamda)

        print("done")

    except ZeroDivisionError as err:

        print(err.args)




Main()