import math
import numpy as np
import sys

sys.path.insert(0, '../Practica3')  # Agregar el directorio padre al sys.path

import logistic_reg as lg # Importar el archivo desde el directorio padre




def cost(theta1, theta2, X, y, lambda_ = 0):
    """
    Compute cost for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    """

    m = X.shape[0]
    a1 = np.hstack([np.ones((m, 1)), X])
    z2 = np.dot(theta1,a1.T)
    a2 = lg.sigmoid(z2)
    a2_1s = np.vstack([np.ones((1,a2.shape[1])), a2])
    z3 = np.dot(theta2,a2_1s)
    h = lg.sigmoid(z3)

    cost = -np.sum(y.T*np.log(h)+(np.ones((y.shape))-y).T*np.log(np.ones((h.shape))-h))/m
    reg = (lambda_/(2*m))*(np.sum(np.square(theta1)) + np.sum(np.square(theta2)))
    
    return cost + reg

def cost_wrapper(weights, X, y, shape1, shape2, lambda_ = 0):
    tam = shape1[0]*shape1[1]
    theta1 = np.array(weights[:tam]).reshape(shape1)
    theta2 = np.array(weights[tam:]).reshape(shape2)
    cost, g1, g2 = backprop(theta1, theta2, X, y, lambda_)
    g = np.concatenate((g1.flatten(), g2.flatten()))
    return cost,  g


def backprop(theta1, theta2, X, y, lambda_):
    """
    Compute cost and gradient for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    grad1 : array_like
        Gradient of the cost function with respect to weights
        for the first layer in the neural network, theta1.
        It has shape (2nd hidden layer size x input size + 1)

    grad2 : array_like
        Gradient of the cost function with respect to weights
        for the second layer in the neural network, theta2.
        It has shape (output layer size x 2nd hidden layer size + 1)

    """
    X = X.T #Cada columna un ejemplo (5) con 3 variables (filas)
    m = X.shape[0]
    a1 = np.vstack([np.ones((1,X.shape[1])), X])
    z2 = np.dot(theta1,a1)
    a2 = lg.sigmoid(z2)
    a2_1s = np.vstack([np.ones((1,a2.shape[1])), a2])
    z3 = np.dot(theta2,a2_1s)
    h = lg.sigmoid(z3)
    a3 = h

    cost = -np.sum(y.T*np.log(h)+(np.ones((y.shape))-y).T*np.log(np.ones((h.shape))-h))/m + (lambda_/(2*m))*(np.sum(np.square(theta1)) + np.sum(np.square(theta2)))
    d3 = a3-y.T
    d2 = np.dot(theta2.T , d3)*(a2_1s*(np.ones(a2_1s.shape)-a2_1s))
    d2 = d2[1:]
    grad1 = (1/m)*(np.dot(d2,a1.T)) 
    grad2 = (1/m)*(np.dot(d3, a2_1s.T)) 

    #Regularizamos todo menos los términos independientes(primera fila)
    grad1[1:] = grad1[1:] + (lambda_/m)*theta1[1:]
    grad2[1:] = grad2[1:] + (lambda_/m)*theta2[1:]


    return cost, grad1, grad2


def gradient_descent(theta1,theta2, X, y, lambda_, alpha, numIters):

    J_history = []

    for i in range(numIters):
        cost, grad1, grad2 = backprop(theta1=theta1, theta2=theta2, X =X, y = y, lambda_ = lambda_)
        #print(cost)
        theta1 = theta1 -alpha*grad1
        theta2 = theta2 - alpha*grad2
        J_history.append(cost)

    return J_history, theta1, theta2