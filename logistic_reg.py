import numpy as np
import copy
import math

def map_feature(X1, X2, degree=6):
    """
    Feature mapping function to polynomial features
    Calcula el polinomio frontera que separa los 0's y los 1's.
    La funcion de debajo se encarga de graficarla.
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)

def sigmoid(z):
    
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """
    return 1/(1+np.exp(-1*z))


def predictor_f(w,b,x):
    return sigmoid(np.dot(x,w)+b)

def loss_function(w,b,x,y):
    return (-y*np.log(predictor_f(w = w, b = b, x = x))) - (1-y)*np.log(1-predictor_f(w = w, b = b, x = x))

# logistic regression

def compute_cost(X, y, w, b, lambda_=0):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value
      w : (array_like Shape (n,)) Values of parameters of the model
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar) cost
    """
    numSamples = X.shape[0]
    return (1/numSamples)*(np.sum(loss_function(w = w, x = X, y = y, b = b))) + (lambda_/(2*numSamples))*np.sum(np.square(w))
    

def compute_gradient(X, y, w, b, lambda_=0):
    """
    Computes the gradient for logistic regression

    Args:
      X : (ndarray Shape (m,n)) variable such as house size
      y : (array_like Shape (m,1)) actual value
      w : (array_like Shape (n,1)) values of parameters of the model
      b : (scalar)                 value of parameter of the model
      lambda_: unused placeholder
    Returns
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b.
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w.
    """

    numRows = X.shape[0]
    dj_db = np.sum(predictor_f(w = w, b = b,x = X) - y)/numRows
   
    dj_dw = np.dot((predictor_f(w = w, b = b,x = X) - y),X)/numRows + (lambda_*w.T/numRows)

    return dj_db, np.array(dj_dw)


# regularized logistic regression
#
def compute_cost_reg(X, y, w, b, lambda_):
    """
    Computes the cost over all examples
    Args:
      X : (array_like Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : (array_like Shape (n,)) Values of bias parameter of the model
      lambda_ : (scalar, float)    Controls amount of regularization
    Returns:
      total_cost: (scalar)         cost 
    """

    return compute_cost(X = X, y = y, w = w, b = b, lambda_ = lambda_)


def compute_gradient_reg(X, y, w, b, lambda_):
    """
    Computes the gradient for linear regression 

    Args:
      X : (ndarray Shape (m,n))   variable such as house size 
      y : (ndarray Shape (m,))    actual value 
      w : (ndarray Shape (n,))    values of parameters of the model      
      b : (scalar)                value of parameter of the model  
      lambda_ : (scalar,float)    regularization constant
    Returns
      dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """

    return compute_gradient(X = X, y = y, w = w, b = b, lambda_ = lambda_)



# gradient descent
#
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_= 0):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant

    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """

    J_history = []
    w = w_in
    b = b_in

    for i in range(num_iters):
        w = w-alpha*(gradient_function(X = X, y = y, w = w, b = b, lambda_ = lambda_)[1])
        b = b-alpha*gradient_function(X = X, y = y, w = w, b = b, lambda_ = lambda_)[0]
        J_history.append(cost_function(X= X, y = y, w = w, b = b, lambda_ = lambda_))
    return w, b, J_history




# predict
#
def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w and b

    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5

    prob: (ndarray (m,1))
          each value represents the probability of each example of being the class 1
    """
    prediction = predictor_f(w = w, b = b, x = X)

    return np.array(prediction) >= 0.5, np.array(prediction)

