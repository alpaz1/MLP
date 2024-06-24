import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import nn
import sys
sys.path.insert(0, '../Practica4')  # Agregar el directorio padre al sys.path
import multi_class as mc
from scipy.optimize import minimize



# data display
#
def displayData(X):
    num_plots = int(np.size(X, 0)**.5)
    fig, ax = plt.subplots(num_plots, num_plots, sharex=True, sharey=True)
    plt.subplots_adjust(left=0, wspace=0, hspace=0)
    img_num = 0
    for i in range(num_plots):
        for j in range(num_plots):
            # Convert column vector into 20x20 pixel matrix
            # transpose
            img = X[img_num, :].reshape(20, 20).T
            ax[i][j].imshow(img, cmap='Greys')
            ax[i][j].set_axis_off()
            img_num += 1

    return (fig, ax)


def displayImage(im):
    fig2, ax2 = plt.subplots()
    image = im.reshape(20, 20).T
    ax2.imshow(image, cmap='gray')
    return (fig2, ax2)


# gradient checking
#
def debugInitializeWeights(fan_in, fan_out):
    """
    Initializes the weights of a layer with fan_in incoming connections and
    fan_out outgoing connections using a fixed set of values.
    """

    W = np.sin(np.arange(1, 1 + (1+fan_in)*fan_out))/10.0
    W = W.reshape(fan_out, 1+fan_in, order='F')
    return W


def computeNumericalGradient(J, Theta1, Theta2):
    """
    Computes the gradient of J around theta using finite differences and
    yields a numerical estimate of the gradient.
    """

    theta = np.append(Theta1, Theta2).reshape(-1)

    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    tol = 1e-4

    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = tol
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)

        # Compute numerical gradient
        numgrad[p] = (loss2 - loss1) / (2 * tol)
        perturb[p] = 0

    return numgrad


def checkNNGradients(costNN, reg_param=0):
    """
    Creates a small neural network to check the backpropagation gradients. It will output the
    analytical gradients produced by your backprop code and the numerical gradients
    (computed using computeNumericalGradient). These two gradient computations should result in
    very similar values.
    Parameters
    ----------
    nnCostFunction : func
        A reference to the cost function implemented by the student.
    reg_param : float (optional)
        The regularization parameter value.
    """

    # Set up small NN
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # Generate some random test data
    Theta1 = debugInitializeWeights(input_layer_size, hidden_layer_size)
    Theta2 = debugInitializeWeights(hidden_layer_size, num_labels)

    # Reusing debugInitializeWeights to get random X
    X = debugInitializeWeights(input_layer_size - 1, m)

    # Set each element of y to be in [0,num_labels]
    y = [(i % num_labels) for i in range(m)]

    ys = np.zeros((m, num_labels))
    for i in range(m):
        ys[i, y[i]] = 1

    # Unroll parameters
    # nn_params = np.append(Theta1, Theta2).reshape(-1)

    # Compute Gradient
    cost, grad1, grad2 = costNN(Theta1, Theta2,
                                X, ys, reg_param)
    grad = np.concatenate((np.ravel(grad1), np.ravel(grad2)))

    def reduced_cost_func(p):
        """ Cheaply decorated nnCostFunction """
        Theta1 = np.reshape(
            p[:hidden_layer_size * (input_layer_size + 1)],
            (hidden_layer_size, (input_layer_size + 1)))
        Theta2 = np.reshape(
            p[hidden_layer_size * (input_layer_size + 1):],
            (num_labels, (hidden_layer_size + 1)))
        return costNN(Theta1, Theta2,
                      X, ys, reg_param)[0]

    numgrad = computeNumericalGradient(reduced_cost_func, Theta1, Theta2)

    # Check two gradients
    # np.testing.assert_almost_equal(grad, numgrad)

    # Evaluate the norm of the difference between two the solutions. If you have a correct
    # implementation, and assuming you used e = 0.0001 in computeNumericalGradient, then diff
    # should be less than 1e-9.
    diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)

    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          'Relative Difference: %g' % diff)


def main():
    data = loadmat('./data/ex3data1.mat', squeeze_me=True)
    y = data['y']
    X = data['X']
    matrix = np.diag(np.full(10,1))
    y_vectors = np.array([matrix[y[i]] for i in range(X.shape[0])])

    weights = loadmat('./data/ex3weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']

    #print(nn.cost(theta1, theta2, X, y_vectors, 0))
    checkNNGradients(nn.backprop, reg_param=0)


    #Red con 400 neuronas en la capa input, 25 en la oculta y 10 en la de salida
    eps_init = 0.12
    num_iters = 1000
    lambda_ = 1
    alpha = 1
    theta1_shape = (25,401)
    theta2_shape = (10,26)
    theta1 = np.random.uniform(low=-eps_init, high=eps_init, size=theta1_shape)
    theta2 = np.random.uniform(low=-eps_init, high=eps_init, size=theta2_shape)


    costes, theta1, theta2 = nn.gradient_descent(theta1=theta1, theta2=theta2, X = X, y = y_vectors, lambda_=lambda_, alpha=alpha, numIters=num_iters)

    plt.figure()
    plt.plot(range(num_iters), costes, color='red', label="Evolución de la funcion de coste en cada iteracion Reg")
    plt.title("Coste final = {}, alpha = {}".format(costes[num_iters-1], alpha))
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.savefig('{}.png'.format("./Plots/Evolución de la funcion de coste en cada iteracion "))
    print("Se ha guardado la imagen del grafico con el nombre {}.png".format("Evolución de la funcion de coste en cada iteracion Reg"))
    preds = mc.predict(theta1, theta2, X)
    index = np.argmax(preds, axis = 1)
    accuracy = np.count_nonzero(index[0] == y)*100/X.shape[0]
    print("La precision del modelo es {}%".format(accuracy))


    lambda_ = 1
    num_iters  = 100
    #Para indicar el numero maximo de iteraciones del metodo minimize


    params = np.concatenate((theta1.flatten(), theta2.flatten()))
    
    result = minimize(fun = nn.cost_wrapper, x0 = params, args=( X, y_vectors, theta1.shape, theta2.shape, lambda_), method='TNC', options = {'maxiter': num_iters}, jac = True)

    # Obtenemos los pesos como un array 
    optimized_params = result.x
  
    # Dividimos el array en dos matrices de las dimensiones especificadas
    lim = theta1.shape[0]*theta1.shape[1]
    theta1 = np.array(optimized_params[:lim]).reshape(theta1.shape)
    theta2 = np.array(optimized_params[lim:]).reshape(theta2.shape)

    preds = mc.predict(theta1, theta2, X)
    index = np.argmax(preds, axis = 1)
    accuracy = np.count_nonzero(index[0] == y)*100/X.shape[0]
    print("La precision del modelo es {}%".format(accuracy))

main()