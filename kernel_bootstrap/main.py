from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return (1 + np.outer(x_i, x_j))**d


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return np.exp(-gamma * np.subtract.outer(x_i,x_j)**2)


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    temp = np.eye(x.shape[0])
    k = kernel_function(x, x, kernel_param)
    return np.linalg.solve(k + _lambda * temp, y)


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    fold_size = len(x) // num_folds

    error = 0
    for i in range(num_folds):
        start = fold_size * i
        end = fold_size * (1+i)
        _x = x[start: end]
        x_train = np.delete(x, np.arange(start=start, stop=end))
        _y = y[start: end]
        y_train = np.delete(y, np.arange(start=start, stop=end))

        alpha = train(x_train, y_train, kernel_function, kernel_param, _lambda)
        k = kernel_function(x_train, _x, kernel_param)
        y_hat = np.sum((k.T * alpha).T, axis = 0)

        error = error + (np.sum(np.square(np.subtract(y_hat, _y))) / _y.shape[0])

    return error/num_folds

@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / median(dist(x_i, x_j)^2 for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """

    gamma = 1 / np.median(np.subtract.outer(x, x)[np.tril_indices(x.shape[0], k=-1)]** 2).item()

    new_lambda = 0.0
    new_loss = float("inf")
    lambdas = 10 ** np.linspace(-5, -1)

    for _lambda in lambdas:
        loss = cross_validation(x, y, rbf_kernel, gamma, _lambda, num_folds)
        if loss < new_loss:
            new_lambda = _lambda
            new_loss = loss

    return new_lambda, gamma

@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You do not really need to search over gamma. 1 / median((x_i - x_j) for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution {7, 8, ..., 20, 21}
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [7, 8, ..., 20, 21]
    """

    new_lambda = 0.0
    new_d = 0.0
    new_loss = float("inf")

    lambdas = 10 ** np.linspace(-5, -1)
    _d = np.arange(7, 22)

    for _lambda in lambdas:
        for d in _d:
            loss = cross_validation(x, y, poly_kernel, d, _lambda, num_folds)
            if loss < new_loss:
                new_d = d
                new_lambda = _lambda
                new_loss = loss

    return new_lambda, new_d

@problem.tag("hw3-A", start_line=1)
def bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    bootstrap_iters: int = 300,
) -> np.ndarray:
    """Bootstrap function simulation empirical confidence interval of function class.

    For each iteration of bootstrap:
        1. Sample len(x) many of (x, y) pairs with replacement
        2. Train model on these sampled points
        3. Predict values on x_fine_grid (see provided code)

    Lastly after all iterations, calculated 5th and 95th percentiles of predictions for each point in x_fine_point and return them.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        bootstrap_iters (int, optional): [description]. Defaults to 300.

    Returns:
        np.ndarray: A (2, 100) numpy array, where each row contains 5 and 95 percentile of function prediction at corresponding point of x_fine_grid.

    Note:
        - See np.percentile function.
            It can take two percentiles at the same time, and take percentiles along specific axis.
    """
    x_= np.linspace(0, 1, 100)
    yhs = np.zeros((bootstrap_iters, len(x_)))

    for i in range(bootstrap_iters):
        index = np.random.choice(len(x), len(x))
        x_train = x[index]
        y_train = y[index]
        alphaV = train(x_train, y_train, kernel_function, kernel_param, _lambda)
        k = kernel_function (x_train, x_, kernel_param)
        yhs[i, :] = np.sum((k.T * alphaV).T, axis=0)

    return np.percentile(yhs, [5, 95], axis=0)

@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid
        C. For both rbf and poly kernels, plot 5th and 95th percentiles from bootstrap using x_30, y_30 (using the same fine grid as in part B)
        D. Repeat A, B, C with x_300, y_300
        E. Compare rbf and poly kernels using bootstrap as described in the pdf. Report 5 and 95 percentiles in errors of each function.

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
                To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    dataset = [(x_30, y_30), (x_300, y_300)]
    xfg = np.linspace(0, 1, 100)

    for i, data in enumerate(dataset):
        x_train = data[0]
        y_train = data[1]

        if i == 0:
            _lambda_rbf, gamma = rbf_param_search(x_train, y_train, len(x_train))
            _lambda_poly, d = poly_param_search(x_train, y_train, len(x_train))
        else:
            _lambda_rbf, gamma = rbf_param_search(x_train, y_train, 10)
            _lambda_poly, d = poly_param_search(x_train, y_train, 10)

        print(str(_lambda_rbf))
        print(str(gamma))
        print(str(_lambda_poly))
        print(str(d))

        alpha_rbf = train(x_train, y_train, rbf_kernel, gamma, _lambda_rbf)
        alpha_poly = train(x_train, y_train, poly_kernel, d, _lambda_poly)

        rbf_k = rbf_kernel(x_train, xfg, gamma)
        poly_k = poly_kernel(x_train, xfg, d)

        rbf_yhat = np.sum((rbf_k.T * alpha_rbf).T, axis = 0)
        poly_yhat = np.sum((poly_k.T * alpha_poly).T, axis = 0)
        true = f_true(xfg)

        plt.scatter(x_train, y_train, color="black", s=5, label="data")
        plt.plot(xfg, poly_yhat, "red", label="Poly Predictions")
        plt.plot(xfg, true, "green", label="True")
        plt.xlabel("x values ")
        plt.ylabel("y Values")
        plt.title("Poly Kernel")
        plt.ylim(-6, 6)
        plt.legend()
        plt.show()

        plt.scatter(x_train, y_train, color="black", s=5, label="data")
        plt.plot(xfg, rbf_yhat, "red", label="RBF Predictions")
        plt.plot(xfg, true, "green", label="True")
        plt.xlabel("x values ")
        plt.ylabel("y Values")
        plt.title("RBF Kernel")
        plt.ylim(-6, 6)
        plt.legend()
        plt.show()

        rbf_p = bootstrap(x_train, y_train, rbf_kernel, gamma, _lambda_rbf, 300)
        poly_p = bootstrap(x_train, y_train, poly_kernel, d, _lambda_poly, 300)

        plt.scatter(x_train, y_train, color="black", s=5, label="Data")
        plt.plot(xfg, poly_yhat, label="predictions ")
        plt.plot(xfg, true, label="True")
        plt.plot(xfg, poly_p[0, :], "blue", label="5th Percentile", linestyle="solid")
        plt.plot(xfg, poly_p[1, :], "pink", label="95th Percentile", linestyle="solid")
        plt.xlabel("x xalues")
        plt.ylabel("y Values")
        plt.title("Poly Bootstrap")
        plt.ylim(-6, 6)
        plt.legend()
        plt.show()

        plt.scatter(x_train, y_train, color="black", s=5, label="Data")
        plt.plot(xfg, rbf_yhat, "red", label="predictions")
        plt.plot(xfg, true, "green", label="True")
        plt.plot(xfg, rbf_p[0, :], "blue", label="5th Percentile ", linestyle = "solid")
        plt.plot(xfg, rbf_p[1, :], "pink", label="95th Percentile", linestyle = "solid")
        plt.xlabel("x values")
        plt.ylabel("y Values")
        plt.title("RBF Bootstrap ")
        plt.ylim(-6, 6)
        plt.legend()
        plt.show()

        if i == 1:
            temp = np.zeros(300)

            for i in range(300):
                index = np.random.choice(len(x_1000), len(x_1000))
                x = x_1000[index]
                y = y_1000[index]
                rbf_k = rbf_kernel(x_300, x, gamma)
                poly_k = poly_kernel(x_300, x, d)
                temp[i] = np.mean((y - np.sum((poly_k.T * alpha_poly).T, axis=0)) ** 2 - (y - np.sum((rbf_k.T * alpha_rbf).T, axis=0)) ** 2)

            bs = np.percentile(temp, [5, 95])

            print(str(bs[0]))
            print(str(bs[1]))

if __name__ == "__main__":
    main()
