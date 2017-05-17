import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split


def triangular_kernel_c(u):
        ker = (1 - u)
        ker[ker < 0] = 0
        return ker


def epanechnikov_kernel_c(u):
    ker = 0.75*(1-u**2)
    ker[ker < 0] = 0
    return ker


def quadric_kernel_c(u):
    ker = 0.9375*u**2
    ker[ker < 0] = 0
    return ker


def tricubic_kernel_c(u):
    ker = 0.8642*u**3
    ker[ker < 0] = 0
    return ker


def uniform_kernel_c(u):
    ker = abs(u)
    ker[ker < 1] = 0.5
    ker[ker >= 1] = 0
    return ker


def gaussian_kernel_c(u):
    ker = 0.4*2.71828**(-0.5*u**2)
    ker[ker < 0] = 0
    return ker


def sobolev_kernel_c( u):
    ker = 0.82857*np.exp(u**2/(u**2-1))
    ker[ker < 0] = 0
    return ker


def sinus_kernel_c(u):
    ker = 0.78539*np.cos(1.57079*u)
    ker[ker < 0] = 0
    return ker


class TriangularKernel:
    def __call__(self, u, *args, **kwargs):
        return triangular_kernel_c(u)


class EpanechnikovKernel:
    def __call__(self, u, *args, **kwargs):
        return epanechnikov_kernel_c(u)


class QuadricKernel:
    def __call__(self, u, *args, **kwargs):
        return quadric_kernel_c(u)


class TricubicKernel:
    def __call__(self, u, *args, **kwargs):
        return tricubic_kernel_c(u)


class UniformKernel:
    def __call__(self, u, *args, **kwargs):
        return uniform_kernel_c(u)


class GaussianKernel:
    def __call__(self, u, *args, **kwargs):
        return gaussian_kernel_c(u)


class SobolevKernel:
    def __call__(self, u, *args, **kwargs):
        return sobolev_kernel_c(u)


class CosinusKernel:
    def __call__(self, u, *args, **kwargs):
        return sinus_kernel_c(u)


def create_kernel(name):
    if name == 'Triangular':
        return TriangularKernel()
    if name == 'Epanechnikov':
        return EpanechnikovKernel()
    if name == 'Quadric':
        return QuadricKernel()
    if name == 'Tricubic':
        return TricubicKernel()
    if name == 'Uniform':
        return UniformKernel()
    if name == 'Gaussian':
        return GaussianKernel()
    if name == 'Sobolev':
        return SobolevKernel()
    if name == 'Cosinus':
        return CosinusKernel()
    assert 0, 'No such kernel: '+name


class KernelRegression(BaseEstimator, RegressorMixin):
    """Kernel Regression

    KR builds model of input and output data, wich later will look
    for nearest points in ``h`` radius and predict output using weighted
    mean of inputs.

    Parameters
    ----------
    ker : {'Epanechnikov', 'Gaussian', 'Triangular', 'Quadric', 'Tricubic',
        'Uniform', 'Sobolev', 'Cosinus'}, (default='Epanechnikov')
        Kernel function to be used.

    train_steps: int (default=50)
        Training steps set the number of iteration to estimate ``h``.

    h_decrease: int (default=30)
        Sets the number witch used to decrease ``h`` over steps.

    h_start_coef: float (default=0.5)
        Sets the coefecente for h to be estimated inn following
        function (X.max - X.min)*h_start_coef.

    Attributes
    ----------
    feature_importances_ : array, shape = [n_features]
        The feature importances (the higher, the more important the feature).

    ker_ : KernelFunction
        The concrete ``KernelFunction`` object.

    tree_ : cKDTree
        Tree object witch contains whole training data set.

    cs_ : list
        Estimated values of ``h`` for every input feature.
    """
    def __init__(self, ker='Epanechnikov', train_steps=50, h_decrease=30, h_start_coef=0.5):

        self.ker = ker
        self.train_steps = train_steps
        self.h_decrease = h_decrease
        self.h_start_coef = h_start_coef

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
        self.X_ = X_train
        self.y_ = y_train
        self.cs_ = [(np.max(a) - np.min(a)) * self.h_start_coef for a in X_train.T]
        self.ker_ = create_kernel(self.ker)
        self.tree_ = cKDTree(self.X_)
        self.debug_ = []
        for cs, value in enumerate(self.cs_):
            score = 0
            for i in range(self.train_steps):
                self.debug_ = []
                temp_cs = self.cs_[cs]
                self.cs_[cs] = self.cs_[cs] - (self.cs_[cs] / self.h_decrease)
                s = self.score(X_test, y_test)
                if s > score:
                    score = s
                if s < score or len(self.debug_) > 1:
                    self.cs_[cs] = temp_cs
                    break
        self.X_ = X
        self.y_ = y
        self.tree_ = cKDTree(self.X_)
        self.feature_importances_ = 1-(np.array(self.cs_))/np.sum(self.cs_)
        return self

    def _model_in_point(self, x):
        dots = self.tree_.query_ball_point(x, p=10, r=np.max(self.cs_), eps=0.1)
        if len(dots) != 0:
            t = self.ker_((self.X_[dots] - x) / self.cs_).prod(axis=1)
            u = np.sum(np.multiply(t, self.y_[dots]))
            t = np.sum(t)
            if t != 0:
                return u / t
        self.debug_.append(x)
        return np.mean(self.y_)

    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_', 'tree_', 'ker_', 'cs_'])
        X = check_array(X)
        f = self._model_in_point
        return np.array([f(x) for x in X])
