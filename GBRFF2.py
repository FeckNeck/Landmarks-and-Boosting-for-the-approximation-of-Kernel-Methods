import numpy as np
from scipy.optimize import fminbound, fmin_l_bfgs_b


class GBRFF:

    def __init__(self, gamma=1, Lambda=1, T=100):
        self.gamma = gamma
        self.Lambda = Lambda
        self.T = T
        np.random.seed(171222)
        self.randomState = np.random

    def loss_gradient(self, w_norm, X, b, y_wave, m):
        cos = np.cos(w_norm@X.T - b)
        exp = np.exp(-y_wave * cos)
        norm = np.linalg.norm(w_norm, 2)  # Float
        norm = norm*norm
        res1 = self.Lambda * norm + (1/m) * np.sum(exp)
        calc = y_wave * exp * np.sin(w_norm@X.T - b)
        res2 = (1/m)*calc@X+self.Lambda*2*w_norm
        return res1, res2

    def alpha(self, omega, X, y, b, w):
        calc = (y*np.cos(omega@X.T - b))@w
        alpha = 0.5*np.log((np.sum(w)+calc)/(np.sum(w)-calc))
        return alpha

    def fit(self, X, y):
        labels = sorted(np.unique(y))
        self.negativeLabel, self.positiveLabel = labels[0], labels[1]
        newY = np.ones(X.shape[0])
        newY[y == labels[0]] = -1
        y = newY
        mean_y = np.mean(y)
        self.H0 = 0.5 * np.log((1+mean_y) / (1-mean_y))
        m, d = X.shape
        Ht = np.full(m, self.H0)
        self.vec_omega = np.empty((self.T, d))
        self.vec_alpha = np.empty(self.T)
        self.vec_b = np.empty(self.T)

        w_norm = self.randomState.randn(self.T, d)*(2*self.gamma)**0.5
        for t in range(self.T):
            w = np.exp(-y*Ht)
            y_wave = y*w

            self.vec_b[t] = np.pi*2*fminbound(lambda n: np.sum(np.exp(
                (-y_wave)*np.cos(np.pi*2*n - w_norm[t]@X.T))),
                x1=-0.5, x2=0.5, xtol=1e-2)

            self.vec_omega[t], _, _ = fmin_l_bfgs_b(
                self.loss_gradient, x0=w_norm[t], args=(X, self.vec_b[t], y_wave, m), maxiter=10)

            self.vec_alpha[t] = self.alpha(
                self.vec_omega[t], X, y, self.vec_b[t], w)

            Ht += self.vec_alpha[t] * \
                np.cos(self.vec_omega[t]@X.T - self.vec_b[t])

    def predict(self, X):
        calc = np.cos(self.vec_b[:, None]-self.vec_omega@X.T)
        pred = self.H0+self.vec_alpha@(calc)
        yPred = np.full(X.shape[0], self.positiveLabel)
        yPred[pred < 0] = self.negativeLabel
        return yPred
