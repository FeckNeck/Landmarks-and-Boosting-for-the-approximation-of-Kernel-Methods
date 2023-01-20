import numpy as np
import pandas as pd
from scipy import optimize
from scipy.optimize import fminbound
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

np.random.seed(171222)


class gbrff(object):

    def __init__(self, gamma=0.1, Lambda=0, T=100, randomState=np.random):
        self.T = T
        self.randomState = randomState
        self.Lambda = Lambda
        self.gamma = gamma

    def loss_grad(self, omega):
        dots = np.dot(omega, self.XT) - self.b
        self.yTildePred = np.cos(dots)
        v0 = np.exp(self.yTildeN*self.yTildePred)
        return ((1/self.n)*np.sum(v0) + self.Lambda*(omega.T.dot(omega)),
                (1/self.n)*(self.yTilde*v0*np.sin(dots)).dot(
            self.X) + self.Lambda*2*omega)

    def fit(self, X, y):
        # Assuming there is two labels in y. Convert them in -1 and 1 labels.
        labels = sorted(np.unique(y))
        self.negativeLabel, self.positiveLabel = labels[0], labels[1]
        newY = np.ones(X.shape[0])  # Set all labels at 1
        newY[y == labels[0]] = -1  # except the smallest label in y at -1.
        y = newY
        self.n, d = X.shape
        meanY = np.mean(y)
        self.initPred = 0.5*np.log((1+meanY)/(1-meanY))
        curPred = np.full(self.n, self.initPred)
        pi2 = np.pi*2
        self.omegas = np.empty((self.T, d))
        self.alphas = np.empty(self.T)
        self.xts = np.empty(self.T)
        inits = self.randomState.randn(self.T, d)*(2*self.gamma)**0.5
        self.X = X
        self.XT = X.T
        for t in range(self.T):
            init = inits[t]
            wx = init.dot(self.XT)
            w = np.exp(-y*curPred)
            self.yTilde = y*w
            self.yTildeN = -self.yTilde

            self.b = pi2*fminbound(lambda n: np.sum(np.exp(
                self.yTildeN*np.cos(pi2*n - wx))), -0.5, 0.5, xtol=1e-2)

            self.xts[t] = self.b
            self.omegas[t], _, _ = optimize.fmin_l_bfgs_b(
                func=self.loss_grad, x0=init, maxiter=10)
            vi = (y*self.yTildePred).dot(w)
            vj = np.sum(w)
            alpha = 0.5*np.log((vj+vi)/(vj-vi))
            curPred += alpha*self.yTildePred
            self.alphas[t] = alpha

    def predict(self, X):
        pred = self.initPred+self.alphas.dot(
            np.cos(self.xts[:, None]-self.omegas.dot(X.T)))
        # Then convert back the labels -1 and 1 to the labels given in fit
        yPred = np.full(X.shape[0], self.positiveLabel)
        yPred[pred < 0] = self.negativeLabel
        return yPred

    def decision_function(self, X):
        return self.initPred+self.alphas.dot(
            np.cos(self.xts[:, None]-self.omegas.dot(X.T)))
