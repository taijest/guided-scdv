from typing import List

import numpy as np


class GaussianMixtureModel:
    def __init__(self, X: np.array, n_dists: int) -> None:
        self.X: np.array = X
        self.n_dists: int = n_dists
        self.n_words: int = X.shape[0]
        self.size: int = X.shape[1]
        self.mix_raito: np.array = None
        self.weights: np.array = None
        self.gausses: List[Gaussian] = [
            Gaussian(self.size) for _ in range(self.n_dists)
        ]

        self.init_params()

    def init_params(self) -> None:
        mix_raito = np.random.rand(self.n_dists)
        self.mix_raito = mix_raito / np.sum(mix_raito, axis=1)
        weights = np.random.rand(self.n_words, self.n_dists)
        self.weights = weights / np.sum(weights, axis=1)

    def _E_step(self):
        pdfs = np.array([gauss.pdf(self.X) for gauss in self.gausses])
        self.weights = np.sum(
            self.mix_raito * pdfs, axis=0
        ) / np.sum(np.sum(
            self.mix_raito * pdfs, axis=0
        ), axis=2)

    def _M_step(self):
        n = np.sum(self.weights, axis=0)
        self.mix_raito = n / self.n_words
        for idx, gauss in enumerate(self.gausses):
            mu = np.sum(self.weights[:, idx] * self.X, axis=0) / n[idx]
            sigma = np.sum(
                self.weights[:, idx] * (self.X - gauss.mu)**2
            ) / n[idx]
            gauss.mu = mu
            gauss.sigma = sigma


class Gaussian:
    def __init__(self, size: int) -> None:
        self.size: int = size
        self.mu: float = None
        self.sigma: float = None

        self.init_params()

    def init_params(self) -> None:
        self.mu = np.random.randn(self.size)
        self.sigma = np.random.rand(self.size)

    def pdf(self, X: np.array) -> float:
        return np.exp(
            - np.sum((X - self.mu)**2, axis=1) / (2 * self.sigma**2)
        ) / (np.sqrt(2 * np.pi * self.sigma**2))
