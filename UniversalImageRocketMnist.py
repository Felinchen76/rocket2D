# !pip install sktime scikit-image albumentations --quiet
import numpy as np
from skimage.color import rgb2gray
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import Rocket


class UniversalImageRocket:
    def __init__(self, num_kernels=10000, random_state=42):
        self.num_kernels = num_kernels
        self.random_state = random_state
        self.rocket = Rocket(num_kernels=self.num_kernels, random_state=self.random_state)
        self.scaler = StandardScaler()
        self.clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.is_fitted = False

    def _reshape_for_rocket(self, X):
        """
        If rgb image, convert to grayscale,
        then reshape for ROCKET: (n_instances, n_channels, series_length)
        """
        if X.ndim == 4 and X.shape[-1] == 3:
            X = np.array([rgb2gray(x) for x in X], dtype=np.float32)
        n_samples, h, w = X.shape
        return X.reshape(n_samples, 1, h * w)

    def fit(self, X, y):
        X_r = self._reshape_for_rocket(X)
        X_transformed = self.rocket.fit_transform(X_r, y)
        X_scaled = self.scaler.fit_transform(X_transformed)
        self.clf.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Modell muss zuerst gefittet werden!")
        X_r = self._reshape_for_rocket(X)
        X_transformed = self.rocket.transform(X_r)
        X_scaled = self.scaler.transform(X_transformed)
        return self.clf.predict(X_scaled)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
