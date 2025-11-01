import numpy as np
import torch
from keras.datasets import cifar10
from skimage.color import rgb2gray
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sktime.transformations.panel.rocket import Rocket

class UniversalImageRocket:
    def __init__(self, num_kernels=6000, pca_components=5000, alpha=1.0, random_state=42):
        self.num_kernels = num_kernels
        self.pca_components = pca_components
        self.alpha = alpha
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rocket = Rocket(num_kernels=self.num_kernels, random_state=self.random_state)
        self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
        self.clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=self.alpha))

        self.X_train_feat = None
        self.X_test_feat = None

        print(f"Running on: {self.device}")

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        y_train, y_test = y_train.flatten(), y_test.flatten()

        # grey scale and normalize
        X_train = np.array([rgb2gray(x) for x in X_train], dtype=np.float32)
        X_test = np.array([rgb2gray(x) for x in X_test], dtype=np.float32)
        X_train /= 255.0
        X_test /= 255.0

        # rocket format: (n_instances, n_channels, series_length)
        X_train = X_train.reshape((X_train.shape[0], 1, -1))
        X_test = X_test.reshape((X_test.shape[0], 1, -1))

        return X_train, X_test, y_train, y_test

    def fit(self, X_train, y_train):
        print("Fitting ROCKET...")
        self.rocket.fit(X_train)
        self.X_train_feat = self.rocket.transform(X_train)
        print("Feature shape after ROCKET:", self.X_train_feat.shape)

        print("Applying PCA...")
        self.X_train_feat = self.pca.fit_transform(self.X_train_feat)

        print("Training RidgeClassifier...")
        self.clf.fit(self.X_train_feat, y_train)
        print("Training complete.")

    def predict(self, X_test):
        print("Transforming test data...")
        self.X_test_feat = self.rocket.transform(X_test)
        self.X_test_feat = self.pca.transform(self.X_test_feat)
        print("Predicting labels...")
        return self.clf.predict(self.X_test_feat)
