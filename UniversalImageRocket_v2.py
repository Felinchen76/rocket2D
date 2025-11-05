import numpy as np

class ImageRocket2D:
    """
    2D adaptation of the ROCKET algorithm for image data
    """

    def __init__(self, n_kernels=200, kernel_sizes=[(3,3),(5,5),(7,7)], random_state=42):
        self.n_kernels = n_kernels
        self.kernel_sizes = kernel_sizes
        self.random_state = random_state
        self.kernels = []
        np.random.seed(random_state)

    def _rocket_2d_activation(self, img, kernel):
        """convolution of image with 2D Kernel
        :return activation matrix """
        kh, kw = kernel.shape
        h, w = img.shape
        padded_img = np.pad(img, ((kh-1, kh-1), (kw-1, kw-1)), mode='constant')
        activation = np.zeros((h + kh -1, w + kw -1))
        for i in range(h + kh -1):
            for j in range(w + kw -1):
                patch = padded_img[i:i+kh, j:j+kw]
                activation[i,j] = np.sum(patch * kernel)
        return activation

    def _features_from_activation(self, activation):
        """extract f_max and f_ppv from activation matrix"""
        f_max = np.max(activation)
        f_ppv = np.mean(activation > 0)
        return np.array([f_max, f_ppv])

    def _generate_kernels(self):
        """generates random kernel for feature extraction."""
        self.kernels = []
        for _ in range(self.n_kernels):
            kh, kw = self.kernel_sizes[np.random.randint(0, len(self.kernel_sizes))]
            kernel = np.random.randn(kh, kw)
            self.kernels.append(kernel)

    def fit(self, X=None, y=None):
        """
        needed for sklearn compatibility.  Generates random kernels.
        """
        self._generate_kernels()
        return self

    def transform(self, X):
        """
        extract Rocket features from an array of images X.
        :return feature matrix (n_samples, n_kernels*2)
        """
        n_samples = len(X)
        features = np.zeros((n_samples, self.n_kernels*2))
        for i, kernel in enumerate(self.kernels):
            for j, img in enumerate(X):
                activation = self._rocket_2d_activation(img, kernel)
                features[j, 2*i:2*i+2] = self._features_from_activation(activation)
        return features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
