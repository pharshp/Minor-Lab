import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as SklearnGNB
from sklearn.metrics import accuracy_score

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {}
        self.means = {}
        self.vars = {}

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0)
    
    def _gaussian_pdf(self, x, mean, var):
        eps = 1e-6
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exponent = np.exp(- (x - mean) ** 2 / (2 * var + eps))
        return coeff * exponent

    def _predict_single(self, x):
        posteriors = []

        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(np.log(self._gaussian_pdf(x, self.means[c], self.vars[c])))
            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

X, y = make_classification(n_samples=1000, n_features=2, n_classes=2,
                           n_clusters_per_class=1, n_redundant=0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)
y_pred_custom = gnb.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print(f"Custom GNB Accuracy: {accuracy_custom:.4f}")

sk_gnb = SklearnGNB()
sk_gnb.fit(X_train, y_train)
y_pred_sklearn = sk_gnb.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Sklearn GNB Accuracy: {accuracy_sklearn:.4f}")

def plot_decision_boundary(model, X, y, title="Decision Boundary", resolution=0.01):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', label='Train')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_custom, cmap=plt.cm.coolwarm, edgecolors='k', marker='x', label='Test (Predicted)')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_decision_boundary(gnb, X, y, title="Custom Gaussian Naive Bayes Decision Boundary")
