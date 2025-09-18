import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNNClassifier:
    def __init__(self, k=3):
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def _get_neighbors_and_predict(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        predicted_label = most_common[0][0]
        num_classes = len(np.unique(self.y_train))
        class_probabilities = np.zeros(num_classes)
        for class_label in range(num_classes):
            class_probabilities[class_label] = k_nearest_labels.count(class_label) / self.k
        epsilon = 1e-9
        non_zero_probabilities = class_probabilities[class_probabilities > 0]
        entropy = -np.sum(non_zero_probabilities * np.log2(non_zero_probabilities))
        max_entropy = np.log2(len(non_zero_probabilities)) if len(non_zero_probabilities) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        trustworthiness = 1 - normalized_entropy
        return predicted_label, class_probabilities, trustworthiness

    def predict(self, X):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("You must call fit before calling predict.")
        predictions = [self._get_neighbors_and_predict(x)[0] for x in X]
        return np.array(predictions)
    
    def trustworthiness(self, X):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("You must call fit before calling trustworthiness.")
        scores = [self._get_neighbors_and_predict(x)[2] for x in X]
        return np.array(scores)

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data[:, :2]  
    y = iris.target
    feature_names = iris.feature_names[:2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    k_values = [1, 3, 5, 7]
    my_accuracies = []
    sklearn_accuracies = []
    print("="*40)
    print("Accuracy Comparison with Scikit-learn")
    print("="*40)
    for k in k_values:
        my_knn = KNNClassifier(k=k)
        my_knn.fit(X_train, y_train)
        my_predictions = my_knn.predict(X_test)
        my_accuracy = accuracy_score(y_test, my_predictions)
        my_accuracies.append(my_accuracy)
        sklearn_knn = SklearnKNeighborsClassifier(n_neighbors=k)
        sklearn_knn.fit(X_train, y_train)
        sklearn_predictions = sklearn_knn.predict(X_test)
        sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
        sklearn_accuracies.append(sklearn_accuracy)
        print(f"k = {k}:")
        print(f"  My KNN Accuracy:         {my_accuracy:.4f}")
        print(f"  Scikit-learn KNN Accuracy: {sklearn_accuracy:.4f}\n")

    k_for_viz = 5
    my_knn_viz = KNNClassifier(k=k_for_viz)
    my_knn_viz.fit(X_train, y_train)
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = my_knn_viz.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ['darkred', 'darkgreen', 'darkblue']
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
    import seaborn as sns
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=iris.target_names[y_train],
                    palette=cmap_bold, alpha=1.0, edgecolor="black", style=y_train, s=50)
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=iris.target_names[y_test],
                    palette=cmap_bold, alpha=1.0, edgecolor="black", marker='X', s=150)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"KNN Decision Boundary (k = {k_for_viz})", fontsize=16)
    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    plt.legend(title="Classes", loc="upper left")
    print("\nDisplaying Decision Boundary Plot...")
    plt.show()
    trustworthiness_scores = my_knn_viz.trustworthiness(X_test)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_light,
                          s=trustworthiness_scores * 300 + 50,
                          edgecolor='black', alpha=0.8, linewidth=1.5)
    plt.title(f"Trustworthiness of Test Points (k = {k_for_viz})", fontsize=16)
    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    class_legend = plt.legend(handles=scatter.legend_elements()[0], 
                              labels=iris.target_names,
                              title="Classes")
    plt.gca().add_artist(class_legend)
    for score in [0.2, 0.6, 1.0]:
        plt.scatter([], [], c='gray', alpha=0.8, s=score * 300 + 50,
                    label=f'{score:.1f}', edgecolor='black')
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1.5, title='Trustworthiness', loc='lower right')
    print("Displaying Trustworthiness Plot...")
    plt.show()
