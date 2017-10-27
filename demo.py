from sklearn.datasets.base import load_iris
from sklearn.datasets.base import load_digits
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neural_network import MLPClassifier
from module.plots import nd_boundary_plot
import numpy as np, matplotlib.pyplot as plt

def main():
    X, y = load_digits(n_class=10, return_X_y=True)
    model = MLPClassifier(hidden_layer_sizes=(100,100)).fit(X, y)
    y_predicted = model.predict(X)
    ax0_0 = plt.subplot2grid([2, 1], (0, 0))
    nd_boundary_plot(X, y, y_predicted, ax0_0, resolution=100, adjustable='box-forced', aspect='equal')

    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression().fit(X, y)
    y_predicted = model.predict(X)
    ax1_0 = plt.subplot2grid([2, 1], (1, 0))
    nd_boundary_plot(X, y, y_predicted, ax1_0, resolution=100, adjustable='box-forced', aspect='equal')
    #plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
