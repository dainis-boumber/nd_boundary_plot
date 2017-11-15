from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.manifold.t_sne import TSNE
import matplotlib.pyplot as plt
from sklearn.datasets.base import load_digits
from sklearn.datasets.base import load_iris
from sklearn.datasets import make_circles
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble.bagging import BaggingClassifier
import numpy as np

# draw a projection onto 2D using vornoi tessalation. Approximates highly dimensional boundaries in 2D
# see https://pure.uva.nl/ws/files/2110683/164710_431596.pdf
def nd_boundary_plot(X_tst, y_predicted, model, ax, resolution = 256):
    if len(X_tst.shape) != 2:
        raise ValueError("X must be ndarray of the form [nsamples, nfeatures]")
    if X_tst.shape[0] < 2:
        raise ValueError("Must have at least 2 dimensions")
    if not hasattr(model, "classes_"):
        raise ValueError("Model has to be trained first")
    if len(model.classes_) < 2:
        raise ValueError("Classification must be at least binary")
    #done with sanity checks

    if X_tst.shape[1] == 2: #2 dimensions
        X = X_tst
        xmin, xmax = np.min(X[:,0]), np.max(X[:,0])
        ymin, ymax = np.min(X[:,1]), np.max(X[:,1])
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, resolution), np.linspace(ymin, ymax, resolution))

        if hasattr(model, "decision_function") or len(model.classes_) != 2: #model does not comute posterior or hard to graph
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:#lots of dimensions
        X = TSNE(n_components=2).fit_transform(X_tst)
        background_model = KNeighborsClassifier(n_neighbors=1).fit(X, y_predicted)
        xmin, xmax = np.min(X[:,0]), np.max(X[:,0])
        ymin, ymax = np.min(X[:,1]), np.max(X[:,1])
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, resolution), np.linspace(ymin, ymax, resolution))
        Z = background_model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape((resolution, resolution))
    ax.contourf(xx, yy, Z, alpha=.3)
    ax.scatter(X[:,0], X[:,1], c=y_predicted)

def main():

    X, y = load_digits(n_class=10, return_X_y=True)
    mesh_limits = (np.min(X[:, 0]) - .5, np.max(X[:, 0]) + .5, np.min(X[:, 1]) - .5, np.max(X[:, 1]) + .5)

    model = MLPClassifier(hidden_layer_sizes=(100,100)).fit(X, y)
    ax = plt.subplot2grid([2, 2], (0, 0))
    nd_boundary_plot(X, model.predict(X), model, ax=ax, resolution=100)

    X, y = make_circles()
    model = KNeighborsClassifier().fit(X, y)
    ax = plt.subplot2grid([2, 2], (1, 1))
    mesh_limits = (np.min(X[:, 0]) - .5, np.max(X[:, 0]) + .5, np.min(X[:, 1]) - .5, np.max(X[:, 1]) + .5)
    nd_boundary_plot(X, model.predict(X), model, ax=ax, resolution=100)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
