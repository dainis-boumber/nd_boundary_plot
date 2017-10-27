from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.manifold.t_sne import TSNE
import matplotlib.pyplot as plt
from sklearn.datasets.base import load_digits
from sklearn.datasets.base import load_iris
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np

# draw a projection onto 2D using vornoi tessalation. Approximates highly dimensional boundaries in 2D
# see https://pure.uva.nl/ws/files/2110683/164710_431596.pdf
def nd_boundary_plot(X, y, y_predicted, ax, resolution = 256, **kwargs):
    X_train = TSNE(n_components=2).fit_transform(X)
    stdx, stdy = np.std(X_train[:,0]), np.std(X_train[:, 1])
    xmin, xmax = np.min(X_train[:,0]) - stdx, np.max(X_train[:,0]) + stdx
    ymin, ymax = np.min(X_train[:,1]) - stdy, np.max(X_train[:,1]) + stdy
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, resolution), np.linspace(ymin, ymax, resolution))
    background_model = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_predicted)
    voronoi = background_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape((resolution, resolution))
    ax.contourf(xx, yy, voronoi)
    ax.scatter(X_train[:,0], X_train[:,1], c=y)
    ax.set(**kwargs)

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
