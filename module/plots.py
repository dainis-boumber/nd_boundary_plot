from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.manifold.t_sne import TSNE
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

