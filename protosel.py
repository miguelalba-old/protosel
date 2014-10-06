import numpy as np
from sklearn import neighbors


def enn(data, target):
    """Edited Nearest Neighbor.

    Args:
        data: Data values array.
        target: Target values array

    Returns: Boolean mask of selected instances.
    """
    clf = neighbors.KNeighborsClassifier(3)
    clf.fit(data, target)
    sel = clf.predict(data) == target
    return sel


def renn(data, target):
    """Repeated ENN.

    Args:
        data: Data values array.
        target: Target values array.

    Returns: Boolean mask of selected instances.
    """
    # Select all the instances
    sel = np.ones(len(data), bool)
    clf = neighbors.KNeighborsClassifier(3)

    stop = False
    while not stop:
        clf.fit(data[sel], target[sel])
        # Indices of misclassified instances
        misclassified = (clf.predict(data[sel]) != target[sel]).nonzero()[0]
        # Unselect misclassified instances
        sel[misclassified] = False
        # If there is no misclassified instances, then stops
        if len(misclassified) == 0:
            stop = True

    return sel
