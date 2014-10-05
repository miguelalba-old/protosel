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
    sel = np.arange(len(data))
    clf = neighbors.KNeighborsClassifier(3)

    stop = False
    while not stop:
        clf.fit(data[sel], target[sel])
        result = clf.predict(data[sel]) == target[sel]
        # If all the selected instances are classified correctly then stops
        if np.count_nonzero(-result) == 0:
            stop = True
        else:
            sel = result.nonzero()[0]

    mask = np.zeros(len(data), bool)
    mask[sel] = True

    return mask
