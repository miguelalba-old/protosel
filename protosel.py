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