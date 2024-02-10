"""
Utility functions and stuff
"""

import math

def getDistance(point1: list[float], point2: list[float]) -> float:
    """
    Returns the euclidean distance between 2 points

    Args:
        point1 (list[float]): Can be (and is encouraged to be) a 1-d numpy array
        point2 (list[float]): Can be (and is encouraged to be) a 1-d numpy array
    Returns:
        float: the distance
    """

    return math.sqrt(sum([(coord1 - coord2)**2 for (coord1, coord2) in zip(point1, point2)]))