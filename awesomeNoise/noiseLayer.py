"""
Class that represents a noise layer.

A noise layer is already capable of generating smooth continous noise throughout all space
using the awesomeNoise method.

TODO: angles
"""

from typing import Union, Callable
import random
import math
import numpy as np
import itertools

from awesomeNoise.utils import getDistance


class NoiseLayer:
    """
    A layer of noise, capable of generating noise by itself

    Attributes:

        nDimensions (int):
            Number of dimensions of the noise (2d, 3d...). Must be at least 1
    
        seed (str):
            A seed for the random number generation

        radius (float):
            The radius of action of the noise

        nCoeffs (Union[int, None]):
            The number of random coefficients that need to be generated for the height function
            If None, it is implied that a customCoeffsFunction is used to get those coefficients.

        weightFunction (Callable[[list[float], float, float, float], float]):
            returns the weight of a gridPoint in an arbitrary point based on the distance and angle between them.
            Of form [(coeffs, distance, angle, radius) -> weight].

        restrictions (list[tuple[int, int]]):
            For each dimension there is a tuple of form (min, max), inclusive. Grid points with values in the corresponding axis
            outside of that interval will be treated as invalid

        customBaseHeightFunction (Union[Callable[["NoiseLayer", list[int]], float], None]):
            Custom function of the form [(noiseLayer, [gridX, gridY, gridZ...]) -> height] that is run instead of the default
            getBaseHeight. If None, the default is run

        customCoeffsFunction (Union[Callable[["NoiseLayer", list[int]], list[float]], None]):
            Custom function of the form [(noiseLayer, [gridX, gridY, gridZ...]) -> coeffs] that is run instead of the default
            getCoeffs. If None, the default is run
             
    """

    def __init__(self, nDimensions: int, seed: str, radius: float, nCoeffs: Union[int, None], weightFunction: Callable[[list[float], float, float, float], float],
                 restrictions: Union[list[Union[tuple[int, int], None]], None] = None,
                 customBaseHeightFunction: Union[Callable[["NoiseLayer", list[int]], float], None] = None,
                 customCoeffsFunction: Union[Callable[["NoiseLayer",list[int]], list[float]], None] = None,
                ) -> None:
        """
        Args:
            nDimensions (int): Number of dimensions of the noise (2d, 3d...). Must be at least 1
        
            seed (str): A seed for the random number generation

            radius (float): The radius of action of the noise

            nCoeffs (Union[int, None]): The number of random coefficients that need to be generated for the height function. If none, custom function should be used

            weightFunction (Callable[[list[float], float, float, float], float]): returns the weight of a gridPoint in an arbitrary point. Of form [(coeffs, distance, angle, radius) -> weight]

            restrictions (Union[list[Union[tuple[int, int], None]], None]): Restructions for grid points considered. If None, no restrictions

            customBaseHeightFunction (Union[Callable[["NoiseLayer", list[int]], float], None]):  Custom function of the form [(noiseLayer, [gridX, gridY, gridZ...]) -> height]. Dafault is run if None

            customCoeffsFunction (Union[Callable[["NoiseLayer", list[int]], float], None]):  Custom function of the form [(noiseLayer, [gridX, gridY, gridZ...]) -> height]. Dafault is run if None

        """
        
        self.nDimensions: int = nDimensions
        self.seed: str = seed
        self.radius: float = radius
        self.nCoeffs: Union[int, None] = nCoeffs
        self.weightFunction: Callable[[list[float], float, float, float], float] = weightFunction

        self.customBaseHeightFunction: Union[Callable[["NoiseLayer", int, int], float], None] = customBaseHeightFunction
        self.customCoeffsFunction: Union[Callable[["NoiseLayer", int, int], list[float]], None] = customCoeffsFunction

        #CONSTRUCT RESTRICTIONS
        self.restrictions: list[tuple[int, int]] = []
        if restrictions is None:
            for _ in range(nDimensions): self.restrictions += [(-10_000_000, 10_000_000)]
        else:
            if len(restrictions) != nDimensions:
                raise Exception(f"Lenght of restrictions: {len(restrictions)} must match number of dimensions {nDimensions}")
            
            for restriction in restrictions:
                if restriction is None: self.restrictions += [(-10_000_000, 10_000_000)]
                else: self.restrictions += [restrictions]

    

    def _getBaseHeight(self, gridPoint: list[int]) -> float:
        """
        Returns the base height of the grid point, either with the default random value of via a custom function

        Args:
            gridPoint (list[int]): Coordinates of the grid point. Can be a 1-d numpy array
    
        Returns:
            float: The height of the gridPoint
        """
        if self.customBaseHeightFunction is not None:
            return self.customBaseHeightFunction(self, gridPoint)
        
        hashString = self.seed + ":base:," + ",".join([str(coord) for coord in gridPoint])
        random.seed(hash(hashString))
        return random.random()
    

    def _getCoeffs(self, gridPoint: list[int]) -> list[float]:
        """
        Returns the coefficients for a gridPoint. These are uniformly distributed in [0,1) if the default function
        is run or fetched via the custom function

        Args:
            gridPoint (list[int]): Coordinates of the grid point. Can be a 1-d numpy array
    
        Returns:
            list[float]: The coeffs for that gridPoint
        """

        if self.customCoeffsFunction is not None:
            return self.customCoeffsFunction(self, gridPoint)
        
        hashString = self.seed + ":coeffs:," + ",".join([str(coord) for coord in gridPoint])
        random.seed(hash(hashString))

        return [random.random() for _ in range(self.nCoeffs)]
    

    def _getWeight(self, coeffs: list[float], distance: float, angle: float) -> float:
        """
        Interpolate-cuts (not sure if anybody else calls this an interpolation cut) the weight function to get the weight exercized from a
        gridPoint to an arbitrary point

        Args:
            coeffs (list[float]): Coefficients of the grid Point. Can be a numpy 1-d array

            distance (float): Euclidean distance between the gridPoint and point

            angle (float): Angle between the gridPoint and point

        Returns:
            float: The weight
        """

        return (1 - distance / self.radius) * self.weightFunction(coeffs, distance, angle, self.radius)
    

    def getValue(self, point: list[float]) -> float:
        """
        Returns the value at a point
        Args:
            point (list[float]): Coords of the point. Can be a numpy 1-d array
        Returns:
            float: the value

        WARNING: for some reason slower than w/out numpy
        """

        if len(point) != self.nDimensions:
            raise Exception(f"Point must have the same dimensions of the noise layer. Point has {len(point)}, layer: {self.nDimensions}")

        floors = [math.floor(coord) for coord in point]
        radiusCeil = math.ceil(self.radius)

        potentialRanges = []

        for i in range(self.nDimensions):
            potentialRanges += [list(range(max(floors[i] - radiusCeil, self.restrictions[i][0]), min(floors[i] + radiusCeil + 2, self.restrictions[i][1] + 1)))]

        compareFunction = lambda gridPoint: getDistance(point, gridPoint)

        potentialPoints = np.vstack([np.ravel(a) for a in np.meshgrid(*potentialRanges)]).T
        potentialDistances = np.apply_along_axis(compareFunction, 1, potentialPoints)

        mask = potentialDistances < self.radius

        validPoints = potentialPoints[mask]

        distances = np.reshape(potentialDistances[mask], (-1, 1))
        baseHeights = np.apply_along_axis(self._getBaseHeight, 1, validPoints)
        coeffs = np.apply_along_axis(self._getCoeffs, 1, validPoints)


        #"Destructures" the compact array that holds distances and coeffs into a tuple with distances and coeffs that 
        #can be fed into self.getWeight
        unpackCompactWeightArray: Callable[[np.ndarray], tuple[list[float], float, float]] = lambda compactArray: (compactArray[1:], compactArray[0], 0.0)

        weights = np.apply_along_axis(lambda x: self._getWeight(*unpackCompactWeightArray(x)), 1, np.concatenate([distances, coeffs], 1))

        totalWeight = np.sum(weights)
        totalHeight = np.dot(weights, baseHeights)

        return totalHeight / totalWeight
    
    
    def getValueNoNumpy(self, point: list[float]) -> float:
        """
        Returns the value at a point
        Args:
            point (list[float]): Coords of the point. Can be a numpy 1-d array
        Returns:
            float: the value
        """

        if len(point) != self.nDimensions:
            raise Exception(f"Point must have the same dimensions of the noise layer. Point has {len(point)}, layer: {self.nDimensions}")

        floors = [math.floor(coord) for coord in point]
        radiusCeil = math.ceil(self.radius)


        potentialRanges = []

        for i in range(self.nDimensions):
            potentialRanges += [list(range(max(floors[i] - radiusCeil, self.restrictions[i][0]), min(floors[i] + radiusCeil + 2, self.restrictions[i][1] + 1)))]

        distances = []
        baseHeights = []
        coeffss = []

        for potentialGridPoint in itertools.product(*potentialRanges): #itertools.product: cartesian product of the lists of ranges
            distance = getDistance(point, potentialGridPoint)
                
            if distance < self.radius:
                distances += [distance]
                baseHeights += [self._getBaseHeight(potentialGridPoint)]
                coeffss += [self._getCoeffs(potentialGridPoint)]


        if len(distances) == 0:
            return 0#raise Exception("Could not get at least 1 grid point")

        totalWeight = 0
        currentHeight = 0

        for i, (distance, baseHeight, coeffs) in enumerate(zip(distances, baseHeights, coeffss)):
            weight = self._getWeight(coeffs, distance, 0)
            totalWeight += weight
            currentHeight += weight * baseHeight

        currentHeight /= totalWeight

        return currentHeight