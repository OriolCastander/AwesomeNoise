"""
Awesome noise stuff.

TODO: Caching stuff, maybe? Angles?
"""

from typing import Union, Callable
import numpy as np

import scipy.signal

from awesomeNoise.noiseLayer import NoiseLayer



class AwesomeNoise:
    """
    Awesome Noise. This class is basically a wrapper for a list of layers

    Attributes:

        nDimensions (int):
            Number of dimensions of the noise (2d, 3d...). Must be at least 1

        sizes (np.ndarray):
            A 1d array with the sizes of each noise layer

        weights (np.ndarray):
            A 1d array with the weights of each noise layer

        layers (list[NoiseLayer]):
            List of noise layers that all contribute to the final value
    
    """

    def __init__(self, nDimensions: int, seed: str, radius: float,nCoeffs: Union[int, None], weightFunction: Callable[[list[float], float, float, float], float],
                sizes: list[float], weights: list[float], restrictions: Union[list[Union[tuple[int, int], None]], None] = None,
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

            sizes (list[float]): Size of each noise layer

            weights (list[float]): Weight of each noise layer

            restrictions (Union[list[Union[tuple[int, int], None]], None]): Restructions for grid points considered. If None, no restrictions

            customBaseHeightFunction (Union[Callable[["NoiseLayer", list[int]], float], None]):  Custom function of the form [(noiseLayer, [gridX, gridY, gridZ...]) -> height]. Dafault is run if None

            customCoeffsFunction (Union[Callable[["NoiseLayer", list[int]], float], None]):  Custom function of the form [(noiseLayer, [gridX, gridY, gridZ...]) -> height]. Dafault is run if None

        """
        
        self.nDimensions: int = nDimensions
        self.sizes: list[float] = np.array(sizes)
        self.weights: list[float] = np.array(weights)

        self.layers: list[NoiseLayer] = []
        for i in range(len(sizes)):
            layer = NoiseLayer(nDimensions, seed + ":" + str(i), radius, nCoeffs, weightFunction, restrictions, customBaseHeightFunction, customCoeffsFunction)
            self.layers += [layer]

    
    @staticmethod
    def Harmonic(nDimensions: int, seed: str, radius: float,nCoeffs: Union[int, None], weightFunction: Callable[[list[float], float, float, float], float],
                nLayers: int, restrictions: Union[list[Union[tuple[int, int], None]], None] = None,
                customBaseHeightFunction: Union[Callable[["NoiseLayer", list[int]], float], None] = None,
                customCoeffsFunction: Union[Callable[["NoiseLayer",list[int]], list[float]], None] = None
                ) -> "AwesomeNoise":
        
        """
        Noise layering pattern in which each layer's size and weight is a power of 2
        
        Args:
            nDimensions (int): Number of dimensions of the noise (2d, 3d...). Must be at least 1
        
            seed (str): A seed for the random number generation

            radius (float): The radius of action of the noise

            nCoeffs (Union[int, None]): The number of random coefficients that need to be generated for the height function. If none, custom function should be used

            weightFunction (Callable[[list[float], float, float, float], float]): returns the weight of a gridPoint in an arbitrary point. Of form [(coeffs, distance, angle, radius) -> weight]

            nLayers (int): The number of layers

            restrictions (Union[list[Union[tuple[int, int], None]], None]): Restructions for grid points considered. If None, no restrictions

            customBaseHeightFunction (Union[Callable[["NoiseLayer", list[int]], float], None]):  Custom function of the form [(noiseLayer, [gridX, gridY, gridZ...]) -> height]. Dafault is run if None

            customCoeffsFunction (Union[Callable[["NoiseLayer", list[int]], float], None]):  Custom function of the form [(noiseLayer, [gridX, gridY, gridZ...]) -> height]. Dafault is run if None

        Returns:
            AwesomeNoise: The noise object
        """
        
        sizes = [2**i for i in range(nLayers)]
        weights = [2**i for i in range(nLayers)][::-1]

        return AwesomeNoise(nDimensions, seed, radius, nCoeffs, weightFunction, sizes, weights, restrictions, customBaseHeightFunction, customCoeffsFunction)
    

    @staticmethod
    def getDivergenceValues(values: np.ndarray) -> np.ndarray:
        """
        Returns the divergence the values (or for that matter, any scalar field)

        Args:
            values (np.ndarray): The values

        Returns:
            np.ndarray: The divergence values
        """

        grad = np.array(np.gradient(values))
        return np.sum(grad,0)
    

    @staticmethod
    def getLocalMaximumValues(values: np.ndarray, nNeightbors: int, ordered: bool = False) -> tuple[list[list[float]], list[float]]:
        """
        Returns indices and heights of local maximums

        Args:
            values (np.ndarray): The values

            nNeightbors (int): The amount of neighbors to compare each value to (must be greater that all of them to be a maximum). The greater, the stricter

            ordered (bool): Uf they should be ordered

        Returns:
            tuple[list[list[float]], list[float]]: First element is a list of the indices (list[[maxX, maxY, maxZ...]]), second is list of heights
        """

        candidates =  np.array(scipy.signal.argrelmax(values, 1, 1)).T
        # NOTE: scipy.signal.argrelmax should already do the job, dkw but it does freaking not. However, it condiderably narrows the amount of points that then
        #have to be "manually" searched as no "real" maximums will be ommited by argrelmax 

        sizes = values.shape

        maxIndices = []
        maxValues = []

        for candidate in candidates:
            validCandidate = True

            for dimension, coord in enumerate(candidate):
                if not nNeightbors < coord <= sizes[dimension] - nNeightbors:
                    validCandidate = False
                    break

            if not validCandidate: continue

            slices = [slice(coord - nNeightbors, coord + nNeightbors + 1) for coord in candidate]
            neighborArray = values[tuple(slices)]

            candidateValue = values[tuple([slice(coord, coord + 1) for coord in candidate])].flatten()[0]

            if  candidateValue >= neighborArray.max():
                maxIndices += [candidate]
                maxValues += [candidateValue]

        if ordered:
            ind = np.argsort(maxValues)[::-1]
            maxIndices = maxIndices[ind]
            maxValues = maxValues[ind]

        return (maxIndices, maxValues)





    def getValue(self, point: list[float]) -> float:
        """
        Gets the height value of a point

        Args:
            point (list[float]): The coords of the point (can be a numpy 1d array)

        Returns:
            float: The height
        """
        
        if type(point) != np.ndarray: point = np.array(point)
        
        compactArray = np.array([{"layer": layer, "size": size} for (layer,size) in zip(self.layers, self.sizes)])
        heights = np.vectorize(lambda compact: compact["layer"].getValueNoNumpy(point * compact["size"]))(compactArray)

        ###NOTE: COMPACT ARRAY IS A FREAKING MAP BECAUSE I DONT KNOW WHY BUT I COULDNT CREATE A FREAKING 1D ARRAY OF TUPLES, ITD ALWAYS BUILD A 2D ARRAY

        return np.sum(heights) / np.sum(self.weights)
        

    def getValues(self, distributions: Union[np.ndarray, list[list[float]]]) -> np.ndarray:
        """
        Returns a grid of heights

        Args:
            distributions (Union[np.ndarray, list[list[float]]]): List of included points for each dimension: E.g. [[2,3],[4,5]] will get heights for points [[2,4], [2,5], [3,4], [3,5]]

        Returns:
            np.ndarray: The heights grid
        """
        
        function = np.vectorize(lambda *vals: self.getValue(vals))
        meshgrid = np.meshgrid(*[d for d in distributions])

        values = function(*meshgrid).reshape(*[len(d) for d in distributions][::-1])

        return values
    

    
