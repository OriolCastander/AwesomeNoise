"""
A bunch of methods with typical noise configurations
"""
from __future__ import annotations

import random

from awesomeNoise.awesomeNoise import AwesomeNoise
from awesomeNoise.noiseLayer import NoiseLayer


class WeightFunctions:
    @staticmethod
    def _polynomicWeightFunction(coeffs: list[float], distance: float, angle: float, radius: float) -> float:
        lPrime = 2 - distance / radius - 1
        sum = 0
        for i, coeff in enumerate(coeffs):
            sum += coeff * lPrime ** i

        return sum
    


class CoeffsFunctions:

    @staticmethod
    def _positiveQuadraticCoeffsFunction(noiseLayer: NoiseLayer, gridPoint: list[int]) -> list[float]:
        random.seed(noiseLayer.seed + ";coeff;" + ",".join([str(coord) for coord in gridPoint]))
        coeffs = [random.random() - 1 for _ in range(2)]
        return [-(coeffs[0] + coeffs[1])] + coeffs



class BaseHeightsFunctions:
    @staticmethod
    def _checkerboardBaseHeightFunction(noiseLayer: NoiseLayer, gridPoint: list[int]) -> float:
        return sum(gridPoint) % 2




class Scenario:
    

    @staticmethod
    def PositiveQuadratic(nDimensions: int, seed: str, radius: float, nLayers: int) -> AwesomeNoise:
        return AwesomeNoise.Harmonic(nDimensions, seed, radius, None, WeightFunctions._polynomicWeightFunction,
                                     nLayers, None, None, CoeffsFunctions._positiveQuadraticCoeffsFunction)
    
    
    @staticmethod
    def Checkerboard(nDimensions: int, seed: str) -> AwesomeNoise:
        return AwesomeNoise.Harmonic(nDimensions, seed, 2.3, None, WeightFunctions._polynomicWeightFunction,
                                     5, None, BaseHeightsFunctions._checkerboardBaseHeightFunction, CoeffsFunctions._positiveQuadraticCoeffsFunction)