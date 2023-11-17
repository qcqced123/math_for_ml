import pygame, sys
from math import pi, sqrt, cos, sin, atan2
from random import randint, uniform
import numpy as np
from typing import Callable
from matplotlib import pyplot as plt


class PolygonModel():
    """
    Express object in 2D space
    Args:
        points: central coordinate value of object
        x: x coordinate value of object
        y: y coordinate value of object
        vx: x velocity value of object
        vy: y velocity value of object
    """
    def __init__(self, points: tuple[float, float]):
        self.points = points
        self.angle = 0
        self.x = 0
        self.y = 0
        self.vx = 0
        self.xy = 0


class Asteroid(PolygonModel):
    def __init__(self, points: tuple[float, float]):
        sides = randint(5, 9)
        vs = [vectors.to_cartesian((uniform(0.7, 1.0), 2 * pi * i / sides))
        super().__init__(vs)
