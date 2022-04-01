import numpy as np
from collections import namedtuple

class Microhpone:
    def __init__(self) -> None:
        pass

class Array:
    def __init__(self) -> None:
        pass

class Source:
    def __init__(self,
        name:str, 
        id:int,
        type:int,
        channel:int, 
        pos:np.ndarray) -> None:

        self.name = name
        self.id = id
        self.type = type
        self.channel = channel
        self.pos = pos
        pass


import numpy as np

class Echo():
    def __init__(self):
        self.toa = 0
        self.amp = 0
        self.wall = 'direct'
        self.order = 0
        self.generator = 'direct'