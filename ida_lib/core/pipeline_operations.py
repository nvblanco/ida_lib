from abc import ABC, abstractmethod
import random


class PipelineOperation(ABC):
    """Abstract class of pipeline operations"""

    def __init__(self, type: str, probability: float = 1):
        """
        :param type : internal parameter to determine how to treat each operation.
            'geometry' | 'color' | 'Normalize' | 'Independient'
                - Geometry      : operations that can be applied by transformation matrix
                - Color         : operations that can be applied using a pixel-by-pixel mathematical formula
                - Normalize     : normalization operation to be applied last within color operations
                - Independient  : concrete operations with direct implementations
        :param probability : probability of applying the transform. Default: 1.
        """
        self.probability = probability
        self.type = type


    @abstractmethod
    def get_op_matrix(self):
        pass

    def get_op_type(self):
        return self.type

    """ returns a boolean based on a random number that determines whether or not to apply the operation"""
    def apply_according_to_probability(self) -> bool:
        return random.uniform(0, 1) < self.probability
