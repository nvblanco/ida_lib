import random
from abc import ABC, abstractmethod


class PipelineOperation(ABC):
    """Abstract class of pipeline operations"""

    def __init__(self, op_type: str, probability: float = 1):
        """

        :param op_type : internal parameter to determine how to treat each operation.
            'geometry' | 'color' | 'Normalize' | 'Independent'
                - Geometry      : operations that can be applied by transformation matrix
                - Color         : operations that can be applied using a pixel-by-pixel mathematical formula
                - Normalize     : normalization operation to be applied last within color operations
                - Independent  : concrete operations with direct implementations
        :param probability : probability of applying the transform. Default: 1.
        """
        self.probability = probability
        self.type = op_type

    @abstractmethod
    def get_op_matrix(self):
        pass

    def get_op_type(self):
        return self.type

    """ returns a boolean based on a random number that determines whether or not to apply the operation"""

    def apply_according_to_probability(self) -> bool:
        return random.uniform(0, 1) < self.probability
