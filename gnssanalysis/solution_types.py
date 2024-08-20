import logging
from typing import Optional

logging.basicConfig(format="%(asctime)s [%(funcName)s] %(levelname)s: %(message)s")

class SolutionType:
    _name: str
    _long_name: str

    def __init__(self, name: str, long_name: str) -> None:
        self._name = name
        self._long_name = long_name

    @property
    def name(self):
        return self._name

    @property
    def long_name(self):
        return self._long_name

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return self._name

    def __eq__(self, other):
        """
        Override default equality check
        """
        # If we're the unknown shorthand "UNK", consider None equivalent. Both are expressions of unknown solution type
        if self._name == "UNK" and other is None:
            return True

        if not isinstance(other, SolutionType):
            return False
        return self._name == other._name
        # Note that in Python, there is both an equality and an inequality check.
        # But in Python 3 the inequality check leverages 'not __eq__()' by default.


class SolutionTypes:
    """
    Defines valid solution type identifiers specified for use in the IGS long product filename convention v2:
    https://files.igs.org/pub/resource/guidelines/Guidelines_For_Long_Product_Filenames_in_the_IGS_v2.0.pdf
    """

    FIN = SolutionType("FIN", "final")  # Final products
    NRT = SolutionType("NRT", "near-real time")  # Near-Real Time (between ULT and RTS)
    PRD = SolutionType("PRD", "predicted")  # Predicted products
    RAP = SolutionType("RAP", "rapid")  # Rapid products
    RTS = SolutionType("RTS", "real-time streamed")  # Real-Time streamed products
    SNX = SolutionType("SNX", "sinex combination")  # SINEX Combination product
    ULT = SolutionType("ULT", "ultra-rapid")  # Ultra-rapid products (every 6 hours)

    # Internal representation of unknown, for contexts where defaults are passed as strings.
    UNK = SolutionType("UNK", "unknown solution type")

    # To support search function below
    _all: list[SolutionType] = [FIN, NRT, PRD, RAP, RTS, SNX, ULT, UNK]

    @staticmethod
    def from_name(name: Optional[str]):
        """
        Returns the relevant static SolutionType object, given the solution type's short name (case insensitive).
        :param str name: The short name of the solution type e.g. 'RAP', 'ULT', 'FIN', 'SNX'. Also accepts unknwon, as
         either UNK or None
        """
        if name is None:  # None is analogous to UNK
            logging.debug("Converted solution type value of None, to SolutionTypes.UNK")
            return SolutionTypes.UNK

        name = name.upper()
        for solution_type in SolutionTypes._all:
            if name == solution_type.name:
                return solution_type
        raise ValueError(f"No known solution type with short name '{name}'")
