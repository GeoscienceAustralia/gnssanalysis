import logging
from gnssanalysis.enum_meta_properties import EnumMetaProperties

logging.basicConfig(format="%(asctime)s [%(funcName)s] %(levelname)s: %(message)s")


# Abstract base class. Leverages above Immutable metaclass to prevent its (effectively) constants, from being modified.
# Note that this doesn't prevent everything. For example, the contents of a list can still be changed.
class SolutionType(metaclass=EnumMetaProperties):
    name: str
    long_name: str

    def __init__(self):
        raise Exception("This is intended to act akin to an enum. Don't instantiate it.")


class FIN(SolutionType):
    """
    Final products
    """

    name = "FIN"
    long_name = "final"


class NRT(SolutionType):
    """
    Near-Real Time (between ULT and RTS)
    """

    name = "PRD"
    long_name = "near-real time"


class PRD(SolutionType):
    """
    Predicted products
    """

    name = "PRD"
    long_name = "predicted"


class RAP(SolutionType):
    """
    Rapid products
    """

    name = "RAP"
    long_name = "rapid"


class RTS(SolutionType):
    """
    Real-Time streamed products
    """

    name = "RTS"
    long_name = "real-time streamed"


class SNX(SolutionType):
    """
    SINEX Combination product
    """

    name = "SNX"
    long_name = "sinex combination"


class ULT(SolutionType):
    """
    Ultra-rapid products
    The only orbit product from IGS which isn't a 1 day span
    """

    name = "ULT"
    long_name = "ultra-rapid"


class UNK(SolutionType):
    """
    Internal representation of an unknown solution type.
    """

    name = "UNK"
    long_name = "unknown solution type"


class SolutionTypes(metaclass=EnumMetaProperties):
    """
    Defines valid solution type identifiers specified for use in the IGS long product filename convention v2:
    https://files.igs.org/pub/resource/guidelines/Guidelines_For_Long_Product_Filenames_in_the_IGS_v2.0.pdf

    Also see here for information on session lengths of products pubished by IGS: https://igs.org/products/#about
    """

    def __init__(self):
        raise Exception("This is intended to act akin to an enum. Don't instantiate it.")

    FIN = FIN  # Final products
    NRT = NRT  # Near-Real Time (between ULT and RTS)
    PRD = PRD  # Predicted products
    RAP = RAP  # Rapid products
    RTS = RTS  # Real-Time streamed products
    SNX = SNX  # SINEX Combination product
    ULT = ULT  # Ultra-rapid products (every 6 hours). The only orbit product from IGS which isn't a 1 day span
    UNK = UNK  # Internal representation of unknown. Useful in contexts where defaults are passed as strings.

    # To support search function below
    _all: list[type[SolutionType]] = [FIN, NRT, PRD, RAP, RTS, SNX, ULT, UNK]

    @staticmethod
    def from_name(name: str):
        """
        Returns the relevant static SolutionType object, given the solution type's short name (case insensitive).
        :param str name: The short name of the solution type e.g. 'RAP', 'ULT', 'FIN', 'SNX'. Though not part of the
         official standard, 'UNK' can also be used to indicate an unknown solution type.
        """
        if name is None or len(name.strip()) == 0:
            raise ValueError("Solution type name passed was None or effectively empty!", name)
        if len(name) > 3:
            raise ValueError("Long solution type names are not supported here. Please use RAP, ULT, etc.", name)
        name = name.upper()
        for solution_type in SolutionTypes._all:
            if name == solution_type.name:
                return solution_type
        raise ValueError(f"No known solution type with short name '{name}'")
