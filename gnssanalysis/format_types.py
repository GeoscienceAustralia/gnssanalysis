import logging
from gnssanalysis.enum_meta_properties import EnumMetaProperties

logging.basicConfig(format="%(asctime)s [%(funcName)s] %(levelname)s: %(message)s")


# Abstract base class. Leverages above Immutable metaclass to prevent its (effectively) constants, from being modified.
# Note that this doesn't prevent everything. For example, the contents of a list can still be changed.
class FormatType(metaclass=EnumMetaProperties):
    name: str
    long_name: str

    def __init__(self):
        raise Exception("This is intended to act akin to an enum. Don't instantiate it.")


class BIA(FormatType):
    """
    bias SINEX
    """

    name = "BIA"
    long_name = "bias SINEX"


class CLK(FormatType):
    """
    clock RINEX
    """

    name = "CLK"
    long_name = "clock RINEX"


class ERP(FormatType):
    """
    IGS ERP (Earth Rotation Parameter) format
    """

    name = "ERP"
    long_name = "IGS ERP (Earth Rotation Parameter) format"


class IOX(FormatType):  # TODO not IGS official!
    """
    TODO
    """

    name = "IOX"
    long_name = ""  # TODO


class INX(FormatType):  # TODO not defined in Ginan
    """
    IONEX ionospheric TEC grid product format
    """

    name = "INX"
    long_name = "IONEX ionospheric TEC grid product format"


class JSON(FormatType):  # TODO not defined in Ginan
    """
    JSON file
    """

    name = "JSON"
    long_name = "JSON file"


class OBX(FormatType):
    """
    ORBEX satellite orbit/attitude format
    """

    name = "OBX"
    long_name = "ORBEX satellite orbit/attitude format"


class SNX(FormatType):
    """
    Solution INdependent EXchange (SINEX) format file
    """

    name = "SNX"
    long_name = "Solution INdependent EXchange (SINEX) format file"


class SP3(FormatType):
    """
    Standard Product 3 (SP3) orbit format
    """

    name = "SP3"
    long_name = "Standard Product 3 (SP3) orbit format"


class SUM(FormatType):
    """
    Summary of the indicated product, combination summary, etc
    """

    name = "SUM"
    long_name = "Summary of the indicated product, combination summary, etc"


class TRO(FormatType):
    """
    SINEX_TRO product format
    """

    name = "TRO"
    long_name = "SINEX_TRO product format"


class YAML(FormatType):  # TODO not defined in Ginan
    """
    YAML file
    """

    name = "YAML"  # NOTE: can also be entered as YML
    long_name = "YAML file"


class UNK(FormatType):
    """
    Internal representation of an unknown format type.
    """

    name = "UNK"
    long_name = "unknown format file"


class FormatTypes(metaclass=EnumMetaProperties):
    """
    Defines valid file formats as specified in section 2.5 of the IGS long product filenames specificiation v2:
    https://files.igs.org/pub/resource/guidelines/Guidelines_For_Long_Product_Filenames_in_the_IGS_v2.0.pdf
    """

    def __init__(self):
        raise Exception("This is intended to act akin to an enum. Don't instantiate it.")

    BIA = BIA
    CLK = CLK
    ERP = ERP
    IOX = IOX
    INX = INX
    JSON = JSON
    OBX = OBX
    SNX = SNX
    SP3 = SP3
    SUM = SUM
    TRO = TRO
    YAML = YAML
    UNK = UNK  # Internal representation of unknown. Useful in contexts where defaults are passed as strings.

    # To support search function below
    _all: list[type[FormatType]] = [BIA, CLK, ERP, IOX, INX, JSON, OBX, SNX, SP3, SUM, TRO, YAML, UNK]
    _supported_types: list[type[FormatType]] = [CLK, ERP, SNX, SP3]

    @staticmethod
    def is_supported(format_type: type[FormatType]) -> bool:
        return format_type in FormatTypes._supported_types

    @staticmethod
    def supported_types() -> list[type[FormatType]]:
        return FormatTypes._supported_types

    @staticmethod
    def from_short_name(name: str):
        """
        Returns the relevant static FormatType object, given the format's short name (case insensitive).
        :param str name: The short name of the format type e.g. 'SP3', 'CLK', 'SNX', etc. Though not part of the
         official standard, 'UNK' can also be used to indicate an unknown format type.
        """
        if name is None or len(name.strip()) == 0:
            raise ValueError("File format short name passed was None or effectively empty!", name)
        if len(name) > 4:
            raise ValueError("File format short names must not exceed 4 chars.", name)
        name = name.upper()

        if name in ("YML", "YAML"):  # YAML can be referred to in two ways
            return FormatTypes.YAML
        for format_type in FormatTypes._all:
            if name == format_type.name:
                return format_type
        raise ValueError(f"No known format type with short name '{name}'")
