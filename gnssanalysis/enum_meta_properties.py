class EnumMetaProperties(type):
    """
    This metaclass:
     - intercepts attempts to set *class* attributes, and rejects them.
       - NOTE: In the class or abstract class using this, you should also define an __init__() which raises
         an exception, to prevent instantiation.
     - defines the class string representation as being *just* the class name, without any fluff.

    Loosely based on carefully reviewed AI generated examples from Microsoft Copilot.
    """

    def __setattr__(cls, name: str, value) -> None:
        raise AttributeError(f"Attributes of {cls} act as constants. Do not modify them.")

    def __repr__(cls) -> str:
        return f"{cls.__name__}"
