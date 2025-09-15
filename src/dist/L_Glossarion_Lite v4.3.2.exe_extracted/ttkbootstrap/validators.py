import re
from typing import Callable


class Validators:
    """Predefined static validators for use with Tkinter Entry widgets."""

    @staticmethod
    def is_numeric() -> Callable[[str], bool]:
        return lambda value: value.isdigit() or value == ""

    @staticmethod
    def is_alpha() -> Callable[[str], bool]:
        return lambda value: value.isalpha() or value == ""

    @staticmethod
    def is_alphanumeric() -> Callable[[str], bool]:
        return lambda value: value.isalnum() or value == ""

    @staticmethod
    def max_length(n: int) -> Callable[[str], bool]:
        return lambda value: len(value) <= n

    @staticmethod
    def min_length(n: int) -> Callable[[str], bool]:
        return lambda value: len(value) >= n or value == ""

    @staticmethod
    def is_email() -> Callable[[str], bool]:
        pattern = re.compile(r"^[^@]+@[^@]+\.[^@]+$")
        return lambda value: bool(pattern.match(value)) or value == ""

    @staticmethod
    def is_float() -> Callable[[str], bool]:
        return lambda value: value == "" or re.fullmatch(r"^-?\d*\.?\d*$", value) is not None

    @staticmethod
    def is_integer() -> Callable[[str], bool]:
        return lambda value: value == "" or re.fullmatch(r"^-?\d+$", value) is not None

    @staticmethod
    def custom(regex: str) -> Callable[[str], bool]:
        pattern = re.compile(regex)
        return lambda value: bool(pattern.fullmatch(value)) or value == ""
