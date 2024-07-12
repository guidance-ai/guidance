import sys

if sys.version_info >= (3, 11):
    import re._constants as constants  # type: ignore[import-not-found]
    import re._parser as parser  # type: ignore[import-not-found]
else:
    import sre_parse as parser
    import sre_constants as constants

from re import RegexFlag
from typing import Any, List, Tuple, Union

from typing_extensions import TypeAlias

from .._grammar import Byte, ByteRange, Join, Select, byte_range, select, capture
from .._guidance import guidance
from ._any_char_but import any_char_but
from ._sequences import sequence

# Type aliases
Node: TypeAlias = Tuple[constants._NamedIntConstant, Any]


class UnsupportedRegexError(Exception):
    pass


class RegexPatternConverter:

    @classmethod
    def parse(cls, pattern: str):
        return cls.convert(parser.parse(pattern))

    @classmethod
    def convert(cls, tree: Union[parser.SubPattern, Node], flags: int = 0):
        if flags != 0:
            # Equivalent to re.NOFLAG
            raise UnsupportedRegexError(
                f"Flags other than re.NOFLAG not supported; got {RegexFlag(flags)}"
            )
        if isinstance(tree, parser.SubPattern):
            if len(tree.data) == 1:
                return cls.convert(tree.data[0])
            return Join([cls.convert(node) for node in tree.data])

        opcode, args = tree
        opcode_name = opcode.name
        try:
            method = getattr(cls, opcode_name)
        except AttributeError as e:
            raise UnsupportedRegexError(
                f"Unsupported regex feature with opcode {opcode_name}"
            ) from e
        return method(args)

    @classmethod
    def SUBPATTERN(cls, args: Tuple[int, int, int, parser.SubPattern]):
        # capture group
        group, add_flags, del_flags, arg = args
        flags = add_flags & ~del_flags
        return cls.convert(arg, flags)

    @classmethod
    def LITERAL(cls, args: int):
        # byte
        return Byte(bytes([args]))

    @classmethod
    def NOT_LITERAL(cls, args: int):
        return any_char_but(chr(args))

    @classmethod
    def RANGE(cls, args: Tuple[int, int]):
        # byte_range
        low, high = args
        return byte_range(bytes([low]), bytes([high]))

    @classmethod
    def ANY(cls, _: None):
        return any_char_but("\n")

    @classmethod
    def IN(cls, args: List[Node]):
        if args[0][0] == constants.NEGATE:
            transformed_args = [cls.convert(arg) for arg in args[1:]]
            negated_bytes = cls._get_negated_bytes(transformed_args)
            return any_char_but(negated_bytes)
        transformed_args = [cls.convert(arg) for arg in args]
        return select(transformed_args)

    @classmethod
    def _get_negated_bytes(cls, grammars: List[Union[Byte, ByteRange, Select]]):
        negated_bytes = set()
        for value in grammars:
            if isinstance(value, Byte):
                negated_bytes.add(value.byte)
            elif isinstance(value, ByteRange):
                low, high = value.byte_range
                negated_bytes.update([bytes([i]) for i in range(low, high + 1)])
            elif isinstance(value, Select):
                negated_bytes.update(cls._get_negated_bytes(value._values))
            else:
                raise TypeError(f"Can't negate {type(value)} object")
        return negated_bytes

    @classmethod
    def BRANCH(cls, args: Tuple[Any, List[parser.SubPattern]]):
        unknown, arg = args
        if unknown is not None:
            # Unsure of the semantics of this value, but it seems to be
            # None in all cases tested so far
            raise UnsupportedRegexError(f"Unkwnown argument in BRANCH: {unknown}")
        transformed_args = [cls.convert(a) for a in arg]
        return select(transformed_args)

    @classmethod
    def MAX_REPEAT(
        cls,
        args: Tuple[int, Union[int, constants._NamedIntConstant], parser.SubPattern],
    ):
        low, high, arg = args
        transformed_arg = cls.convert(arg)
        if isinstance(high, constants._NamedIntConstant):
            if high != constants.MAXREPEAT:
                raise UnsupportedRegexError(f"Unsupported high value in range: {high}")
            return sequence(transformed_arg, min_length=low)
        return sequence(transformed_arg, min_length=low, max_length=high)

    @classmethod
    def CATEGORY(cls, args: constants._NamedIntConstant):
        # \d
        if args.name == "CATEGORY_DIGIT":
            return cls.parse(r"[0-9]")
        # \D
        if args.name == "CATEGORY_NOT_DIGIT":
            return cls.parse(r"[^0-9]")
        # \w
        if args.name == "CATEGORY_WORD":
            return cls.parse(r"[0-9A-Za-z_]")
        # \W
        if args.name == "CATEGORY_NOT_WORD":
            return cls.parse(r"[^0-9A-Za-z_]")
        # \s
        if args.name == "CATEGORY_SPACE":
            return cls.parse(r"[ \t\n\r\f\v]")
        # \S
        if args.name == "CATEGORY_NOT_SPACE":
            return cls.parse(r"[^ \t\n\r\f\v]")
        raise UnsupportedRegexError(f"Unsupported category: {args.name}")


@guidance(stateless=True)
def regex(lm, pattern, *, name=None):
    return lm + capture(RegexPatternConverter.parse(pattern), name=name)
