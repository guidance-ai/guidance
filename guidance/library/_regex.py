import sys

if sys.version_info >= (3, 11):
    import re._constants as constants  # type: ignore[import-not-found]
    import re._parser as parser  # type: ignore[import-not-found]
else:
    import sre_parse as parser
    import sre_constants as constants

from typing import Any, List, Tuple, Type, Union

from typing_extensions import TypeAlias

from .._grammar import Byte, Join, byte_range, select
from .._guidance import guidance
from ._any_char import any_char
from ._any_char_but import any_char_but
from ._optional import optional
from ._zero_or_more import zero_or_more

# Type aliases
Subpattern: TypeAlias = parser.SubPattern
Opcode: TypeAlias = constants._NamedIntConstant  # TODO: enum?
Argument: TypeAlias = Any
Node: TypeAlias = Tuple[Opcode, Argument]


class Transformer:

    @classmethod
    def transform(cls, tree: Union[Subpattern, Node]):
        if isinstance(tree, Subpattern):
            if len(tree.data) == 1:
                return cls.transform(tree.data[0])
            return Join([cls.transform(node) for node in tree.data])

        opcode, args = tree
        opcode_name = opcode.name
        try:
            method = getattr(cls, opcode_name)
        except AttributeError as e:
            raise NotImplementedError(
                f"No method implemented for opcode {opcode_name}"
            ) from e
        return method(args)

    @classmethod
    def SUBPATTERN(cls, args: Tuple[int, int, int, Subpattern]):
        # capture group
        # TODO: handle/capture?
        _, _, _, arg = args
        return cls.transform(arg)

    @classmethod
    def LITERAL(cls, args: int):
        # byte
        return Byte(args.to_bytes(length=1, byteorder="big"))

    @classmethod
    def RANGE(cls, args: Tuple[int, int]):
        # byte_range
        low, high = args
        return byte_range(
            low.to_bytes(length=1, byteorder="big"),
            high.to_bytes(length=1, byteorder="big"),
        )

    @classmethod
    def ANY(cls, _: Type[None]):
        return any_char()

    @classmethod
    def IN(cls, args: List[Node]):
        # char_set
        if args[0] == (constants.NEGATE, None):
            args.pop(0)
            if all([node[0] == constants.LITERAL for node in args]):
                bytes = [cls.transform(arg).byte for arg in args]
                return any_char_but(bytes)
            raise NotImplementedError(
                "Negation not implemented for non-literals (e.g. ranges)"
            )
        transformed_args = [cls.transform(arg) for arg in args]
        return select(transformed_args)

    @classmethod
    def BRANCH(cls, args: Tuple[Any, List[Subpattern]]):
        _, arg = args
        if _ is not None:
            raise NotImplementedError(
                "First time seeing BRANCH with non-None first arg"
            )
        transformed_args = [cls.transform(a) for a in arg]
        return select(transformed_args)

    @classmethod
    def MAX_REPEAT(
        cls, args: Tuple[int, Union[int, constants._NamedIntConstant], Subpattern]
    ):
        # TODO: type for this constants._NamedIntConstant? (not an opcode)
        low, high, arg = args
        transformed_arg = cls.transform(arg)
        if isinstance(high, constants._NamedIntConstant):
            if high != constants.MAXREPEAT:
                raise NotImplementedError(f"No handler for MAX_REPEAT with high={high}")
            if low == 0:
                # kleene star
                return zero_or_more(transformed_arg)
            if low > 0:
                return Join([transformed_arg] * low + [zero_or_more(transformed_arg)])
        if isinstance(high, int):
            return Join(
                [transformed_arg] * low + [optional(transformed_arg)] * (high - low)
            )
        raise TypeError(
            "high has type {type(high)}, expected one of int, constants._NamedIntConstant"
        )

    @classmethod
    def CATEGORY(cls, args: constants._NamedIntConstant):
        # TODO: type for this constants._NamedIntConstant? (not an opcode)
        if args.name == "CATEGORY_DIGIT":
            return byte_range(b"0", b"9")
        raise NotImplementedError(f"No implementation for category {args}")


@guidance(stateless=True)
def regex(lm, pattern):
    return lm + Transformer.transform(parser.parse(pattern))
