import sys

if sys.version_info >= (3, 11):
    import re._constants as constants  # type: ignore[import-not-found]
    import re._parser as parser  # type: ignore[import-not-found]
else:
    import sre_parse as parser
    import sre_constants as constants

from typing import Any, List, Tuple, Union

from typing_extensions import TypeAlias

from .._grammar import Byte, ByteRange, Join, byte_range, select
from .._guidance import guidance
from ._any_char_but import any_char_but
from ._optional import optional
from ._zero_or_more import zero_or_more

# Type aliases
Node: TypeAlias = Tuple[constants._NamedIntConstant, Any]


class Transformer:

    @classmethod
    def transform(cls, tree: Union[parser.SubPattern, Node]):
        if isinstance(tree, parser.SubPattern):
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
    def SUBPATTERN(cls, args: Tuple[int, int, int, parser.SubPattern]):
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
    def ANY(cls, _: None):
        # TODO: introduce re flags, e.g. DOTALL?
        return any_char_but("\n")

    @classmethod
    def IN(cls, args: List[Node]):
        # char_set
        if args[0] == (constants.NEGATE, None):
            args.pop(0)
            excluded_bytes = set()
            for node in args:
                value = cls.transform(node)
                if isinstance(value, Byte):
                    excluded_bytes.add(value.byte)
                elif isinstance(value, ByteRange):
                    low, high = value.byte_range
                    excluded_bytes.update([bytes([i]) for i in range(low, high + 1)])
                else:
                    raise NotImplementedError(
                        f"No NEGATE handler for type {type(value)} (opcode was {node[0]})"
                    )
            return any_char_but(excluded_bytes)
        transformed_args = [cls.transform(arg) for arg in args]
        return select(transformed_args)

    @classmethod
    def BRANCH(cls, args: Tuple[Any, List[parser.SubPattern]]):
        _, arg = args
        if _ is not None:
            raise NotImplementedError(
                "First time seeing BRANCH with non-None first arg"
            )
        transformed_args = [cls.transform(a) for a in arg]
        return select(transformed_args)

    @classmethod
    def MAX_REPEAT(
        cls,
        args: Tuple[int, Union[int, constants._NamedIntConstant], parser.SubPattern],
    ):
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
        return Join(
            [transformed_arg] * low + [optional(transformed_arg)] * (high - low)
        )

    @classmethod
    def CATEGORY(cls, args: constants._NamedIntConstant):
        if args.name == "CATEGORY_DIGIT":
            return byte_range(b"0", b"9")
        raise NotImplementedError(f"No implementation for category {args}")


@guidance(stateless=True)
def regex(lm, pattern):
    return lm + Transformer.transform(parser.parse(pattern))
