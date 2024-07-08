"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""

import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing.final
class Grammar(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NODES_FIELD_NUMBER: builtins.int
    @property
    def nodes(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___GrammarFunction]: ...
    def __init__(
        self,
        *,
        nodes: collections.abc.Iterable[global___GrammarFunction] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["nodes", b"nodes"]) -> None: ...

global___Grammar = Grammar

@typing.final
class EngineCallResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing.final
    class CaptureGroupsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        @property
        def value(self) -> global___Value: ...
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: global___Value | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing.Literal["value", b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing.Literal["key", b"key", "value", b"value"]) -> None: ...

    @typing.final
    class CaptureGroupLogProbsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        @property
        def value(self) -> global___Value: ...
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: global___Value | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing.Literal["value", b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing.Literal["key", b"key", "value", b"value"]) -> None: ...

    NEW_BYTES_FIELD_NUMBER: builtins.int
    IS_GENERATED_FIELD_NUMBER: builtins.int
    NEW_BYTES_PROB_FIELD_NUMBER: builtins.int
    CAPTURE_GROUPS_FIELD_NUMBER: builtins.int
    CAPTURE_GROUP_LOG_PROBS_FIELD_NUMBER: builtins.int
    NEW_TOKEN_COUNT_FIELD_NUMBER: builtins.int
    new_bytes: builtins.bytes
    is_generated: builtins.bool
    new_bytes_prob: builtins.float
    new_token_count: builtins.int
    @property
    def capture_groups(self) -> google.protobuf.internal.containers.MessageMap[builtins.str, global___Value]: ...
    @property
    def capture_group_log_probs(self) -> google.protobuf.internal.containers.MessageMap[builtins.str, global___Value]: ...
    def __init__(
        self,
        *,
        new_bytes: builtins.bytes = ...,
        is_generated: builtins.bool = ...,
        new_bytes_prob: builtins.float = ...,
        capture_groups: collections.abc.Mapping[builtins.str, global___Value] | None = ...,
        capture_group_log_probs: collections.abc.Mapping[builtins.str, global___Value] | None = ...,
        new_token_count: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["capture_group_log_probs", b"capture_group_log_probs", "capture_groups", b"capture_groups", "is_generated", b"is_generated", "new_bytes", b"new_bytes", "new_bytes_prob", b"new_bytes_prob", "new_token_count", b"new_token_count"]) -> None: ...

global___EngineCallResponse = EngineCallResponse

@typing.final
class Value(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STRING_VALUE_FIELD_NUMBER: builtins.int
    BYTES_VALUE_FIELD_NUMBER: builtins.int
    FLOAT_VALUE_FIELD_NUMBER: builtins.int
    LIST_VALUE_FIELD_NUMBER: builtins.int
    string_value: builtins.str
    bytes_value: builtins.bytes
    float_value: builtins.float
    @property
    def list_value(self) -> global___ListValue: ...
    def __init__(
        self,
        *,
        string_value: builtins.str = ...,
        bytes_value: builtins.bytes = ...,
        float_value: builtins.float = ...,
        list_value: global___ListValue | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["bytes_value", b"bytes_value", "float_value", b"float_value", "kind", b"kind", "list_value", b"list_value", "string_value", b"string_value"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["bytes_value", b"bytes_value", "float_value", b"float_value", "kind", b"kind", "list_value", b"list_value", "string_value", b"string_value"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["kind", b"kind"]) -> typing.Literal["string_value", "bytes_value", "float_value", "list_value"] | None: ...

global___Value = Value

@typing.final
class ListValue(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALUES_FIELD_NUMBER: builtins.int
    @property
    def values(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Value]: ...
    def __init__(
        self,
        *,
        values: collections.abc.Iterable[global___Value] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["values", b"values"]) -> None: ...

global___ListValue = ListValue

@typing.final
class Byte(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    BYTE_FIELD_NUMBER: builtins.int
    HIDDEN_FIELD_NUMBER: builtins.int
    COMMIT_POINT_FIELD_NUMBER: builtins.int
    NULLABLE_FIELD_NUMBER: builtins.int
    CAPTURE_NAME_FIELD_NUMBER: builtins.int
    TEMPERATURE_FIELD_NUMBER: builtins.int
    byte: builtins.bytes
    hidden: builtins.bool
    commit_point: builtins.bool
    nullable: builtins.bool
    capture_name: builtins.str
    temperature: builtins.float
    def __init__(
        self,
        *,
        byte: builtins.bytes = ...,
        hidden: builtins.bool = ...,
        commit_point: builtins.bool = ...,
        nullable: builtins.bool = ...,
        capture_name: builtins.str = ...,
        temperature: builtins.float = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["byte", b"byte", "capture_name", b"capture_name", "commit_point", b"commit_point", "hidden", b"hidden", "nullable", b"nullable", "temperature", b"temperature"]) -> None: ...

global___Byte = Byte

@typing.final
class ByteRange(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    BYTE_RANGE_FIELD_NUMBER: builtins.int
    HIDDEN_FIELD_NUMBER: builtins.int
    COMMIT_POINT_FIELD_NUMBER: builtins.int
    CAPTURE_NAME_FIELD_NUMBER: builtins.int
    TEMPERATURE_FIELD_NUMBER: builtins.int
    byte_range: builtins.bytes
    hidden: builtins.bool
    commit_point: builtins.bool
    capture_name: builtins.str
    temperature: builtins.float
    def __init__(
        self,
        *,
        byte_range: builtins.bytes = ...,
        hidden: builtins.bool = ...,
        commit_point: builtins.bool = ...,
        capture_name: builtins.str = ...,
        temperature: builtins.float = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["byte_range", b"byte_range", "capture_name", b"capture_name", "commit_point", b"commit_point", "hidden", b"hidden", "temperature", b"temperature"]) -> None: ...

global___ByteRange = ByteRange

@typing.final
class Null(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___Null = Null

@typing.final
class ModelVariable(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAME_FIELD_NUMBER: builtins.int
    HIDDEN_FIELD_NUMBER: builtins.int
    COMMIT_POINT_FIELD_NUMBER: builtins.int
    CAPTURE_NAME_FIELD_NUMBER: builtins.int
    NULLABLE_FIELD_NUMBER: builtins.int
    name: builtins.str
    hidden: builtins.bool
    commit_point: builtins.bool
    capture_name: builtins.str
    nullable: builtins.bool
    def __init__(
        self,
        *,
        name: builtins.str = ...,
        hidden: builtins.bool = ...,
        commit_point: builtins.bool = ...,
        capture_name: builtins.str = ...,
        nullable: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["capture_name", b"capture_name", "commit_point", b"commit_point", "hidden", b"hidden", "name", b"name", "nullable", b"nullable"]) -> None: ...

global___ModelVariable = ModelVariable

@typing.final
class Join(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NULLABLE_FIELD_NUMBER: builtins.int
    VALUES_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    HIDDEN_FIELD_NUMBER: builtins.int
    COMMIT_POINT_FIELD_NUMBER: builtins.int
    CAPTURE_NAME_FIELD_NUMBER: builtins.int
    MAX_TOKENS_FIELD_NUMBER: builtins.int
    nullable: builtins.bool
    name: builtins.str
    hidden: builtins.bool
    commit_point: builtins.bool
    capture_name: builtins.str
    max_tokens: builtins.int
    @property
    def values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """Use a repeated field to store the list of values"""

    def __init__(
        self,
        *,
        nullable: builtins.bool = ...,
        values: collections.abc.Iterable[builtins.int] | None = ...,
        name: builtins.str = ...,
        hidden: builtins.bool = ...,
        commit_point: builtins.bool = ...,
        capture_name: builtins.str = ...,
        max_tokens: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["capture_name", b"capture_name", "commit_point", b"commit_point", "hidden", b"hidden", "max_tokens", b"max_tokens", "name", b"name", "nullable", b"nullable", "values", b"values"]) -> None: ...

global___Join = Join

@typing.final
class Select(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NULLABLE_FIELD_NUMBER: builtins.int
    VALUES_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    HIDDEN_FIELD_NUMBER: builtins.int
    COMMIT_POINT_FIELD_NUMBER: builtins.int
    CAPTURE_NAME_FIELD_NUMBER: builtins.int
    MAX_TOKENS_FIELD_NUMBER: builtins.int
    RECURSIVE_FIELD_NUMBER: builtins.int
    nullable: builtins.bool
    name: builtins.str
    hidden: builtins.bool
    commit_point: builtins.bool
    capture_name: builtins.str
    max_tokens: builtins.int
    recursive: builtins.bool
    @property
    def values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """Use a repeated field to store the list of values"""

    def __init__(
        self,
        *,
        nullable: builtins.bool = ...,
        values: collections.abc.Iterable[builtins.int] | None = ...,
        name: builtins.str = ...,
        hidden: builtins.bool = ...,
        commit_point: builtins.bool = ...,
        capture_name: builtins.str = ...,
        max_tokens: builtins.int = ...,
        recursive: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["capture_name", b"capture_name", "commit_point", b"commit_point", "hidden", b"hidden", "max_tokens", b"max_tokens", "name", b"name", "nullable", b"nullable", "recursive", b"recursive", "values", b"values"]) -> None: ...

global___Select = Select

@typing.final
class GrammarFunction(google.protobuf.message.Message):
    """message Terminal {
        oneof function_type {
            Byte byte = 1;
            ByteRange byte_range = 2;
        }
    }
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    JOIN_FIELD_NUMBER: builtins.int
    SELECT_FIELD_NUMBER: builtins.int
    BYTE_FIELD_NUMBER: builtins.int
    BYTE_RANGE_FIELD_NUMBER: builtins.int
    MODEL_VARIABLE_FIELD_NUMBER: builtins.int
    @property
    def join(self) -> global___Join: ...
    @property
    def select(self) -> global___Select: ...
    @property
    def byte(self) -> global___Byte: ...
    @property
    def byte_range(self) -> global___ByteRange: ...
    @property
    def model_variable(self) -> global___ModelVariable: ...
    def __init__(
        self,
        *,
        join: global___Join | None = ...,
        select: global___Select | None = ...,
        byte: global___Byte | None = ...,
        byte_range: global___ByteRange | None = ...,
        model_variable: global___ModelVariable | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["byte", b"byte", "byte_range", b"byte_range", "function_type", b"function_type", "join", b"join", "model_variable", b"model_variable", "select", b"select"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["byte", b"byte", "byte_range", b"byte_range", "function_type", b"function_type", "join", b"join", "model_variable", b"model_variable", "select", b"select"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["function_type", b"function_type"]) -> typing.Literal["join", "select", "byte", "byte_range", "model_variable"] | None: ...

global___GrammarFunction = GrammarFunction