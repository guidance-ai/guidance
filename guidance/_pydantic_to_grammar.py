from typing import Union

import pydantic

from ._grammar import GrammarFunction
from ._json_schema_to_grammar import _json_schema_obj_to_grammar


def pydantic_model_to_grammar(
    model: Union[type, pydantic.BaseModel]
) -> GrammarFunction:
    # Rather than 'type' I think it should be pydantic._internal._model_construction.ModelMetaclass
    json_schema = model.model_json_schema()

    return _json_schema_obj_to_grammar(json_schema)
