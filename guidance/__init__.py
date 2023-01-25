__version__ = "0.0.17"

from ._prompt import Prompt
from . import generators
from . import library

default_generator = generators.OpenAI()