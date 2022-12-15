__version__ = "0.0.15"

from ._prompt import Prompt
from . import generators
from . import library

default_generator = generators.OpenAI()