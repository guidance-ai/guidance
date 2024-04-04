import os
import re
import codecs
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name="guidance",
    version=find_version("guidance", "__init__.py"),
    url="https://github.com/guidance-ai/guidance",
    author="Guidance Maintainers",
    author_email="maintainers@guidance-ai.org",
    description="A guidance language for controlling large language models.",
    long_description="Guidance enables you to control modern language models more effectively and efficiently than traditional prompting or chaining. Guidance programs allow you to interleave generation, prompting, and logical control into a single continuous flow matching how the language model actually processes the text.",
    packages=find_packages(exclude=["notebooks", "client", "tests", "tests.*"]),
    package_data={"guidance": ["resources/*"]},
    ext_modules=[Pybind11Extension("guidance.cpp", ["guidance/_cpp/main.cpp", "guidance/_cpp/byte_trie.cpp"])],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "diskcache",
        "openai>=1.0",
        "platformdirs",
        "tiktoken>=0.3",
        "requests",
        "numpy",
        "ordered_set",
        "pyformlang",
        "protobuf",
        "fastapi",
        "uvicorn"
    ],
    extras_require={
        'docs': [
            'ipython',
            'numpydoc',
            'sphinx_rtd_theme',
            'sphinx',
            'nbsphinx'
        ],
        'test': [
            "jsonschema",
            "jupyter",
            "papermill",
            'pytest',
            'pytest-cov',
            'torch',
            'transformers',
        ]
    },
    
)