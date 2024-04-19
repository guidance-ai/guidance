import os
import re
import codecs
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools_rust import Binding, RustExtension

here = os.path.abspath(os.path.dirname(__file__))

instal_requires = [
    "diskcache",
    "numpy",
    "openai>=1.0",
    "ordered_set",
    "platformdirs",
    "pyformlang",
    "protobuf",
    "requests",
    "tiktoken>=0.3",
]

# Our basic list of 'extras'
extras_requires = {
    "schemas": ["jsonschema", "pydantic"],
    "server": ["fastapi", "uvicorn"],
}

# Create the union of all our requirements
all_requires = set()
for v in extras_requires.values():
    all_requires = all_requires.union(v)

# Required for builds etc.
doc_requires = ["ipython", "nbsphinx", "numpydoc", "sphinx_rtd_theme", "sphinx"]
test_requires = [
    "jupyter",
    "papermill",
    "pytest",
    "pytest-cov",
    "torch",
    "transformers",
    "mypy==1.9.0",
    "types-protobuf",
    "types-regex",
    "types-requests",
    "types-jsonschema",
]


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
    ext_modules=[
        Pybind11Extension(
            "guidance.cpp", ["guidance/_cpp/main.cpp", "guidance/_cpp/byte_trie.cpp"]
        )
    ],
    rust_extensions=[
        RustExtension(
            "guidance._rust.guidancerust",
            "guidance/_rust/Cargo.toml",
            binding=Binding.PyO3,
        )
    ],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=instal_requires,
    extras_require={
        "all": all_requires,
        "docs": doc_requires,
        "test": test_requires,
        **extras_requires,
    },
)
