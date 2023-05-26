import os
import re
import codecs
from setuptools import setup, find_packages

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
    url="https://github.com/slundberg/guidance",
    author="Scott Lundberg and Marco Tulio Ribeiro",
    author_email="scott.lundberg@microsoft.com",
    description="A guidance language for controlling large language models.",
    long_description="Guidance enables you to control modern language models more effectively and efficiently than traditional prompting or chaining. Guidance programs allow you to interleave generation, prompting, and logical control into a single continuous flow matching how the language model actually processes the text.",
    packages=find_packages(exclude=["user_studies", "notebooks", "client"]),
    package_data={"guidance": ["resources/*"]},
    install_requires=[
        "diskcache",
        "openai>=0.27",
        "parsimonious",
        "pygtrie",
        "platformdirs",
        "tiktoken>=0.3",
        "nest_asyncio",
        "msal",
        "requests",
        "numpy"
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
            'pytest',
            'transformers',
            'torch'
        ]
    }
)
