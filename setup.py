#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import os
import re

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with open(os.path.join(here, *parts), "r", encoding="utf8") as fp:
        return fp.read()


# Get package version
def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


requirements = [
    "matplotlib",
    "pandas",
    "numpy",
    "shapely",
    "scipy",
    "scipy",
    "psutil",
    "tzlocal",
]

extras_requirements = {"all": ["torch", "rpy2", "tzlocal"]}

this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="traja",
    version=find_version("traja", "__init__.py"),
    description="Traja is a trajectory analysis and visualization tool",
    url="https://github.com/justinshenk/traja",
    author="Justin Shenk",
    author_email="shenkjustin@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    extras_require=extras_requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">= 3.7",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    keywords="trajectory analysis",
    zip_safe=False,
)
