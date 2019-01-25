#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import os
from os import path

# Get package version
exec(open('traja/version.py', 'r').read())

requirements = ['matplotlib','pandas','numpy','seaborn', 'shapely','psutil', 'scipy']

this_dir = path.abspath(path.dirname(__file__))
with open(os.path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='traja',
    version=__version__,
    description='Traja is a trajectory analysis and visualization tool',
    url='https://github.com/justinshenk/traja',
    author='Justin Shenk',
    author_email='shenkjustin@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT license',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    python_requires='!= 3.0.*, != 3.1.*',
    packages=find_packages(),
    include_package_data=True,
    keywords='trajectory analysis',
    zip_safe=False,
)
