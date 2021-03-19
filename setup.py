#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 16:11:09 2019

Fiducia setup script.

@author: Pawel M. Kozlowski
"""

from setuptools import setup


# Read the contents of the README file to include in the long
# description. The long description then becomes part of the pypi.org
# page.
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='fiducia',
      version='0.2.3',
      description='Filtered Diode Unfolder (using) Cubic-spline Algorithm',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/lanl/fiducia',
      author='Pawel M. Kozlowski, et al.',
      author_email='pkozlowski@lanl.gov',
      license='BSD',
      packages=['fiducia'],
      zip_safe=True)