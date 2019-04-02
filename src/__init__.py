from daco.daco_main import daco

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__doc__ = """
daco - a tool for comparing datasets
====================================

**daco** is a Python package designed for comparing datasetes statistically.

Main Features
-------------
- Plot distributions and correlation matrices
- Compare distributions and check the relative differences
- Calculate the statistical differences and distances between the datasets
"""
