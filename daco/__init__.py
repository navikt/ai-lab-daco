from daco.main import main
from daco.privacy import privacy
from daco.plot import plot
# from daco.miscellaneous import *

__doc__ = """
daco - a tool for comparing datasets
====================================

**daco** is a Python package designed for comparing datasetes statistically.

Main Features
-------------
- Plot distributions and correlation matrices
- Compare distributions and check the relative differences
- Calculate the statistical differences and distances between the datasets
- Compare datasets by using machine learning
"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__all__ = ['main', 'plot', 'privacy']
