from __future__ import print_function, division, absolute_import

__version__ = "0.1.8"
__bibtex__ = """
@article{iyer2019non,
  title={Non-parametric Star Formation History Reconstruction with Gaussian Processes I: Counting Major Episodes of Star Formation},
  author={Iyer, Kartheik G and Gawiser, Eric and Faber, Sandra M and Ferguson, Henry C and Koekemoer, Anton M and Pacifici, Camilla and Somerville, Rachel},
  journal={arXiv preprint arXiv:1901.02877},
  year={2019}
}
"""  # NOQA

# from . import train_data
# from . import pregrids
# from . import filters

#print('Starting dense_basis. please wait ~ a minute for the FSPS backend to initialize.')

from .priors import *
from .gp_sfh import *
from .sed_fitter import *
from .pre_grid import *
from .tests import *
from .plotter import *
from .mcmc import *
from .parallelization import *
from .basic_fesc import *
