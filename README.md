# fiducia

[![Documentation Status](https://readthedocs.org/projects/fiducia/badge/?version=latest)](https://fiducia.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://fiducia.readthedocs.io/en/latest/license.html)
[![GitHub Actions — CI](https://github.com/lanl/fiducia/workflows/CI/badge.svg)](https://github.com/lanl/fiducia/actions?query=workflow%3ACI+branch%3Amain)
[![codecov](https://codecov.io/gh/lanl/fiducia/branch/main/graph/badge.svg)](https://codecov.io/gh/lanl/fiducia)
[![GitHub Actions — Style linters](https://github.com/lanl/fiducia/workflows/Style%20linters/badge.svg)](https://github.com/lanl/fiducia/actions?query=workflow%3AStyle-linters+branch%3Amain)

FIDUCIA (C20073): Filtered Diode Unfolder (using) Cubic-spline Algorithm

This is a diode array signal deconvolver based on the [cubic splines method](https://doi.org/10.1063/5.0002856). 
Given time-resolved diode data (e.g. Dante), this code can produce time-resolve x-ray 
spectra, and radiation temperatures. Functionality is described in [Fiducia's online documentation](https://fiducia.readthedocs.io/).


## Installation
[Official releases of Fiducia](https://pypi.org/project/fiducia/) are published to pypi.org and can simply be pip installed like so:
```
pip install fiducia
```

More detailed installation instructions can be found [here](https://fiducia.readthedocs.io/en/latest/install.html).


## License
Fiducia is released under a [3-clause BSD license](https://fiducia.readthedocs.io/en/latest/license.html).

## Citing Fiducia
If you use Fiducia in your work, please follow the best practices for citing Fiducia, which can be found [here](https://fiducia.readthedocs.io/en/latest/citing.html).

## Acknowledgements
Development of Fiducia was supported by the U.S. Department of Energy, and the NNSA.
