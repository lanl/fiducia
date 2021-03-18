 .. _install:

*******************
Installing Fiducia
*******************

Requirements
============

Fiducia require Python version 3.7 or newer.
Fiducia also require the following openly available packages for installation:


- `NumPy <https://www.numpy.org/>`_ — 1.15.0 or newer
- `SciPy <https://www.scipy.org/>`_ — 1.1.0 or newer
- `pandas <https://pandas.pydata.org/>`_ — 0.23.0 or newer 
- `matplotlib <https://matplotlib.org/>`_ — 3.0.0 or newer
- `xarray <http://xarray.pydata.org>`_ — 0.15.1 or newer
- `Astropy <https://www.astropy.org/>`_ — 3.1 or newer


Installation with pip
=====================
`Official releases of Fiducia <https://pypi.org/project/fiducia/>`_ are
published to `pypi.org <https://pypi.org/>`_ and can simply be pip installed
like so:

.. code-block:: python

   pip install fiducia


Building and installing from source (for contributors)
======================================================
Make sure you have python installed, preferably via Anaconda
------------------------------------------------------------
Here is where you get Anaconda, and make sure to get the Python 3 version.
https://www.anaconda.com/distribution/

Setup installation directory
----------------------------
Make a directory called "fiducia" in a sensible place on your system. Preferably in a directory where none of the higher level directory names have spaces in them.

Setup a virtual environment
---------------------------
If you have python installed via Anaconda, then create your virtual environment like this

.. code-block:: bash

   conda create --name fiducia


Clone the repository using git
------------------------------
In the fiducia directory you created, run the following on the command line

.. code-block:: bash

   git clone https://github.com/lanl/fiducia.git


Activate your virtual environment
---------------------------------
Still on the command line, run

.. code-block:: bash

   source activate fiducia


Install requirements
--------------------

.. code-block:: bash

   pip install -r requirements.txt


Install fiducia
---------------
If you are a user then do

.. code-block:: bash

   pip install .


If you wish to help in developing fiducia, then do

.. code-block:: bash

   pip install -e .


Test if install was successful
------------------------------
Open a python and try doing ``import fiducia``. If all went well then you shouldn't get any error messages.
