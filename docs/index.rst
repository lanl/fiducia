.. fiducia documentation master file, created by
   sphinx-quickstart on Tue Mar 16 15:24:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Fiducia's Documentation
===================================

Fiducia is an open source package for unfolding spectral information from
filtered diode array diagnostics (such as Dante) using the `cubic splines analysis method <https://doi.org/10.1063/5.0002856>`_ . This method simply
assumes that the underlying spectrum is smoothly varying, and does not
impose any other constraints on the shapes of spectrum. See below for
instructions on how to install Fiducia, and for examples on how to run an
analysis using Fiducia.

.. toctree::
   :caption: First Steps
   :maxdepth: 1

   Installing <install>
   Examples <auto_examples/index>
   Contributing <contributing>
   Citing <citing>
   License <license>

.. toctree::
   :maxdepth: 1
   :caption: Submodules

   cspline
   error
   loader
   main
   misc
   pltDefaults
   rawProcess
   response
   stats
   visualization
 

.. _toplevel-development-guide:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
