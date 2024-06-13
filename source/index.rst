Welcome to QuPyt's documentation!
=================================

QuPyt aims to be general purpose lab software. While QuPyt is highly flexible, it's main focus is on measurements with the following general setup:

#. A sensor to record some value of interest.
#. Zero to multipel signal sources interacting with a sample. These signal sources can either emit the same value for the entire measurement
   or update their value for every measurement iteration. How these values are updated is entirely configurable.
#. A synchronising device to ensure proper timing of all updates, readouts and interactions. This synchronising can also be used
   to play elaborate pulse sequences.

QuPyt was designed to perform complex measurements using modern Qunatum Sensors.
However, we believe that it can be applied to a wide range of interesting experiments.

We expect most of our users to be scientists from various backgrounds, without a heavy programming background.
We will therefore keep the documentation verbose and provide lots of example code.

If you have questions or suggestions or would like to contribute,
head over to our `github <https://github.com/KarDB/QuPyt.git>`_! Contributions and discussions are always welcome.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   measurement_specification
   api_reference/api_reference
   example_configs/example_configs
   example_pulse_sequences/example_pulse_sequences



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
