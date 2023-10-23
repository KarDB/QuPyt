Example Configuration Files
===========================

QuPyt executes measurements by following instructions contained in a measurement configuration file.
These files can be handed to QuPyt at any time, once the programs CLI tool ``qupyt`` has been started, by copying or saving
the file to the ``~/.qupyt/waiging_room/`` folder. 

Conceptually, the measurement configuration is separated into two steps.

1) Selecting which hardware devices to use, and how to configure them. This includes e.g. the frequency and amplitude of a signal source,
the region of interest of a camera, the input channel of a data acquisition unit (DAQ) and so fourth. 
2) The pulse sequence, which is to be played by the synchronisers. It is used to execute measurement instructions
on a very short time scale (picoseconds to nanaoseconds) at a faster rate than a computer could manage. 

Below you find example configuration files:

.. toctree::
   :maxdepth: 1

   esr
   rabi
