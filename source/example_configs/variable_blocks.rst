.. _variable_blocks_configs:

Variable-Duration AWG Mock Measurements
============================================================

These configurations use the same :ref:`variable-duration pulse generator
<variable_blocks_ps>` with two AWG sequence modes. All devices are mocked. Each
measurement acquires two small simulated images and saves data plus metadata.
The generator writes numbered sequence YAML files under
``~/.qupyt/sequences/``.

Run a measurement
------------------------------------------------------------

Start ``qupyt --verbose`` from the repository root. Copy either configuration
into ``~/.qupyt/waiting_room/`` using a new filename. QuPyt claims the
instruction, generates the pulse YAML, prepares the mock hardware, acquires
data, and saves results in the process's working directory. The instruction
is removed after success. Use an absolute ``ps_path`` if launching QuPyt from
another directory.

The mock can validate the logical child layout and compile the numeric AWG
archive for variable-duration blocks. It does not send VISA commands, test
physical wiring, validate an instrument's reported constraints, or prove
hardware timing. Automated tests also exercise the production AWG compiler and
read the resulting NPZ without pickle. No connected AWG has been tested for
these examples.

For an instrument measurement, retain your established AWG connection,
channel mapping, source order, sensor configuration, and static/dynamic
devices, and use the generator's timing options as an example. Match the
generator's ``sampling_rate`` to ``synchroniser.config.sampling_rate``. The
default 13.92-microsecond body is a compact waveform demonstration; adapt the
readout timing to your actual sensor before using it for acquisition.

Flat mode
------------------------------------------------------------

.. literalinclude:: Variable_blocks_flat.yaml
   :language: yaml

Repeated-child mode
------------------------------------------------------------

.. literalinclude:: Variable_blocks_child.yaml
   :language: yaml
