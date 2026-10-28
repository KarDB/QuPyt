.. _variable_blocks_ps:

AWG Blocks with Different Durations
============================================================

The :download:`variable_blocks.py <variable_blocks.py>` generator writes an
expanded logical pulse sequence that can be played through either the flat
or repeated-child Tektronix AWG paths. The accompanying
:ref:`mock measurement configurations <variable_blocks_configs>` use simulated
devices and save real measurement data and metadata without connecting to an AWG.

Block durations and timing
------------------------------------------------------------

``total_duration`` remains the default duration **per block**, in microseconds.
Existing generators using ``YamlSequence(duration=...)`` retain their meaning.
``block_durations`` overrides that default for named blocks; it does not change
pulse start times or merge blocks into a continuous waveform.
Legacy numeric strings in the scalar ``total_duration`` remain readable. New
named overrides require numeric values, rather than quoted strings or booleans.

.. code-block:: python

   sequence = YamlSequence(
       duration=1.2,
       block_durations={"prep": 1.2, "drive": 2.4, "readout": 0.96},
   )
   sequence.add_pulse("MW", 0.2, 0.8, amplitude=0.2,
                      frequency=50e6, sequence_blocks=["drive"])
   sequence.set_block_duration("done", 1.2)  # Explicit, entirely idle block.

Every pulse start is relative to its own block. Time after the final pulse
remains idle until the declared block duration is reached. Duration changes
do not stretch pulses or change modulation phase conventions. The independent
sample count is ``int(round(sampling_rate_hz * duration_us * 1e-6))``; pulses
must fit the resulting sample allocation.

At 5 GHz the example produces:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Block
     - Duration (microseconds)
     - Samples
     - Content
   * - prep
     - 1.2
     - 6,000
     - Source 2 analog and marker pulses, followed by idle time.
   * - drive
     - 2.4
     - 12,000
     - Source 1 microwave pulse; optional A1 step flag.
   * - readout
     - 0.96
     - 4,800
     - Source 1 marker, source 2 analog; optional B2 step flag.
   * - done
     - 1.2
     - 6,000
     - No pulses; the full block is idle.

The body is ``prep, drive x2, readout, drive x2, readout, done``.
It occupies 69,600 samples, or 13.92 microseconds at 5 GHz. The generator
returns this rounded-sample duration as ``body_duration_us`` in the saved
measurement metadata. It excludes the AWG trigger-gate waveform, trigger
waiting, acquisition overhead, and hardware latency. There are two readout
pulses, so the generator also returns ``sensor.config.number_measurements: 2``.
This demonstrates returning dependent parameters through the existing generator
interface.

Flat and repeated-child playback
------------------------------------------------------------

Both modes write the same expanded order and per-step repeat counts:

.. code-block:: yaml

   sequencing_order: [prep, drive, readout, drive, readout, done]
   sequencing_repeats: [1, 2, 1, 2, 1, 1]

The child configuration additionally writes:

.. code-block:: yaml

   awg_sequence_mode: repeated_child
   awg_repeated_child:
     prefix_steps: 1
     child_steps: 2
     repetitions: 2

Here ``prep`` is the prefix, ``[drive x2, readout]`` is the two-step child,
and ``done`` is the suffix. These counts describe logical sequence entries,
rather than samples. The child retains its internal waveform repeat count of
two for ``drive``. Reusing a child does not pad its waveforms to a common length
or duplicate the block's samples in the archive. The repeated region must match
in block names, repeat counts, and flag state. Existing model checks for child
sequencing still apply when using a real AWG.

NPZ compilation and cache migration
------------------------------------------------------------

The compiler still writes ``~/.qupyt/sequences/sequence.npz``. It stores each
unique block as a separate rectangular numeric array with shape
``(2 * number_of_sources, samples_in_this_block)``. Consecutive rows contain
one source's analog samples and packed marker values. The new versioned archive
uses named members and JSON metadata; it contains no pickled objects. Equal
and unequal block durations use the same schema.

Finite analog samples retain their existing values, including values outside
the normalized ``[-1, 1]`` range. The archive and uploader do not clip or
normalize them. This preserves legacy pulse generators using amplitudes such
as ``1.3``. Non-finite samples and invalid packed marker bytes are rejected.

The archive's block names identify unique waveform assets; its expanded
``step_block_names`` and ``repeat_counts`` identify playback. A repeated name
always refers to the same samples and duration. Give blocks different names
when their samples or durations differ. Changing between flat and child mode
changes the instrument's sequence tables, while preserving this logical program.

The archive is a local compiler cache. The AWG receives waveform samples,
marker bytes, and SCPI sequencing commands through VISA, rather than the NPZ
file itself. The compiler automatically regenerates older caches from the
original numbered YAML files. Directly loading a legacy archive requires
regenerating it; it does not restore pickled metadata. Keep the generator and
its YAML files when moving a measurement between versions.

Channel mappings and hardware checks
------------------------------------------------------------

This example deliberately uses the established integer channel mapping:
``MW: 0``, ``READ: 1``, ``LASER: 5``, and ``SYNC: 6``. The source ordering is
``[1, 2]``. Configure those numbers for the outputs established on your AWG.
The existing marker packing, marker configuration, DAC-resolution selection,
and integer mapping behavior are preserved.

``A1`` and ``B2`` select flags on physical outputs 1 and 2 respectively.
They represent sequencer-step state: the nominal pulse start/duration under a
flag channel does not define timed flag edges. Set ``include_flags: false`` to
omit flag channels from the generated blocks. For a single-output instrument,
set ``second_source: false`` and ``synchroniser.config.channels: [1]``; remove
the ``LASER``, ``SYNC``, and ``READOUT_ACTIVE`` mappings. Keep the remaining
integer mappings consistent with your existing sequences.

The supplied lengths meet the AWG70000 minimums and are multiples of the
240-sample flag-stability interval. When ``block_durations`` is nonempty, real
loading queries waveform-length constraints on the connected instrument and
rejects unsupported lengths rather than silently padding or shortening the
sequence. Existing scalar-duration sequences retain their previous hardware
query behavior. When changing durations or sampling rate, check the resulting
sample counts against your instrument.

The uploader currently uses one binary transfer per analog or marker vector.
New variable-block inputs are therefore limited to 249,999,999 samples per block
by the analog message size, even when the instrument reports a larger waveform
memory limit. Larger requests fail before allocation with an explicit error;
chunked transfers remain a separate future extension. No durations are padded.

Generator source
------------------------------------------------------------

.. literalinclude:: variable_blocks.py
   :language: python
