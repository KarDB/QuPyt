# AWG transmission reference

`awg_legacy_transmissions.json` was captured from the production compiler,
archive loader, waveform uploader, and sequencer before the variable block
length migration. It contains six cases: active physical sources `[1]`,
`[1, 2]`, and `[2, 1]`, each in flat and repeated-child mode.

Each case embeds its original YAML instructions and channel mapping. Text
commands retain their original order and spelling. Binary entries record the
length and SHA-256 digest of the complete IEEE block produced by PyVISA's real
`to_ieee_block` serializer, including the binary block header. No instrument
was connected during capture. The sampling rate was 1 MHz so the reference
arrays remain small. Tests replay production functions against this reference;
they do not reconstruct the SCPI command generator.
