import logging
from typing import Dict, Any, List, Tuple
import hashlib
from pathlib import Path
import numpy as np
from termcolor import colored
import yaml
from qupyt.set_up import get_seq_dir


class PulseSequenceYaml:
    def __init__(self,
                 channel_mapping: Dict[str, Any],
                 samprate: float = 5e9,
                 yaml_file: Path = get_seq_dir() / 'sequence.yaml') -> None:
        self.yaml_file = yaml_file
        self.channel_mapping = channel_mapping
        self.samp_rate = float(samprate)  # samples per second

    def _sequence_didnt_change(self) -> bool:
        with open(self.yaml_file, 'r', encoding='utf-8') as file:
            sequence_instructions = yaml.safe_load(file)
        try:
            with open(self.yaml_file.with_suffix('.aux'), 'r', encoding='utf-8') as file:
                previous_sequence_instructions = yaml.safe_load(file)
            with open(self.yaml_file.with_suffix('.aux'), 'w', encoding='utf-8') as file:
                yaml.dump(sequence_instructions, file)
            return previous_sequence_instructions == sequence_instructions
        except FileNotFoundError:
            with open(self.yaml_file.with_suffix('.aux'), 'w', encoding='utf-8') as file:
                yaml.dump(sequence_instructions, file)
            return False

    def translate_yaml_to_numeric_instructions(self) -> None:
        if self._sequence_didnt_change():
            return
        with open(self.yaml_file, 'r', encoding='utf-8') as file:
            sequence_instructions = yaml.safe_load(file)
        sequence_order = sequence_instructions['sequencing_order']
        sequencing_repeats = sequence_instructions['sequencing_repeats']
        duration = float(sequence_instructions['total_duration'])
        sorted_pulse_blocks = sorted(set(sequence_order))
        seq = PulseSequence(
            len(sorted_pulse_blocks),
            duration,
            samprate=self.samp_rate
        )
        for i, block in enumerate(sorted_pulse_blocks):
            for channel, pulses in sequence_instructions[block].items():
                for pulse in pulses.values():
                    seq.add_pulse(
                        i,
                        float(pulse['start']),
                        float(pulse['duration']),
                        channel=self.channel_mapping[channel],
                        inputtype="time",
                        amplitude=float(pulse['amplitude']),
                        freq=(
                            float(pulse['frequency']),
                            float(pulse['phase'])
                        )
                    )
        seq.sequencer = sequencing_repeats
        seq.sequencernames = sequence_order
        seq.make('sequence.npz')


class PulseSequence:
    def __init__(self,
                 numseqs,
                 duration: float = 1.4,
                 samprate: float = 2.5e9) -> None:
        self.samp_rate = samprate  # samples per second
        self.min_time = 1 / samprate
        self.num_points = int(samprate * duration * 1e-6)
        self.time = np.linspace(
            0, duration, self.num_points)  # in microseconds
        self.numseqs = numseqs

        if self.num_points != samprate * duration * 10**-6:
            print(
                colored(
                    "WARNING! the sequence duration is not an integer multiple of samples"
                    .ljust(65, ".") + "! [WARNING]", "red",
                )
            )
            logging.warning(
                "The sequence duration is not an integer multiple of samples"
                .ljust(65, ".") + "[WARNING]"
            )
        self.pulses = np.zeros((numseqs, 10, self.num_points))
        self.sequencer = None
        self.sequencernames = None
        self.warning_counter = 0
        self.properties = {"Values": "None"}

    def time_to_index(self, time: float) -> int:
        points = self.samp_rate * time * 1e-6
        if not np.isclose(points, int(round(points))):
            print(
                colored(
                    "WARNING! the start or duration time is not an integer multiple of samples".ljust(
                        65, "."
                    )
                    + "! "
                    "[WARNING]",
                    "red",
                )
            )
            print(
                colored(
                    "WARNING! {} comp. to {}".format(points, int(round(points))).ljust(
                        65, "."
                    )
                    + "! [WARNING]",
                    "red",
                )
            )
            logging.warning('The start of duration is not an integer mulitple of samples'
                            .ljust(65, '.') + '[WARNING]')
            self.warning_counter += 1
        return int(round(points))

    def add_pulse(
        self,
        numseq,
        start: float,
        duration: float,
        channel: int = 0,
        inputtype: str = "time",
        amplitude: float = 1.0,
        freq=None,
    ) -> None:
        """return       analog channel pulse.
        numseq:          which sequence to add the pulse to.
        start & duration:time in microseconds or points.
        inputtype:       'time'   => input in microseconds
                         'points' => points
        channel can be 0 or 5 and will write to AWG source 1 or 2.
        """
        if inputtype == "time":
            start = self.time_to_index(start)
            duration = self.time_to_index(duration)
        else:
            print("Interpreting input as number of sampling points!")

        if freq is not None:
            frequency, phase = freq
            frequency *= 10**-6
            fr = np.cos(
                2 * np.pi * frequency *
                self.time[int(start): int(start + duration)] + phase
            )
        else:
            fr = 1
        try:
            self.pulses[numseq[0]: numseq[1], channel, int(start): int(start + duration)] = (
                amplitude * fr
            )
        except Exception:
            self.pulses[numseq, channel, int(start): int(start +
                        duration)] = amplitude * fr

    def make(self, name: str) -> None:
        print(f'there where {self.warning_counter} warnings in PS generation')
        logging.info(f'there where {self.warning_counter} warnings in PS generation'
                     .ljust(65, '.') + '[success]')
        final = np.zeros((self.numseqs, 4, self.num_points))
        final[:, 0, :] = self.pulses[:, 0, :]
        final[:, 1, :] = (
            self.pulses[:, 1, :] * 2**7
            + self.pulses[:, 2, :] * 2**6
            + self.pulses[:, 3, :] * 2**5
            + self.pulses[:, 4, :] * 2**4
        )
        final[:, 2, :] = self.pulses[:, 5, :]
        final[:, 3, :] = (
            self.pulses[:, 6, :] * 2**7
            + self.pulses[:, 7, :] * 2**6
            + self.pulses[:, 8, :] * 2**5
            + self.pulses[:, 9, :] * 2**4
        )

        hash1 = hashlib.sha1(final.tobytes()).hexdigest()
        hash2 = hashlib.sha1(str(self.sequencer).encode("utf-8")).hexdigest()
        hash3 = hashlib.sha1(
            str(self.sequencernames).encode("utf-8")).hexdigest()
        hash4 = hash1 + hash2 + hash3
        finalhash = hashlib.sha1(hash4.encode("utf-8")).hexdigest()

        file_dir = get_seq_dir()
        np.savez(
            file_dir / name,
            final,
            self.sequencer,
            self.sequencernames,
            finalhash,
            self.properties,
        )
        print(colored(f"Pulse sequence written to {name}", "green"))
        print(f"There where {self.warning_counter} warnings")


class PulseBlasterSequence:
    def __init__(self,
                 channel_mapping: Dict[str, Any],
                 yaml_file: Path = get_seq_dir() / 'sequence.yaml') -> None:
        self.event_times: List[float] = []
        self.event_durations: List[float] = []
        self.event_channel: List[str] = []
        self.events: List[str] = []
        self.channel_bits: List[int] = []
        self.segment_durations: List[float] = []
        self.channel_mapping = channel_mapping
        self.ps: Dict[str, Any] = {}
        self.yaml_sequence: Dict[str,
                                 Any] = self._load_yaml_sequence(yaml_file)
        self.total_duration = self.yaml_sequence['total_duration']

    def _load_yaml_sequence(self, path: Path) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as file:
            yaml_sequence = yaml.safe_load(file)
        return yaml_sequence

    def parse_pulse_sequence_file(self) -> None:
        for block in self.yaml_sequence['sequencing_order']:
            self._parse_block(self.yaml_sequence[block])
            self._sort_pulses()
            self._get_event_durations()
            self._compute_channel_bits()
            self.ps[block] = {'channel_bits': self.channel_bits,
                              'durations': self.event_durations}
            self._reset_attributes()

    def compile(self) -> Tuple[List[int], List[float]]:
        """
        combines the individual sub seqeunces into one long sequence that
        can be uploaded to the pulse blaster card in one go.
        """
        channel_bits = []
        bits_duration = []
        sequencing_info = zip(self.yaml_sequence['sequencing_order'],
                              self.yaml_sequence['sequencing_repeats'])
        for sequence_block, block_repeats in sequencing_info:
            channel_bits += self.ps[sequence_block]['channel_bits'] * \
                block_repeats
            bits_duration += self.ps[sequence_block]['durations'] * \
                block_repeats

        return channel_bits, bits_duration

    def _reset_attributes(self) -> None:
        self.event_times = []
        self.event_durations = []
        self.event_channel = []
        self.events = []
        self.channel_bits = []
        self.segment_durations = []

    def _append_event(self, channel: str, pulse: Dict[str, float]) -> None:
        self.event_times.append(pulse['start'])
        self.event_times.append(pulse['start'] + pulse['duration'])
        self.event_channel.append(channel)
        self.event_channel.append(channel)
        self.events.append('up')
        self.events.append('down')

    def _parse_channel(self,
                       channel: str,
                       channel_pulses: Dict[str, Any]) -> None:
        for pulse in channel_pulses.values():
            self._append_event(channel, pulse)

    def _parse_block(self, ps_block: Dict[str, Any]) -> None:
        for channel, channel_pulses in ps_block.items():
            self._parse_channel(channel, channel_pulses)

    def _sort_pulses(self) -> None:
        self.event_times, self.event_channel, self.events = zip(*sorted(zip(
            self.event_times,
            self.event_channel,
            self.events
        )))

    def _get_event_durations(self) -> None:
        self.event_durations = [
            i-j for i, j in zip(self.event_times[1:], self.event_times[:-1])]
        if self.event_times[-1] != self.total_duration:
            self.event_durations.append(
                self.total_duration - self.event_times[-1])

    def _event_to_sign(self, event: str) -> int:
        if event == 'up':
            return 1
        if event == 'down':
            return -1
        raise ValueError(f'No event {event}, options are "up" or "down"')

    def _compute_channel_bits(self) -> None:
        for i, _ in enumerate(self.event_durations):
            prev_val = self.channel_bits[-1] if self.channel_bits else 0
            self.channel_bits.append(
                prev_val + 2 ** (self.channel_mapping[self.event_channel[i]])
                * self._event_to_sign(self.events[i]))
        if self.event_times[0] != 0:
            self.channel_bits.insert(0, 0)
            self.event_durations.insert(0, self.event_times[0])
        self._pop_uncessary_entries()

    def _pop_uncessary_entries(self) -> None:
        indices_to_pop = [index for index, duration in enumerate(
            self.event_durations) if duration == 0]
        for pop_index in indices_to_pop:
            self.event_durations.pop(pop_index)
            self.channel_bits.pop(pop_index)
