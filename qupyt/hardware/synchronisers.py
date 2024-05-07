# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
"""
Handle AWG input and output.
"""

from __future__ import annotations
import logging
from time import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List
from pathlib import Path
import sys
import ctypes as ct

import yaml
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from termcolor import colored

from qupyt.set_up import get_seq_dir
from qupyt.pulse_sequences.SequenceDesigner import PulseSequenceYaml, PulseBlasterSequence
from pulsestreamer import PulseStreamer
from pulsestreamer import findPulseStreamers
from pulsestreamer import TriggerStart, TriggerRearm
from pulsestreamer import Sequence, OutputState
from qupyt.hardware.visa_handler import VisaObject
from qupyt import set_up
from qupyt.mixins import ConfigurationMixin, UpdateConfigurationType, PulseSequenceError
try:
    import qupyt.hardware.wrappers.spinapi_adapted as spapi
except (ImportError, NameError):
    spapi = None
    logging.warning(
        "Could not load spinapi library".ljust(65, '.')+'[failed]\nIf you are not using a Pulse Streamer you do not need this!')


class SynchroniserFactory:
    """
    Synchroniser Factory responsible for creating and returning an instance of the
    requested synchroniser. All synchronisers created from this factory will adhere to
    an interface specified in :class:`Synchroniser`. This ensures, that all synchronisers
    can be used interchangeably in the measurement code.

    Methods:
        - :meth:`create_synchroniser`: This method creates the appropriate synchroniser
          instance and configures it. **Note for non programmers**: create_synchroniser
          is a static method. This means you don't have to create a class
          instance to call it.

    Example:
        ::

            cam = SynchroniserFactory.create_synchroniser('TekAWG',
                                                          {'address': 'TCPIP::ipaddress::INSTR'},
                                                          {"LASER": 1,
                                                           "READ": 2})
    """
    @staticmethod
    def create_synchroniser(sync_type: str,
                            configuration: Dict[str, Any],
                            channel_mapping: Dict[str, Any]) -> Synchroniser:
        """
        :param sync_type: Synchroniser model identifier e.g. 'TekAWG'.
         Current models include: SwabInstPS, TekAWG, MockSynchroniser, PulseBlaster.
        :type sync_type: string
        :param configuration: configuration parameters for the synchroniser.
         Provided parameters need to match those available for the synchroniser.
         Please check the specific synchronisers for more information.
        :type configuration: dict
        :param channel_mapping: Matches channel descriptors to ports on the device.
        :type channel_mapping: dict
        :return: Instance of the requested synchroniser.
        :rtype: Synchroniser
        :raises ValueError:
        """
        if sync_type == 'SwabInstPS':
            return PStreamer(
                configuration, channel_mapping
            )
        if sync_type == 'TekAWG':
            return AWGenerator(
                configuration, channel_mapping
            )
        if sync_type == 'MockSynchroniser':
            return MockGenerator(
                configuration, channel_mapping
            )
        if sync_type == 'PulseBlaster':
            return PulseBlaster(
                configuration, channel_mapping
            )
        raise ValueError(f"Unknown synchroniser type {sync_type}")


class Synchroniser(ABC, ConfigurationMixin):
    """
    Abstract Base Class for all synchronisers. All synchronisers implemented in QuPyt
    should inherit from this class. This helps ensure compliance with the
    synchroniser API.

    Synchronisers handle the interplay of all other devices during
    one measurement step. To do this they need a pulse sequence they are
    supposed to play during the measurement. This pulse sequence is read
    from a yaml file from its default location
    (~/.qupyt/sequences/sequence.yaml). The specifics of how a pulse sequence
    is played depends on the synchroniser (Tektronix AWG, PulseStreamer, ...)
    and have to be dealt with in the implementation of this class for the
    synchroniser in question.

    **Note**: The attributes listed below are never explicitly set by the user.
    Please use the ``configuration`` constructor argument to configure the sensor.

    Arguments:
        - **configuration** (dict): Configuration dictionary. Keys will be used
          to select setter methods from an attribute map dicionary to set
          associated values.

          Possible configuration values:
            - **address** (str): Address used to open a connection to the device.
              For VISA devices this could for example be: "TCPIP::<idaddress>::INSTR".

          Concrete sensor classes may have additional configuration values.
    """
    attribute_map: UpdateConfigurationType

    def __init__(self) -> None:
        self.address: str
        self.attribute_map = {
            'address': self._set_address,
        }

    def _set_address(self, address: str) -> None:
        self.address = address

    @abstractmethod
    def load_sequence(self) -> None:
        """
        Load the YAML formatted pulse sequence from its default location
        (see above), and parse it to the specifics of the synchroniser. Finally
        it uploads the instructions to the hardware.
        """

    @abstractmethod
    def run(self) -> None:
        """
        Activates all output channels on the devices and starts playing.
        This command does not yet start the measurement sequence. Sometimes
        this step is needed to avoid slow ramp up times and undesired pulses
        some synchronisers emit when they first start.
        For some sychronisers this step is unecessary. However, it still needs
        to be implemented for compatibility reasons.
        """

    @abstractmethod
    def open(self) -> None:
        """
        Establishes the connection to the :class:`Synchroniser` and
        runs configuration.
        """

    @abstractmethod
    def trigger(self) -> None:
        """
        Sends a command to the synchroniser to play the measurement sequence
        once. Note that this may include however many readouts or repetitions
        as desired per measurement burst.
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Deactivates all outputs of the synchroniser. An stops the
        execution of the pulse sequence.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Closes the connection to the synchroniser.
        """


class AWGenerator(VisaObject, Synchroniser):
    """
    Synchroniser implementation for the Tektronix AWG 5000 series.
    """

    def __init__(self,
                 configuration: Dict[str, Any],
                 channel_mapping: Dict[str, Any]) -> None:
        self.device_type: str = "TekAWG"
        self.samprate: float = 5e9
        self.channel_mapping = channel_mapping

        self.wavenames: list[str]
        self.seqrepeats: list[int]
        self.waveform_block: np.ndarray
        self.analog_amplitude: float = 1.0
        self.marker_amplitude: float = 1.75
        self.dac_resolution: int = 12
        self.channels: list[int] = [1, 2]
        self.marker_channels: list[int] = [1, 2, 3, 4]

        Synchroniser.__init__(self)
        self.attribute_map['device_type'] = self._set_device_type
        self.attribute_map['sampling_rate'] = self._set_sampling_rate_attribute
        self.attribute_map['channels'] = self._set_channels_attribute
        if configuration is not None:
            self._update_from_configuration(configuration)
        VisaObject.__init__(self, self.address, self.device_type)
        self.instance.timeout = 20000

    def open(self) -> None:
        self._configure()

    def close(self) -> None:
        """Passing"""

    def run(self) -> None:
        self.instance.write('awgcontrol:run:immediate')
        logging.info('Turned on AWG output to RUN immediate'.ljust(
            65, '.') + '[done]')
        self.opc_wait()

    def stop(self) -> None:
        self.instance.write('awgcontrol:stop:immediate')
        logging.info('Turned on AWG output to STOP immediate'.ljust(
            65, '.') + '[done]')
        self.opc_wait()

    def trigger(self) -> None:
        self.instance.write('SOUR1:JUMP:FORC 2')
        logging.info('Sent trigger to AWG'.ljust(65, '.') + '[done]')

    def load_sequence(self) -> None:
        self.stop()
        self._clear_awg()
        sequence_translator = PulseSequenceYaml(
            self.channel_mapping, self.channels, samprate=self.samprate)
        sequence_translator.translate_yaml_to_numeric_instructions()
        self._load_sequence_block(get_seq_dir() / 'sequence.npz')
        self._upload_waveforms()
        self._sequence('autoseq', nongatereps=1)
        logging.info('Loaded and sequenced current pulse sequence'.ljust(
            65, '.') + '[done]')
        self.opc_wait()

    def _configure(self) -> None:
        self._set_sampling_rate()
        for channel in self.channels:
            self._set_daq_resolution(channel, self.dac_resolution)
            self._set_analog_amplitude(channel, self.analog_amplitude)
            self._set_output_on(channel)
            for marker in self.marker_channels:
                self._set_marker_amplitude(
                    channel, marker, self.marker_amplitude)
        print('Configuring AWG'.ljust(65, '.') + colored(' [done]', 'green'))

    def _upload_waveform(self, wavename: str, waveform: np.ndarray) -> None:
        self.instance.write('wlist:waveform:delete "'+wavename+'"')
        self.instance.write('wlist:waveform:new "'+wavename +
                            '",'+str(waveform.shape[1])+',real')
        self.instance.write_binary_values(
            'wlist:waveform:data "'+wavename+'",', waveform[0, :])
        self.instance.write_binary_values(
            'wlist:waveform:marker:data "'+wavename+'",',
            waveform.astype(np.uint8)[1, :], datatype='B')
        logging.info(f'Uploaded waveform {wavename}'.ljust(65, '.') + '[done]')
        self.opc_wait()

    def _sequence(self,
                  seqname: str,
                  nongatereps: int = 1) -> None:
        print('Setting up sequencer'.ljust(65, '.'), end='')
        for channel in self.channels:
            self.instance.write(f'slist:sequence:delete "sub_{channel}"')
            self.instance.write(
                f'slist:sequence:new "sub_{channel}",{len(self.wavenames)},1')
            self.instance.write(
                f'slist:sequence:event:jtiming "sub_{channel}" immediate')
            for i, wavename in enumerate(self.wavenames):
                self.instance.write(
                    f'slist:sequence:step{i+1}:rcount "sub_{channel}",{self.seqrepeats[i]}')
                self.instance.write(
                    f'slist:sequence:step{i+1}:tasset1:waveform "sub_{channel}","{wavename}_{channel}"')

            self.instance.write(f'slist:sequence:delete "{seqname}_{channel}"')
            self.instance.write(
                f'slist:sequence:new "{seqname}_{channel}",2,1')
            self.instance.write(
                f'slist:sequence:step2:goto "{seqname}_{channel}",first')
            self.instance.write(
                f'slist:sequence:event:jtiming "{seqname}_{channel}" immediate')

            # for gating pulse
            self.instance.write(
                f'slist:sequence:step1:rcount "{seqname}_{channel}", INF')
            self.instance.write(
                f'slist:sequence:step1:tasset1:waveform "{seqname}_{channel}","{self.wavenames[0]}_{channel}"')

            # for actual seq
            self.instance.write(
                f'slist:sequence:step2:rcount "{seqname}_{channel}", {nongatereps}')
            self.instance.write(
                f'slist:sequence:step2:tasset1:sequence "{seqname}_{channel}","sub_{channel}"')

            self.instance.write(
                f'source{channel}:casset:sequence "{seqname}_{channel}",1')
        self.opc_wait()
        print(colored(' [done]', 'green'))

    def _load_sequence_block(self, seqname: Path) -> None:
        block = np.load(seqname)
        self.waveform_block = block['arr_0']
        self.seqrepeats = list(block['arr_1'])
        self.wavenames = list(block['arr_2'])
        logging.info('loaded wave sequence from file'.ljust(
            65, '.')+f'{seqname}')

    def _clear_awg(self) -> None:
        self.instance.write('slist:sequence:delete all')
        self.instance.write('wlist:waveform:delete all')
        logging.info('Clear all AWG slist and wlist'.ljust(65, '.') + '[done]')
        self.opc_wait()

    def _upload_waveforms(self) -> None:
        time_1 = time()
        sorted_wavenames = sorted(set(self.wavenames))
        for i, wavename in tqdm(enumerate(sorted_wavenames),
                                ascii=True,
                                desc="uploading waveforms",
                                ):
            for channel_index, channel in enumerate(self.channels):
                self._upload_waveform(
                    f'{wavename}_{channel}', self.waveform_block[i, channel_index*2:(channel_index*2)+2])
                self.opc_wait()
        logging.info(
            f"Uploaded Tektronix AWG waveforms in {time() - time_1} seconds".ljust(65, ".") + '[done]')

    def _set_output_on(self, channel: int) -> None:
        self.instance.write(f'outp{channel} on')
        logging.info(f'Set channel{channel} output to on'
                     .ljust(65, '.') + '[done]')

    def _set_daq_resolution(self, channel: int, dac_resolution: int) -> None:
        self.instance.write(
            f'source{channel}:dac:resolution {dac_resolution}')
        logging.info(f'Set AWG channel{channel} resolution to bit'
                     .ljust(65, '.')+'{dac_resolution}')

    def _set_sampling_rate(self) -> None:
        self.instance.write(f'source:frequency {self.samprate}')
        logging.info('AWG sampling rate'.ljust(
            65, '.')+f'{self.samprate}')
        self.opc_wait()

    def _set_marker_amplitude(self, channel: int,
                              marker: int, voltage: float = 1.75) -> None:
        self.instance.write(
            f'SOURCE{channel}:MARKER{marker}:VOLTAGE:LEVEL:IMMEDIATE:HIGH {voltage}')
        self.opc_wait()
        logging.info(f'Set AWG channel{channel} marker{marker} to / V'
                     .ljust(65, '.') + f'{voltage}')

    def _set_analog_amplitude(self,
                              channel: int,
                              amplitude: float = 1.0) -> None:
        # amplitude is given in fractions of the max amplitude.
        self.instance.write(
            f'source{channel}:voltage:level:immediate:amplitude {amplitude}')
        self.opc_wait()
        logging.info(f'Set AWG channel{channel} analog amplitude to'
                     .ljust(65, '.') + f'{amplitude}')

    def _set_device_type(self, device_type: str) -> None:
        self.device_type = device_type

    def _set_sampling_rate_attribute(self, sampling_rate: float) -> None:
        self.samprate = sampling_rate

    def _set_channels_attribute(self, channels: list[int]) -> None:
        self.channels = channels

    def plot_waveform(self, wavename: str) -> None:
        """
        Convenience function to query and plot the values that were previously
        uploaded to the AWG. This can be very usefull to verify, that
        the upload worked as expected.

        :param wavename: Identifier of the
         uploaded sequence block to query and plot.
        :type wavename: str
        """
        marker = self.instance.query_binary_values(
            'wlist:waveform:marker:data? "' + wavename + '"', datatype='B')
        analog = self.instance.query_binary_values(
            'wlist:waveform:data? "' + wavename+'"')
        markers = np.zeros((4, len(marker)))
        print(np.shape(markers), np.shape(analog))
        for i, mark in enumerate(marker):
            num = bin(mark)[2:]
            if not len(num) < 5:
                for j in range(1, 5):
                    try:
                        markers[-j, i] = int(num[-(j+4)])
                    except Exception:
                        break

        plt.plot(np.asarray((analog))*0.5+4)
        plt.plot(markers[0, :]*0.5+3)
        plt.plot(markers[1, :]*0.5+2)
        plt.plot(markers[2, :]*0.5+1)
        plt.plot(markers[3, :]*0.5+0)
        plt.show()


class PStreamer(Synchroniser):
    def __init__(self,
                 configuration: Dict[str, Any],
                 channel_mapping: Dict[str, Any]) -> None:
        super().__init__()
        self.channel_mapping = channel_mapping
        self._update_from_configuration(configuration)
        if configuration['address'] == 'None':
            self._find_pulse_streamers()

    def _find_pulse_streamers(self) -> None:
        devices = findPulseStreamers()
        if devices:
            print("Detected PulseStreamer: ")
            print(devices)
            print("----------------------------------------------------\n")
            self.address = devices[0][0]
        else:
            print(
                "No PulseStreamers found by autodetect.\n-------------------\n"
            )
            logging.info('Pulse Streamer not found'.ljust(
                65, '.') + '[failed]')

    def open(self) -> None:
        """
        Opens the connection with the given device of IP = ip_address.
        If an IP is not introduced -- search a device in the network.
        If an IP is introduced -- checks if correct and tries to connect.
        """
        # if no IP adress is provided, try to detect one:
        try:
            self.pulser = PulseStreamer(self.address)
            logging.info(f'Pulse Streamer connected at {self.address}'
                         .ljust(65, '.') + '[done]')
            print("Connected.\n-------------------------------------------\n")
        except AssertionError:
            logging.exception(f"No pulse streamer with the IP address {self.address}"
                              .ljust(65, '.') + '[failed]')

    def close(self) -> None:
        """
        There is no function for closing the connection of the device. We have
        tested that running thousands of connections does not slow down the
        execution, we assume we can call pass on the close function.
        """

    def stop(self) -> None:
        """
        Function that stofull_pulse_list the playing sequence and sets all
        the channels to a ZERO state (0V).
        """
        try:
            # define the final state of the Pulsestreamer
            _ = OutputState.ZERO()
            # force the final state.
            self.pulser.forceFinal()
            # print a text if the program has successfully ended
            if not self.pulser.isStreaming():
                print("Stop PulseStreamer: The sequence has been stoped.")
                logging.info('Pulse Streamer: stopped pulse sequence execution'
                             .ljust(65, '.') + '[done]')
            else:
                logging.warning(
                    'Warning full_pulse_listtreamer.stop(): The sequence could\
                            not be stoped'.ljust(65, '.') + '[done]')

        except AttributeError:
            logging.exception('Pulse Streamer: error stopping pulse sequence'
                              .ljust(65, '.') + '[done]')

    def writeDigSeq(self, channel_key: str) -> List[Tuple[int, int]]:
        """
        Parameters
        ----------
        channel_key : str
            channel_key of the pulse to be written (LASER, MW or READ)

        Recieves a dictionary of pulses and returns sequence for the
        PulseStreamer of the given channel: LASER, MW or READ.
        """
        self.channel_key = channel_key
        self.seq = []  # sequence of the channel
        pointer_i = 0  # pointer in time
        # Check if the asked channel_key exists in the file
        if self.channel_key not in list(self.pulse_list.keys()):
            logging.error("KeyError: No element named " +
                          str(self.channel_key) + ".")
            raise KeyError

        for i in range(len(self.pulse_list[self.channel_key])):
            # Start and length of the pulse turned into
            # float in case it has str format.
            start_pulse_i = (
                float(
                    self.pulse_list.get(self.channel_key)
                    .get("pulse" + str(i + 1))
                    .get("start")
                )
                * 1e3
            )
            len_pulse_i = (
                float(
                    self.pulse_list.get(self.channel_key)
                    .get("pulse" + str(i + 1))
                    .get("duration")
                )
                * 1e3
            )
            # Check if pulses are well defined
            if pointer_i > start_pulse_i:
                logging.error(
                    "Error: "
                    + self.channel_key
                    + " sequence definition makes no sense, pulses are overlaping!"
                )
                raise PulseSequenceError

            # Check for unsupported analog signals
            freq_pulse_i = (
                self.pulse_list.get(self.channel_key)
                .get("pulse" + str(i + 1))
                .get("frequency")
            )
            if freq_pulse_i != 0:
                logging.warning(
                    "Warning: Frequency different than 0. This programm does not support analog signals. Set frequency to 0."
                )
                raise PulseSequenceError

            ampl_pulse_i = (
                self.pulse_list.get(self.channel_key) .get(
                    "pulse" + str(i + 1))
                .get("amplitude")
            )
            if ampl_pulse_i != 1 or ampl_pulse_i is None:
                logging.warning(
                    "Warning: Amplitude of the pulse different than 1 (can only be 0 or 1). This programm does not support analog signals. Set amplitude to 1."
                )
                raise PulseSequenceError

            phase_pulse_i = (
                self.pulse_list.get(self.channel_key)
                .get("pulse" + str(i + 1))
                .get("phase")
            )
            if phase_pulse_i != 0:
                logging.warning(
                    "Warning: Phase of the pulse different than 0. This programm does not support analog signals. Set amplitude to 1."
                )
                raise PulseSequenceError

            # Check for non-multples of the sampling time.
            if (start_pulse_i - round(start_pulse_i) != 0) or (
                len_pulse_i - round(len_pulse_i) != 0
            ):
                logging.warning(
                    "Warning: Sampling unit is 1ns. Time values are being rounded.")
                # Round the values and turn into integers
                start_pulse_i = int(round(start_pulse_i))
                len_pulse_i = int(round(len_pulse_i))
                # Append a low and a high for each pulse: ___----
                self.seq.append((start_pulse_i - pointer_i, 0))
                self.seq.append((len_pulse_i, 1))
                # Update pointer
                pointer_i = start_pulse_i + len_pulse_i

            else:
                # Turn into integers
                start_pulse_i = int(start_pulse_i)
                len_pulse_i = int(len_pulse_i)
                # Append a low and a high for each pulse: ___----
                self.seq.append((start_pulse_i - pointer_i, 0))
                self.seq.append((len_pulse_i, 1))
                # Update pointer
                pointer_i = start_pulse_i + len_pulse_i

        # Check if the sequence is longer than the defined total time.
        if (pointer_i > self.total_duration and self.total_duration_unparsed != "ignore"):
            logging.error(
                f"Error: {self.channel_key} duration exceeds the defined total time.")
            raise PulseSequenceError
        # Add the final low to make the sequence last its length
        if self.total_duration_unparsed != "ignore":
            self.seq.append((self.total_duration - pointer_i, 0))
        return self.seq

    def plot_sequence(self) -> None:
        """
        From the given file plots the defined sequences
        """
        # check if the sequence has been loaded.
        if self.pulser.isStreaming() is True:
            self.sequence.plot()
        else:
            print(
                "Warning (plot_sequence()): No sequence is streaming in the PulseStreamer."
            )

    def load_sequence(self) -> None:
        """
        Calls the WriteDigSig() function and loads the corresponding sequences
        to the respective channels.
        """
        try:
            # Selected folder:
            self.yaml_file = set_up.get_seq_dir() / "sequence.yaml"
            with open(self.yaml_file, "r", encoding='utf-8') as file:
                full_pulse_list = yaml.load(file, Loader=yaml.FullLoader)
            sequence_order = full_pulse_list['sequencing_order']
            sequencing_repeats = full_pulse_list['sequencing_repeats']

            total_duration_unparsed = full_pulse_list['total_duration']
            self.total_duration_unparsed = total_duration_unparsed
            if total_duration_unparsed == 'ignore':
                self.total_duration = np.inf
            if total_duration_unparsed != 'ignore':
                total_duration = float(
                    total_duration_unparsed) * 1e3  # convert to ns
                if total_duration - round(total_duration) != 0:
                    logging.warning(
                        "Warning: The total duration is not multiple of the\
                                sampling time and is being rounded!".ljust(65, '.')
                        + '[WARNING]'
                    )
                    self.total_duration = int(round(total_duration))
                else:
                    self.total_duration = int(total_duration)

            # Generate Sequence objects for all pulseseqeunce blocks.
            # These will be sequenced together later.
            sequences_to_write = {}
            for block in set(sequence_order):
                sequences_to_write[block] = Sequence()
                self.pulse_list = full_pulse_list[block]
                self.check_types(self.pulse_list)
                for channel in self.pulse_list:
                    sequences_to_write[block].setDigital(
                        self.channel_mapping[channel], self.writeDigSeq(channel))
            self.sequence = Sequence()
            for block, repetitions in zip(sequence_order, sequencing_repeats):
                self.sequence += repetitions * sequences_to_write[block]

        except AttributeError:
            logging.exception("pulseseqeunce upload failed")

    def check_types(self, pulse_list) -> None:
        for channel in pulse_list:
            for pulse, pulse_params in pulse_list[channel].items():
                for par, par_value in pulse_params.items():
                    pulse_list[channel][pulse][par] = float(par_value)

    def run(self) -> None:
        """
        Function that triggers the device:
        send trigger + play sequence one time.
        For now pass, write it later.
        """
        try:
            # never runs the seq
            n_runs = 1

            # reset the device - all outputs 0V
            self.pulser.reset()
            self.pulser.constant(OutputState.ZERO())  # all outputs 0V
            final = OutputState.ZERO()

            # Start the sequence after the upload and disable
            # the retrigger-function
            start = TriggerStart.IMMEDIATE
            rearm = TriggerRearm.MANUAL
            self.pulser.setTrigger(start=start, rearm=rearm)

            # upload the sequence and arm the device
            self.pulser.stream(self.sequence, n_runs, final)
            while self.pulser.isStreaming():
                pass
            # check if the sequence has been started correctly.
            logging.info('Pulse Strearmer: sent run signal'.ljust(
                65, '.') + '[done]')

        except AttributeError:
            logging.exception('Pulse Strearmer: problem setting to run'.ljust(
                65, '.') + '[done]')

    def trigger(self) -> None:
        """
        Function that sets the Pulser  to do one loop of the previously-
        defined sequence. It does not ask for a Trigger. It sets the final
        state to the ZERO state. Resets the channels before re-streaming.
        """
        self.pulser.rearm()
        self.pulser.startNow()
        logging.info('Pulse Strearmer: Sent signal to play sequence'.ljust(
            65, '.') + '[done]')


class MockGenerator(Synchroniser):
    def __init__(self,
                 configuration: Dict[str, Any],
                 channel_mapping: Dict[str, Any]):
        self.channel_mapping = channel_mapping
        self.device_type: str = "MockGenerator"

        Synchroniser.__init__(self)
        if configuration is not None:
            self._update_from_configuration(configuration)
        logging.info('MockSynchroniser instance created'.ljust(
            65, '.') + '[done]')

    def open(self) -> None:
        logging.info('Opened MockSynchroniser'.ljust(
            65, '.') + '[done]')

    def load_sequence(self) -> None:
        try:
            self.yaml_file = set_up.get_seq_dir() / "sequence.yaml"
            with open(self.yaml_file, "r", encoding='utf-8') as file:
                full_pulse_list = yaml.load(file, Loader=yaml.FullLoader)
            total_duration = float(
                full_pulse_list['total_duration']) * 1e3  # convert to ns
            _ = full_pulse_list['sequencing_order']
            _ = full_pulse_list['sequencing_repeats']
            if total_duration - round(total_duration) != 0:
                logging.warning(
                    "Warning: The total duration is not multiple of the sampling time and is being rounded!"
                )
                self.total_duration = int(round(total_duration))
            else:
                self.total_duration = int(total_duration)

            logging.info('Loaded sequence for MockSynchroniser'.ljust(
                65, '.') + '[done]')

        except AttributeError:
            logging.exception(
                "Caught attribute error when writing loading from yaml pulse sequence")

    def run(self) -> None:
        logging.info('Sent run to MockSynchroniser'.ljust(
            65, '.') + '[done]')

    def trigger(self) -> None:
        logging.info('Sent trigger from MockSynchroniser'.ljust(
            65, '.') + '[done]')

    def stop(self) -> None:
        logging.info('Stopped MockSynchroniser'.ljust(
            65, '.') + '[done]')

    def close(self) -> None:
        logging.info('Closed MockSynchroniser'.ljust(
            65, '.') + '[done]')


class PulseBlaster(Synchroniser):
    """
        Class to represent PulseBlaster card.

        Attributes
        ----------
        int PBclk :
            PulseBlaster clock frequency (in MHz)
        int PB_min_instr_clk_cycles:
            minimum instruction time, in clock periods
        int PB_STARTtrig:
            PB channel bit to DAQ start trigger
        int PB_DAQ:
            PB channel bit to DAQ gate/ sample clock
        int PB_AOM:
            PB channel bit to TTL of AOM driver
        int PB_MW:
            PB channel bit to TTL of microwave switch

        Methods
        -------
        error_catcher(int status): Catches error in PB board status
        configure_pb():
            Configures PB board
        int status pb_inst_pbonly(int bit flags, int instruction,
            int instruction_data, int pulse_length):
            Create single instruction for the pulse program
        create_json_sequence(pulseseq_file_name):
            Loads sequence array from npz file and save the decomposed
            data (PB channel bit masks and pulse durations) into json file
        decompose_sequence_array(seq_array, pulseseq_file_name):
            Decomposes seq_array into PB channel bit masks
            and corresponding pulse durations
        decompose_sequence(sequence):
            Decompose single sequence into PB channel
            bit masks and its pulse durations
        get_int_powers_of_two(sequence):
            Finds integer list of powers of 2 for each bit sequence
        load_sequence(pulseseq_file_name):
            Loads Pb channel bit masks and pulse durations from json file
            to pulse program memory
        start_programming():
            Starts the programming of PB pulse program
        stop_programming():
            Stops the programming of PB pulse program
        program_pb(channel_bit_masks, pulse_duration_list):
            Program pulse program memory using
            PB channel bits and pulse durations
        run():
            Triggers/starts the execution of the Pulse Program
        close():
            Closes the communication with the PB board
        stop():
            Stops the execution of the Pulse Program and closes
            the communication. TTL outputs will return to zero.
    """

    def __init__(self,
                 configuration: Dict[str, Any],
                 channel_mapping: Dict[str, Any]) -> None:
        '''
        Configure the PB board core clock frequency,
        minimum instruction clock cycle and PB channel connections.

        '''
        if spapi is None:
            raise RuntimeError(
                "This class requires 'spinapi' by SpinCore to be installed and functional")
        self.device_type: str = "PulseBlaster"
        self.samprate: float = 500  # MHz
        self.pb_min_instr_clk_cycles = 5
        self.channel_mapping = channel_mapping
        Synchroniser.__init__(self)
        self.attribute_map['sampling_rate'] = self._set_sampling_rate_attribute
        self.attribute_map['min_instr_clk_cycles'] = self._set_min_instr_clk_cycles
        if configuration is not None:
            self._update_from_configuration(configuration)

    def open(self) -> None:
        self.configure_pb()

    def load_sequence(self) -> None:
        yaml_sequence_transpiler = PulseBlasterSequence(self.channel_mapping)
        yaml_sequence_transpiler.parse_pulse_sequence_file()
        channel_bit_mask, pulse_duration_list = yaml_sequence_transpiler.compile()
        self.program_pb(channel_bit_mask,
                        pulse_duration_list)

    def close(self) -> None:
        '''
        Releases the PulseBlasterESR-PRO board
        and check the current status of PB board.
        '''
        # Close the communication with the PB board
        status = spapi.pb_close()
        self.error_catcher(status)

    def run(self) -> None:
        '''
        Triggers/starts the execution of the Pulse Program
        and closes communication with the board.
        However, Pulse Program execution will continue.
        '''
        # Returns a 0 on success or a negative number on an error.
        status = spapi.pb_start()
        self.error_catcher(status)
        print(colored('Started execution of Pulseblaster card pulse program!',
                      'green'))

    def trigger(self) -> None:
        status = spapi.pb_start()
        self.error_catcher(status)
        print(colored('Started execution of Pulseblaster card pulse program!',
                      'green'))

    def stop(self) -> None:
        '''
            Stops the execution of the Pulse Program and closes the
            communication. TTL outputs will return to zero.
        '''
        # Returns a 0 on success or a negative number on an error.
        status = spapi.pb_stop()
        self.error_catcher(status)
        print(colored('Stopped execution of Pulseblaster card pulse program!', 'green'))

    def error_catcher(self, status: int) -> None:
        '''
            Checks the status of the PulseBlasterESR.
            PB returns a negative number on an error,
            and 0 or the instruction number on success.
            If error, prints the error message.
            Parameters:
                int status: current status of PB board
        '''
        if status < 0:
            print('Error: ', spapi.pb_get_error())
            sys.exit()

    def configure_pb(self) -> None:
        '''
        Initializes the PulseBlasterESR-PRO board and set
        the core clock frequency of the board.
        '''
        # 1 - Enable the spincore log file, appears as log.txt in current working directory
        spapi.pb_set_debug(0)
        # Initializing communication with the PulseBlasterESR-PRO board
        status = spapi.pb_init()
        # catch error if any and exit the program
        self.error_catcher(status)
        # Configure the core clock frequency (in MHz)
        spapi.pb_core_clock(self.samprate)

    def pb_inst_pbonly(flags, instruction, instruction_data, pulse_length):
        '''
        Create single instruction to send to the pulse program. It returns a negative number on an error, or the instruction number upon success. 
        If the function returns -99, an invalid parameter was passed to the function.
        Instruction format:
            int status pb_inst_pbonly(int bit flags, int instruction, int instruction_data, int pulse_length)
        Parameters:
            int bit flags: state of each TTL output bit.
            int instruction: type of instruction is to be executed.
            int instruction_data: data to be used with the instruction field.
            int pulse_length: duration of this pulse program instruction
        Returns:
            int status: current PB board status
    '''
        return spapi.spinapi.pb_inst_pbonly(ct.c_uint(flags),
                                            ct.c_int(instruction),
                                            ct.c_int(instruction_data),
                                            ct.c_double(pulse_length))

    def check_pulse_length_short(self, pulse_duration: float, current_option: float) -> float:
        if pulse_duration != current_option:
            logging.warning(
                f"pulse duration of {pulse_duration} not possible with PB card. Setting to {current_option}")
            pulse_duration = current_option
            return pulse_duration
        return pulse_duration

    def program_pb(self, channel_bit_masks: List[int],
                   pulse_duration_list: List[float]) -> None:
        '''
        Program the PB pulse program memory by sending instructions
        for each channel bit mask and corresponding pulse duration.
        Channel bit mask can be a decimal, hexadecimal or binary.
        '''
        self.start_programming()

        # Send instructions to the pulse program
        # Instruction format:
        # int status pb_inst_pbonly(int bit flags, int instruction,
        # int instruction_data, int pulse_length)
        for i, pulse_duration in zip(range(len(channel_bit_masks)),
                                     pulse_duration_list):
            # All pulse duration checks in mus.
            if pulse_duration >= 0.01:
                channel_bit_mask = channel_bit_masks[i]
            else:
                # Short Pulse Feature:
                # bits 23-21 controls the number of clock periods
                if 0 < pulse_duration <= 0.003:
                    # 001 for 1 clock period, 2ns for 500 MHz
                    pulse_duration = self.check_pulse_length_short(
                        pulse_duration, 0.002)
                    channel_bit_mask = channel_bit_masks[i] + 2**21
                elif 0.003 < pulse_duration <= 0.005:
                    # 010 for 2 clock periods
                    pulse_duration = self.check_pulse_length_short(
                        pulse_duration, 0.004)
                    channel_bit_mask = channel_bit_masks[i] + 2**22
                elif 0.005 < pulse_duration <= 0.007:
                    # 011 for 3 clock periods
                    pulse_duration = self.check_pulse_length_short(
                        pulse_duration, 0.006)
                    channel_bit_mask = channel_bit_masks[i] + 2**21 + 2**22
                elif 0.007 < pulse_duration <= 0.009:
                    # 100 for 4 clock periods
                    pulse_duration = self.check_pulse_length_short(
                        pulse_duration, 0.008)
                    channel_bit_mask = channel_bit_masks[i] + 2**23
                elif 0.009 < pulse_duration < 0.01:
                    # 100 for 4 clock periods
                    pulse_duration = self.check_pulse_length_short(
                        pulse_duration, 0.01)
                    channel_bit_mask = channel_bit_masks[i]

                # Shortest minimum instruction time is 5 clock periods
                # i.e. 10 ns for 500 MHz
                pulse_duration = self.pb_min_instr_clk_cycles

            # Time resolution of PulseBlaster, given by 1/(clock frequency):
            # t_min = 1e3/self.samprate  # in ns
            # upload times as is
            t_min = 1

            # Instructions for pulse sequence
            if i == 0:
                start_instr_num = spapi.pb_inst_pbonly(
                    channel_bit_mask, spapi.Inst.CONTINUE, 0, pulse_duration * t_min * spapi.us)
                self.error_catcher(start_instr_num)
            elif i != len(channel_bit_masks)-1:
                status = spapi.pb_inst_pbonly(
                    channel_bit_mask, spapi.Inst.CONTINUE, 0, pulse_duration * t_min * spapi.us)
                self.error_catcher(status)
            else:
                status = spapi.pb_inst_pbonly(
                    channel_bit_mask, spapi.Inst.BRANCH, start_instr_num, pulse_duration * t_min * spapi.us)
                self.error_catcher(status)

        status = spapi.pb_inst_pbonly(
            0, spapi.Inst.STOP, 0, self.pb_min_instr_clk_cycles * t_min * spapi.us)
        self.error_catcher(status)

        self.stop_programming()
        print(colored('Pulse sequence is loaded to Pulseblaster card!', 'green'))

    def start_programming(self) -> None:
        '''
            Starts the programming of PB pulse program
        '''
        # Returns a 0 on success or a negative number on an error.
        status = spapi.pb_start_programming(
            spapi.PULSE_PROGRAM)
        self.error_catcher(status)

    def stop_programming(self) -> None:
        '''
            Stops the programming of PB pulse program
        '''
        status = spapi.pb_stop_programming(
        )  # Returns a 0 on success or a negative number on an error.
        self.error_catcher(status)

    def _set_sampling_rate_attribute(self, samprate: int) -> None:
        self.samprate = samprate

    def _set_min_instr_clk_cycles(self, clk_cycles: int) -> None:
        # Minimum instruction time, in clock periods
        # (5 clk cycles or 10 ns for 500 MHz)
        self.pb_min_instr_clk_cycles = clk_cycles
