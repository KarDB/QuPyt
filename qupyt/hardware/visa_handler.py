"""
Create one Class to handle all connections through the VISA library.
For different microwave sources or visa devices, the commands to set e.g.
an output power are unified by storing the device speicif commands in a
dictionary for each device.
"""
from time import sleep
import logging
from typing import Dict
import pyvisa
from qupyt.mixins import ConfigurationError


class VisaObject:
    """
    Visa class acting as parent for all devices intended
    to connect via the VISA protocol.
    """

    def __init__(self, handle: str, s_type: str) -> None:
        """
        handle: visa adress of signal source
        s_type: source type (SRS, RS)...
        """
        self.known_s_types = ["SRS", "SMB", "Rigol", "TekAWG", "TekAFG"]
        self.handle = handle
        self.s_type = s_type
        if self.s_type not in self.known_s_types:
            raise ConfigurationError(
                'the VISA device type', self.s_type, self.known_s_types)
        self.command: Dict[str, str]
        self._get_instructions()
        try:
            resource_manager = pyvisa.ResourceManager()
            self.instance = resource_manager.open_resource(handle)
            self.instance.timeout = 60000
            logging.info(
                f"Opening {s_type} at adress {handle}".ljust(65, ".")
                + "[done]"
            )
        except Exception as exc:
            logging.exception(
                f"Opening {s_type} at adress {handle}".ljust(65, ".")
                + "[failed]"
            )
            resource_manager.close()
            resource_manager.visalib._registry.clear()
            raise exc

    def __repr__(self) -> str:
        return f'VisaObject(handle: {self.handle}, s_type: {self.s_type})'

    def __str__(self) -> str:
        return f'VisaObject(handle: {self.handle}, s_type: {self.s_type})'

    def _get_instructions(self) -> None:
        """
        Get set of instructions depending
        on type of signal source.
        """
        if self.s_type == "SRS":
            self.command = {
                "SetAmpl1": "AMPR ",
                "GetAmpl1": "AMPR?",
                "SetFreq1": "FREQ ",
                "GetFreq1": "FREQ?",
                "OPC": "*OPC?",
            }

        elif self.s_type == "SMB":
            self.command = {
                "SetAmpl1": "POW ",
                "GetAmpl1": "POW?",
                "SetFreq1": "FREQ ",
                "GetFreq1": "FREQ?",
                "OPC": "*OPC?",
            }

        elif self.s_type == "Rigol":
            self.command = {
                "OPC": "*OPC?",
                "GetAmpl1": "VOLT?",
                "SetAmpl1": "VOLT ",
                "SetPhase1": "BURS:PHAS ",
                "GetPhase1": "BURS:PHAS?",
                "GetFreq1": "FREQ?",
                "SetFreq1": "FREQ ",
                "GetNCycles1": "BURS:NCYC?",
                "SetNCycles1": "BURS:NCYC ",
                "SetBurstMode1": "BURS:MODE ",
                "GetBurstMode1": "BURS:MODE?",
                "Outp1": "OUTP ",
                "GetOutp1": "OUTP?",
                "GetAmpl2": "SOUR2:VOLT?",
                "SetAmpl2": "SOUR2:VOLT ",
                "SetPhase2": "SOUR2:BURS:PHAS ",
                "GetPhase2": "SOUR2:BURS:PHAS?",
                "GetFreq2": "SOUR2:FREQ?",
                "SetFreq2": "SOUR2:FREQ ",
                "GetNCycles2": "SOUR2:BURS:NCYC?",
                "SetNCycles2": "SOUR2:BURS:NCYC ",
                "SetBurstMode2": "SOUR2:BURS:MODE ",
                "GetBurstMode2": "SOUR2:BURS:MODE?",
                "Outp2": "OUTP2 ",
                "GetOutp2": "OUTP2?",
            }

        elif self.s_type == "TekAWG":
            self.command = {"OPC": "*OPC?"}

        elif self.s_type == "TekAFG":
            self.command = {
                "SetAmpl1": "SOURce1:VOLTage:LEVel:IMMediate:AMPLitude ",
                "GetAmpl1": "SOURce1:VOLTage:LEVel:IMMediate:AMPLitude?",
                "SetFreq1": "SOURce1:FREQuency:FIXed ",
                "GetFreq1": "SOURce1:FREQuency:FIXed?",
                # The Tek AFG does not implement an OPC.
                # We therefore skip the waiting time and
                # Query impedance which will alwasy return
                # Non zeros numbers.
                "OPC": "OUTPut1:IMPedance?",
            }

    def opc_wait(self) -> None:
        """
        Check if the device has finished all tasks and is
        ready to execute the next command.
        Pauses execution until the device is ready.
        """
        opc_val = 0
        while opc_val == 0:
            opc = self.instance.query(self.command["OPC"])
            opc_val = int(opc)

    def close(self) -> None:
        if self.s_type == "TekAWG":
            print("Sleeping for 5 seconds in close to prevent TCPIP issues:")
            sleep(5)
        try:
            self.instance.close()
            self.instance.visalib._registry.clear()
            logging.info(
                f"Closing {self.s_type} at adress {self.handle}"
                .ljust(65, ".")
                + "[done]"
            )
        except Exception:
            logging.exception('Closing {self.s_type} at adress {self.handle} failed'
                              .ljust(65, '.') + '[failed]')
