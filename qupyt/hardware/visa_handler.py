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


class VisaObject:
    """Visa class acting as parent for all devices intended to connet via the VISA protocol."""

    def __init__(self, handle: str, s_type: str) -> None:
        """
        handle: visa adress of signal source
        s_type: source type (SRS, RS)...
        """
        self.handle = handle
        self.s_type = s_type
        self.command: Dict[str, str]
        self._get_instructions()
        try:
            rm = pyvisa.ResourceManager()
            self.instance = rm.open_resource(handle)
            self.instance.timeout = 60000
            logging.info(
                "Opening {} at adress {}".format(s_type, handle).ljust(65, ".")
                + "[done]"
            )
        except Exception:
            rm.close()
            rm.visalib._registry.clear()
            print('Opening VISA Object Failed')
            logging.exception(
                "Opening {} at adress {}".format(s_type, handle).ljust(65, ".")
                + "[failed]"
            )

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

        if self.s_type == "SMB":
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

    def opc_wait(self) -> None:
        opcVal = 0
        while opcVal == 0:
            opc = self.instance.query(self.command["OPC"])
            opcVal = int(opc)

    def close(self) -> None:
        if self.s_type == "TekAWG":
            print("Sleeping for 5 seconds in close to prevent TCPIP issues:")
            sleep(5)
        try:
            self.instance.close()
            self.instance.visalib._registry.clear()
            logging.info(
                "Closing {} at adress {}"
                .format(self.s_type, self.handle)
                .ljust(65, ".")
                + "[done]"
            )
        except Exception:
            logging.exception('Closing {} at adress {} failed'.format(
                self.s_type, self.handle).ljust(65, '.') + '[failed]')
