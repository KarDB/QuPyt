"""Manage VISA connections and device-specific signal-source commands.

Store device commands in a common dictionary so signal-source operations share
a consistent interface.
"""

from __future__ import annotations

import logging
from time import sleep
from typing import Any

import pyvisa

from qupyt.mixins import ConfigurationError
from qupyt.utils.errors import cleanup_after_error


class VisaObject:
    """Base connection for devices controlled through the VISA protocol."""

    # PyVISA chooses a backend-specific resource class at runtime.
    instance: Any

    def __init__(self, handle: str, s_type: str) -> None:
        """Connect to a supported VISA device and load its command set.

        Parameters
        ----------
        handle : str
            VISA resource address of the signal source.
        s_type : str
            Device type: SRS, SMA, SMB, Rigol, TekAWG, or TekAFG.

        Raises
        ------
        ConfigurationError
            If the device type is unsupported.
        """
        self.known_s_types = ["SRS", "SMA", "SMB", "Rigol", "TekAWG", "TekAFG"]
        self.handle = handle
        self.s_type = s_type
        self.instance = None
        if self.s_type not in self.known_s_types:
            raise ConfigurationError(
                "the VISA device type", self.s_type, self.known_s_types
            )
        self.command: dict[str, str]
        self._get_instructions()
        try:
            resource_manager = pyvisa.ResourceManager()
            self.instance = resource_manager.open_resource(handle)
            self.instance.timeout = 60000
            logging.info(
                f"Opening {s_type} at adress {handle}".ljust(65, ".") + "[done]"
            )
        except BaseException:
            logging.exception(
                f"Opening {s_type} at adress {handle}".ljust(65, ".") + "[failed]"
            )
            # Resource managers are shared; closing one also closes other devices.
            if self.instance is not None:
                cleanup_after_error(
                    lambda: VisaObject.close(self), f"VISA setup cleanup at {handle}"
                )
            raise

    def __repr__(self) -> str:
        return f"VisaObject(handle: {self.handle}, s_type: {self.s_type})"

    def __str__(self) -> str:
        return f"VisaObject(handle: {self.handle}, s_type: {self.s_type})"

    def _get_instructions(self) -> None:
        """Select the command set for the configured signal-source type."""
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

        elif self.s_type == "SMA":
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
                "GetBurstNCycles1": "BURS:NCYC?",
                "SetBurstNCycles1": "BURS:NCYC ",
                "SetBurstMode1": "BURS:MODE ",
                "SetBurstState1": "BURS:STAT ",
                "SetBurstPhase1": ":SOUR1:BURSt:PHASe ",
                "GetBurstMode1": "BURS:MODE?",
                "Outp1": "OUTP ",
                "GetOutp1": "OUTP?",
                "GetAmpl2": "SOUR2:VOLT?",
                "SetAmpl2": "SOUR2:VOLT ",
                "SetPhase2": "SOUR2:BURS:PHAS ",
                "GetPhase2": "SOUR2:BURS:PHAS?",
                "GetFreq2": "SOUR2:FREQ?",
                "SetFreq2": "SOUR2:FREQ ",
                "GetBurstNCycles2": "SOUR2:BURS:NCYC?",
                "SetBurstNCycles2": "SOUR2:BURS:NCYC ",
                "SetBurstMode2": "SOUR2:BURS:MODE ",
                "SetBurstState2": "SOUR2:BURS:STAT ",
                "SetBurstPhase2": ":SOUR2:BURSt:PHASe ",
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
                "SetPhase1": "SOURce1:PHASe:ADJust ",
                "SetBurstMode1": "SOURce1:BURSt:MODE ",
                "SetBurstState1": "SOURce1:BURSt:STATe ",
                "SetAmpl2": "SOURce2:VOLTage:LEVel:IMMediate:AMPLitude ",
                "SetFreq2": "SOURce2:FREQuency:FIXed ",
                "SetPhase2": "SOURce2:PHASe:ADJust ",
                # The Tek AFG does not implement an OPC.
                # We therefore skip the waiting time and
                # Query impedance which will alwasy return
                # Non zeros numbers.
                "OPC": "OUTPut1:IMPedance?",
            }

    def opc_wait(self) -> None:
        """Wait until the device reports that its pending operations are complete."""
        last_response = ""
        logging.debug(
            "Waiting for %s at %s to complete pending operations",
            self.s_type,
            self.handle,
        )
        for attempt in range(3):
            try:
                opc = self.instance.query(self.command["OPC"])
            except Exception:
                logging.exception(
                    "Failed to query OPC for %s at %s", self.s_type, self.handle
                )
                raise
            last_response = opc
            try:
                if float(opc.strip()) != 0:
                    logging.debug(
                        "%s at %s completed pending operations",
                        self.s_type,
                        self.handle,
                    )
                    return
            except ValueError:
                logging.warning(
                    "%s returned invalid OPC response %r; retrying (%s/3)",
                    self.s_type,
                    opc,
                    attempt + 1,
                )
            sleep(0.1)
        logging.error(
            "%s at %s timed out waiting for OPC; last_response=%r",
            self.s_type,
            self.handle,
            last_response,
        )
        raise TimeoutError(
            f"{self.s_type} did not return a valid OPC response. "
            f"Last response was {last_response!r}."
        )

    def close(self) -> None:
        if getattr(self, "instance", None) is None:
            return
        if self.s_type == "TekAWG":
            logging.debug(
                "Waiting 5 seconds before closing TekAWG at %s to prevent TCP/IP issues",
                self.handle,
            )
            sleep(5)
        try:
            self.instance.close()
            self.instance = None
            logging.info(
                f"Closing {self.s_type} at adress {self.handle}".ljust(65, ".")
                + "[done]"
            )
        except Exception:
            logging.exception(
                "Closing %s at address %s failed", self.s_type, self.handle
            )
            raise
