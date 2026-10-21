"""Load user pulse generators and merge their dependent measurement parameters."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any, Protocol, cast

from qupyt.set_up import get_seq_dir


# pylint: disable=too-few-public-methods
class UserPulseSeqProtocol(Protocol):
    """Protocol for generators accepting parameters and returning updates or None."""

    def generate_sequence(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """Generate the pulse sequence from the supplied measurement parameters.

        Parameters
        ----------
        params : dict[str, Any]
            Parameters needed to construct the experiment's pulse sequence.

        Returns
        -------
        dict[str, Any] or None
            Updated or derived measurement parameters, or None when no updates are needed.
        """


def _load_module_from_path(path: Path) -> ModuleType:
    logging.debug("Loading user pulse-sequence module from %s", path)
    spec = importlib.util.spec_from_file_location("user_pulse_seq", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Module user_pulse_seq cannot be loaded from path {path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure the loader can execute the module
    if hasattr(spec.loader, "exec_module"):
        spec.loader.exec_module(module)
    else:
        raise ImportError("The loader for user_pulse_seq doesn't support execution")
    return module


def write_user_ps(path: Path, params: dict[str, Any]) -> dict[str, Any] | None:
    """Load a user pulse-generator module and execute its generate_sequence function.

    Parameters
    ----------
    path : pathlib.Path
        Path to the user pulse-generator module.
    params : dict[str, Any]
        Measurement parameters passed to the generator.

    Returns
    -------
    dict[str, Any] or None
        Dependent parameters returned by the generator.

    Raises
    ------
    ImportError
        If the module cannot be loaded.
    TypeError
        If the generator returns a value other than a dictionary or None.
    """
    started = perf_counter()
    sequence_dir = get_seq_dir()
    logging.info(
        "Generating pulse sequences using %s; output_directory=%s", path, sequence_dir
    )
    try:
        previous_sequence_files = {
            sequence_file: sequence_file.stat().st_mtime_ns
            for sequence_file in sequence_dir.glob("sequence_*.yaml")
        }
        user_ps = cast(UserPulseSeqProtocol, _load_module_from_path(path))
        dependent_parameters = user_ps.generate_sequence(params)
        if dependent_parameters is not None and not isinstance(
            dependent_parameters, dict
        ):
            raise TypeError(
                "generate_sequence must return a parameter dictionary or None"
            )
        changed_sequence_files = [
            sequence_file
            for sequence_file in sequence_dir.glob("sequence_*.yaml")
            if previous_sequence_files.get(sequence_file)
            != sequence_file.stat().st_mtime_ns
        ]
    except Exception:
        logging.exception("Pulse sequence generation failed for %s", path)
        raise
    logging.info(
        "Pulse sequence generation completed: module=%s, changed_files=%s, elapsed=%.3fs",
        path,
        len(changed_sequence_files),
        perf_counter() - started,
    )
    logging.debug("Generated pulse sequence files: %s", changed_sequence_files)
    return dependent_parameters


def update_params_dict(
    params: dict[str, Any], update_params: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge dependent parameters recursively into the measurement configuration.

    Parameters
    ----------
    params : dict[str, Any]
        Measurement configuration to update in place.
    update_params : dict[str, Any] or None
        New values, including nested dictionaries. None leaves params unchanged.

    Returns
    -------
    dict[str, Any]
        The updated params dictionary.

    Raises
    ------
    TypeError
        If non-None inputs are not dictionaries.
    """
    if update_params is None:
        return params
    if not isinstance(params, dict) or not isinstance(update_params, dict):
        raise TypeError("Measurement parameters and their updates must be dictionaries")
    for key, value in update_params.items():
        if isinstance(value, dict):
            params[key] = update_params_dict(params.get(key, {}), value)
        else:
            logging.debug(
                "Updating dependent measurement parameter %s: %r -> %r",
                key,
                params.get(key),
                value,
            )
            params[key] = value
    return params
