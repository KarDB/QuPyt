"""
This file handles the use of different synchronisers.
By reading the parameters passed through the instructions file
"""
import importlib.util
from types import ModuleType
from pathlib import Path
from typing import Dict, Any, Optional, Protocol, cast


# pylint: disable=too-few-public-methods
class UserPulseSeqProtocol(Protocol):
    """
    Protocol to define that every generate_sequence
    function takes a dict as input and returns either
    None or another dict.
    """

    def generate_sequence(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        :param params: Input to pass the parameters needed
         to constuct the pulse sequence.
        :type params: dict[str, any]
        :return: Either returns a dictionary with updated, or dependent
         parameters, or None.
        :rtype: "dict[str, any] | None"
        """


def _load_module_from_path(path: Path) -> ModuleType:
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


def write_user_ps(path: Path, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Load user specified pulse sequence definition and
    execute it to generate the pulse sequence.
    """
    user_ps = cast(UserPulseSeqProtocol, _load_module_from_path(path))
    dependent_parameters = user_ps.generate_sequence(params)
    return dependent_parameters


def update_params_dict(
    params: Dict[str, Any], update_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update the parameters dictonary. This is done via recursion to reach
    nested parameter dicts without overwriting the full config.

    :param params: Full configuration dictionary to be updated.
    :type param: Dict[str, Any]
    :param update_params: Dictionary with the values to be updated.
    :type update_params: Dict[str, Any]
    """

    for key, value in update_params.items():
        if isinstance(value, dict):
            params[key] = update_params_dict(params.get(key, {}), value)
        else:
            params[key] = value

    return params
