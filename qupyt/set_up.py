"""
This is run at every startup up qupyt to make sure
all necessary user directories exist, and provide
get functions to query to correct path
on the user's machine.
"""
import pathlib
from pathlib import Path
from typing import Dict


def get_log_dir() -> Path:
    """
    :return: Directory for logging output.
    :rtype: Path
    """
    return _make_user_dir_list()['homepath_log']


def get_seq_dir() -> Path:
    """
    :return: Directory to which pulse sequences
     will be written and read.
    :rtype: Path
    """
    return _make_user_dir_list()['homepath_sequences']


def get_waiting_room() -> Path:
    """
    :return: Directory to which measurement instruction
     files will be written and read.
    :rtype: Path
    """
    return _make_user_dir_list()['homepath_waitingroom']


def _make_user_dir_list() -> Dict[str, Path]:
    homepath = pathlib.Path.home() / '.qupyt'
    homepath_log = pathlib.Path.home() / '.qupyt/log'
    homepath_sequences = pathlib.Path.home() / '.qupyt/sequences'
    homepath_waitingroom = pathlib.Path.home() / '.qupyt/waiting_room'
    pathdict = {'homepath': homepath,
                'homepath_log': homepath_log,
                'homepath_sequences': homepath_sequences,
                'homepath_waitingroom': homepath_waitingroom}
    return pathdict


def make_userdirs() -> None:
    """
    Create all necessary user directories if they don't
    yet exists.
    """
    dirs = _make_user_dir_list()
    for directory in dirs.values():
        directory.mkdir(exist_ok=True)
