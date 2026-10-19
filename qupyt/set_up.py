"""Provide QuPyt user-directory paths and create missing directories."""

from __future__ import annotations

import pathlib
from pathlib import Path


def get_log_dir() -> Path:
    """Return the user logging directory.

    Returns
    -------
    pathlib.Path
        Directory for logging output.
    """
    return _make_user_dir_list()["homepath_log"]


def get_seq_dir() -> Path:
    """Return the user pulse-sequence directory.

    Returns
    -------
    pathlib.Path
        Directory for writing and reading pulse sequences.
    """
    return _make_user_dir_list()["homepath_sequences"]


def get_waiting_room() -> Path:
    """Return the measurement instruction waiting-room directory.

    Returns
    -------
    pathlib.Path
        Directory for writing and reading measurement instruction files.
    """
    return _make_user_dir_list()["homepath_waitingroom"]


def get_home_dir() -> Path:
    """Return the QuPyt user-data directory.

    Returns
    -------
    pathlib.Path
        Base directory containing status files and other QuPyt user data.
    """
    return _make_user_dir_list()["homepath"]


def _make_user_dir_list() -> dict[str, Path]:
    homepath = pathlib.Path.home() / ".qupyt"
    homepath_log = pathlib.Path.home() / ".qupyt/log"
    homepath_sequences = pathlib.Path.home() / ".qupyt/sequences"
    homepath_waitingroom = pathlib.Path.home() / ".qupyt/waiting_room"
    pathdict = {
        "homepath": homepath,
        "homepath_log": homepath_log,
        "homepath_sequences": homepath_sequences,
        "homepath_waitingroom": homepath_waitingroom,
    }
    return pathdict


def make_userdirs() -> None:
    """Create the required QuPyt user directories if they are missing."""
    dirs = _make_user_dir_list()
    for directory in dirs.values():
        directory.mkdir(exist_ok=True)
