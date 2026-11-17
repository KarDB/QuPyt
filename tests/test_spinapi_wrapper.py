"""Exercise the original SpinAPI wrapper with native callbacks and no hardware."""

from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from qupyt.hardware import synchronisers


def load_wrapper() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "qupyt/hardware/wrappers/spinapi_adapted.py"
    )
    spec = importlib.util.spec_from_file_location("qupyt_spinapi_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native_api(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, MagicMock]:
    library = MagicMock()
    monkeypatch.setattr(ctypes, "CDLL", Mock(return_value=library))
    return load_wrapper(), library


def attach_callback(
    library: MagicMock,
    name: str,
    result: int | float,
    array_lengths: dict[int, int] | None = None,
) -> Mock:
    """Use the production declaration to check ctypes argument conversion."""
    symbol = getattr(library, name)
    callback = Mock(return_value=result)
    prototype = ctypes.CFUNCTYPE(symbol.restype, *symbol.argtypes)

    def record(*arguments: Any) -> int | float:
        # Copy pointers while the production wrapper's temporary arrays exist.
        lengths = array_lengths or {}
        callback(
            *(
                argument[: lengths[index]] if index in lengths else argument
                for index, argument in enumerate(arguments)
            )
        )
        return result

    setattr(library, name, prototype(record))
    return callback


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("pb_core_clock", (500.0,)),
        ("pb_set_freq", (12.5,)),
        ("pb_set_phase", (90.0,)),
        ("pb_set_amp", (0.25, 3)),
        ("pb_start_programming", (0,)),
        ("pb_inst_pbonly", (17, 6, 0, 2500.0)),
    ],
)
def test_native_scalar_arguments_and_status_are_preserved(
    native_api: tuple[ModuleType, MagicMock],
    name: str,
    arguments: tuple[int | float, ...],
) -> None:
    module, library = native_api
    callback = attach_callback(library, name, 42)
    assert getattr(module, name)(*arguments) == 42
    callback.assert_called_once_with(*arguments)


def test_dds_load_passes_float_array_pointer(
    native_api: tuple[ModuleType, MagicMock]
) -> None:
    module, library = native_api
    callback = attach_callback(library, "pb_dds_load", 0, {0: 3})
    assert module.pb_dds_load([0.25, -0.5, 1.0], 1) == 0
    samples, device = callback.call_args.args
    assert samples == [0.25, -0.5, 1.0]
    assert device == 1


def test_pulseblaster_instruction_helper_uses_native_wrapper(
    native_api: tuple[ModuleType, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, library = native_api
    callback = attach_callback(library, "pb_inst_pbonly", 42)
    monkeypatch.setattr(synchronisers, "spapi", module)
    synchroniser = synchronisers.PulseBlaster({}, {})
    assert synchroniser.pb_inst_pbonly(17, 6, 0, 2500.5) == 42
    callback.assert_called_once_with(17, 6, 0, 2500.5)


def test_fft_resonance_returns_double_and_passes_integer_arrays(
    native_api: tuple[ModuleType, MagicMock]
) -> None:
    module, library = native_api
    callback = attach_callback(library, "pb_fft_find_resonance", 123.75, {3: 2, 4: 2})
    assert module.pb_fft_find_resonance(2, 1.5e6, 2500.5, [10, 20], [-1, -2]) == 123.75
    count, frequency, width, real, imaginary = callback.call_args.args
    assert (count, frequency, width) == (2, 1.5e6, 2500.5)
    assert real == [10, 20]
    assert imaginary == [-1, -2]


@pytest.mark.parametrize("name", ["pb_write_ascii_verbose", "pb_write_jcamp"])
def test_native_file_writer_passes_filename_scalars_and_arrays(
    native_api: tuple[ModuleType, MagicMock], name: str
) -> None:
    module, library = native_api
    callback = attach_callback(library, name, 7, {4: 2, 5: 2})
    assert getattr(module, name)("spectrum.txt", 2, 12.5, 30.25, [1, 2], [3, 4]) == 7
    filename, count, width, frequency, real, imaginary = callback.call_args.args
    assert (filename, count, width, frequency) == (b"spectrum.txt", 2, 12.5, 30.25)
    assert real == [1, 2]
    assert imaginary == [3, 4]


def test_felix_writer_passes_integer_array_pointers(
    native_api: tuple[ModuleType, MagicMock]
) -> None:
    module, library = native_api
    callback = attach_callback(library, "pb_write_felix", 0, {5: 2, 6: 2})
    assert (
        module.pb_write_felix("spectrum.felix", "title", 2, 12.5, 30.25, [1, 2], [3, 4])
        == 0
    )
    filename, title, count, width, frequency, real, imaginary = callback.call_args.args
    assert (filename, title, count, width, frequency) == (
        b"spectrum.felix",
        b"title",
        2,
        12.5,
        30.25,
    )
    assert real == [1, 2]
    assert imaginary == [3, 4]


@pytest.mark.parametrize(
    "name", ["pb_get_version", "pb_get_error", "pb_status_message"]
)
def test_native_strings_decode_utf8(
    native_api: tuple[ModuleType, MagicMock], name: str
) -> None:
    module, library = native_api
    getattr(library, name).return_value = "board ready µs".encode("utf-8")
    assert getattr(module, name)() == "board ready µs"


@pytest.mark.parametrize(
    "name", ["pb_get_version", "pb_get_error", "pb_status_message"]
)
def test_null_native_string_has_descriptive_error(
    native_api: tuple[ModuleType, MagicMock], name: str
) -> None:
    module, library = native_api
    getattr(library, name).return_value = None
    with pytest.raises(RuntimeError, match=f"{name} returned a null string pointer"):
        getattr(module, name)()


def test_native_library_falls_back_to_alternate_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = MagicMock()
    loader = Mock(side_effect=[OSError("spinapi64 unavailable"), library])
    monkeypatch.setattr(ctypes, "CDLL", loader)
    module = load_wrapper()
    assert module.spinapi is library
    assert [call.args for call in loader.call_args_list] == [
        ("spinapi64",),
        ("spinapi",),
    ]


def test_missing_native_libraries_preserve_loading_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = OSError("spinapi unavailable")
    monkeypatch.setattr(
        ctypes, "CDLL", Mock(side_effect=[OSError("spinapi64 unavailable"), error])
    )
    with pytest.raises(ImportError, match="Could not load either") as caught:
        load_wrapper()
    assert caught.value.__cause__ is error
