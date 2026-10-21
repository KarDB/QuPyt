"""Command-line failures must explain what failed without requiring --verbose."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


def _run_cli(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    """Exercise import-time CLI setup with an isolated user directory."""
    script = """
import logging
from pathlib import Path
import sys
from unittest.mock import Mock, patch

home = Path(sys.argv[1])
sys.argv = ["qupyt"]
with patch.object(Path, "home", return_value=home):
""" + textwrap.indent(
        textwrap.dedent(source), "    "
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


def test_default_cli_reports_malformed_instruction_and_recovery_path(
    tmp_path: Path,
) -> None:
    result = _run_cli(
        tmp_path,
        """
        import qupyt.main as service
        instruction = service.get_waiting_room() / "malformed.yaml"
        instruction.write_text("sensor: [unterminated", encoding="utf-8")
        try:
            service._process_instruction(instruction, Mock(), Mock())
        except Exception:
            pass
        logging.shutdown()
        """,
    )
    assert result.returncode == 0, result.stderr
    retained = list((tmp_path / ".qupyt" / "waiting_room").glob("*_failed_*"))
    logs = list((tmp_path / ".qupyt" / "log").glob("*.log"))
    assert len(retained) == len(logs) == 1
    assert "malformed.yaml" in result.stderr
    assert "ParserError" in result.stderr
    assert str(retained[0]) in result.stderr
    assert str(logs[0]) in result.stderr
    assert "Traceback (most recent call last)" not in result.stderr
    assert "Traceback (most recent call last)" in logs[0].read_text(encoding="utf-8")


def test_default_cli_reports_failed_measurement_status_and_original_cause(
    tmp_path: Path,
) -> None:
    result = _run_cli(
        tmp_path,
        """
        import qupyt.main as service
        instruction = service.get_waiting_room() / "camera.yaml"
        instruction.write_text("{}", encoding="utf-8")
        service._prepare_measurement = Mock(return_value=(Mock(), Mock()))
        def fail_measurement(static, dynamic, sensor, sync, params):
            params["measurement_error"] = {
                "type": "OSError",
                "message": "camera connection lost",
                "stage": "sensor acquisition",
            }
            return "failed"
        service.run_measurement = fail_measurement
        service._process_instruction(instruction, Mock(), Mock())
        logging.shutdown()
        """,
    )
    assert result.returncode == 0, result.stderr
    retained = next((tmp_path / ".qupyt" / "waiting_room").glob("*_failed_*"))
    logfile = next((tmp_path / ".qupyt" / "log").glob("*.log"))
    assert "camera.yaml" in result.stderr
    assert "OSError" in result.stderr
    assert "camera connection lost" in result.stderr
    assert "sensor acquisition" in result.stderr
    assert str(retained) in result.stderr
    assert str(logfile) in result.stderr


@pytest.mark.parametrize("failure_at", ["directories", "logfile"])
def test_cli_startup_failure_reports_cause_before_logging_is_available(
    tmp_path: Path, failure_at: str
) -> None:
    setup = (
        'patch("qupyt.set_up.make_userdirs", side_effect=PermissionError("user directory denied"))'
        if failure_at == "directories"
        else 'patch.object(logging, "FileHandler", side_effect=PermissionError("logfile denied"))'
    )
    result = _run_cli(
        tmp_path,
        f"""
        with {setup}:
            import qupyt.main
        """,
    )
    assert result.returncode != 0
    assert "PermissionError" in result.stderr
    assert (
        "user directory denied" if failure_at == "directories" else "logfile denied"
    ) in result.stderr
    assert "QuPyt" in result.stderr


def test_cli_import_failure_reports_dependency_and_preserves_file_traceback(
    tmp_path: Path,
) -> None:
    result = _run_cli(
        tmp_path,
        """
        import builtins
        original_import = builtins.__import__
        def unavailable_dependency(name, *args, **kwargs):
            if name == "qupyt.hardware.device_handler":
                raise ImportError("required driver dependency unavailable")
            return original_import(name, *args, **kwargs)
        with patch.object(builtins, "__import__", new=unavailable_dependency):
            import qupyt.main
        """,
    )
    assert result.returncode != 0
    assert "ImportError" in result.stderr
    assert "required driver dependency unavailable" in result.stderr
    logfile = next((tmp_path / ".qupyt" / "log").glob("*.log"))
    assert str(logfile) in result.stderr
    assert "Traceback (most recent call last)" in logfile.read_text(encoding="utf-8")


@pytest.mark.parametrize("failed_component", ["observer", "emitter"])
def test_service_reports_dead_instruction_watcher(
    service: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failed_component: str,
) -> None:
    emitter = Mock(is_alive=Mock(return_value=failed_component != "emitter"))
    observer = Mock(
        is_alive=Mock(return_value=failed_component != "observer"), emitters={emitter}
    )
    worker = Mock(is_alive=Mock(return_value=True))
    monkeypatch.setattr(service, "_get_observer", Mock(return_value=observer))
    monkeypatch.setattr(service.threading, "Thread", Mock(return_value=worker))
    monkeypatch.setattr(service, "_join_service_thread", Mock())
    # A missed health check must fail the test instead of leaving it running.
    monkeypatch.setattr(service, "sleep", Mock(side_effect=[None, KeyboardInterrupt]))

    component_name = "observer" if failed_component == "observer" else "watcher"
    with pytest.raises(RuntimeError, match=f"(?i){component_name}"):
        service.main()

    assert component_name in caplog.text.lower()
    observer.stop.assert_called_once()


def test_event_handler_initialization_failure_is_logged(
    service: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = ValueError("invalid waiting-room event filters")
    monkeypatch.setattr(service, "_get_observer_event_handler", Mock(side_effect=error))

    with pytest.raises(ValueError) as caught:
        service.main()

    assert caught.value is error
    assert "invalid waiting-room event filters" in caplog.text


def test_worker_initialization_failure_is_logged(
    service: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = OSError("cannot initialize device handler")
    monkeypatch.setattr(service, "DeviceHandler", Mock(side_effect=error))

    with pytest.raises(OSError) as caught:
        service.parse_input(threading.Event())

    assert caught.value is error
    assert "cannot initialize device handler" in caplog.text


def test_fatal_worker_system_exit_is_reported_on_default_cli(tmp_path: Path) -> None:
    result = _run_cli(
        tmp_path,
        """
        import threading
        from queue import Queue
        import qupyt.main as service
        service.queue = Queue()
        service.event_thread = threading.Event()
        service.DeviceHandler = Mock(side_effect=SystemExit("driver terminated initialization"))
        try:
            service.parse_input()
        except SystemExit:
            pass
        logging.shutdown()
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "SystemExit" in result.stderr
    assert "driver terminated initialization" in result.stderr
    assert "worker" in result.stderr.lower()


def test_join_reports_ongoing_shutdown_without_abandoning_worker(
    service: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    worker = Mock(ident=123, is_alive=Mock(side_effect=[True, False]))
    service._join_service_thread(worker, "measurement worker")

    output = capsys.readouterr()
    assert "measurement worker" in (output.err + output.out).lower()
    assert worker.join.call_count == 2
    assert all(call.kwargs.get("timeout") for call in worker.join.call_args_list)
