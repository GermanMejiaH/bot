"""Unit tests for DTA CLI commands in src/dta/cli/main.py."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from dta.cli.main import app

runner = CliRunner()


def test_cli_list_windows_command() -> None:
    with patch("pygetwindow.getWindowsWithTitle") as mock_gw:
        win1 = MagicMock()
        win1.title = "Dofus - Character"
        win1.left = 10
        win1.top = 20
        win1.width = 1920
        win1.height = 1080
        mock_gw.return_value = [win1]

        result = runner.invoke(app, ["list-windows", "--query", "Dofus"])

        assert result.exit_code == 0
        assert "Found 1 open window" in result.output
        assert "Dofus - Character" in result.output


def test_cli_screenshot_missing_window() -> None:
    with patch("dta.vision.window_finder.WindowFinder.is_window_available", return_value=False):
        result = runner.invoke(app, ["screenshot", "--window-title", "NonExistentWindow"])
        assert result.exit_code == 1
        assert "not found" in result.output


def test_cli_debug_missing_window() -> None:
    with patch("dta.vision.window_finder.WindowFinder.is_window_available", return_value=False):
        result = runner.invoke(app, ["debug", "--window-title", "NonExistentWindow"])
        assert result.exit_code == 1
        assert "not found" in result.output
