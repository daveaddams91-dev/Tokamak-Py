"""
Tests for Command-Line Interface and headless runs.
"""

import os
import argparse
import pytest
from tokamak_py.cli import run_simulation, list_presets


def test_cli_run_simulation_headless(tmp_path):
    output_path = str(tmp_path / "test_cli_output.png")
    args = argparse.Namespace(
        preset="tokamak_2d",
        particles=50,
        steps=10,
        dt=0.001,
        b_field=1.0,
        solver="barnes_hut",
        output=output_path,
        quiet=True
    )
    run_simulation(args)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 1000


def test_cli_list_presets(capsys):
    list_presets()
    captured = capsys.readouterr()
    assert "Tokamak Poloidal Slice" in captured.out
    assert "classic_2d" in captured.out
