import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from tdpy.verbosity import print


@pytest.mark.parametrize("verbosity", ["0", "1"])
def test_metric_corner_cli_narrates_paths(tmp_path, verbosity):
    columns = {
        "r_p": [0.1, 0.2, 0.3, 0.4],
        "r_s": [10.0, 11.0, 12.0, 13.0],
        "a": [0.02, 0.04, 0.06, 0.08],
        "P_days": [2.0, 4.0, 6.0, 8.0],
        "inc": [80.0, 82.0, 84.0, 86.0],
    }
    pos_path = tmp_path / "positive.csv"
    neg_path = tmp_path / "negative.csv"
    for path, snr_values in ((pos_path, [2.0, 3.0, 5.0, 8.0]), (neg_path, [1.0, 2.0, 4.0, 6.0])):
        print(f"Writing to {path}...")
        pd.DataFrame(columns | {"snr": snr_values}).to_csv(path, index=False)
    output_directory = tmp_path / "results"
    repository_path = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    python_paths = [str(repository_path / "datavis_scripts")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "PYTHONPATH": os.pathsep.join(python_paths),
            "TDPY_VERBOSITY": verbosity,
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            str(repository_path / "datavis_scripts" / "metric_corner_plots.py"),
            "--pos", str(pos_path),
            "--neg", str(neg_path),
            "--outdir", str(output_directory),
            "--threshold", "3.7",
            "--bins", "3",
            "--min-total", "1",
            "--min-pred", "1",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=True,
        text=True,
    )

    output_paths = [output_directory / "corner_precision.png", output_directory / "corner_recall.png"]
    for path in output_paths:
        assert path.is_file()
        print(f"Reading from {path}...")
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    if verbosity == "0":
        assert result.stdout == ""
    else:
        assert result.stdout.splitlines() == [
            f"Reading from {pos_path}...",
            f"Reading from {neg_path}...",
            "[INFO] Fixed threshold: 3.7000",
            *(f"Writing to {path}..." for path in output_paths),
        ]