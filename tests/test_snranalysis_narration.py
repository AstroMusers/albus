import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from tdpy.verbosity import print


@pytest.mark.parametrize("verbosity", ["0", "1"])
def test_snranalysis_narrates_csv_reads_and_count(tmp_path, capsys, verbosity):
    data_directory = tmp_path / "data_outputs"
    data_directory.mkdir()
    input_rows = (
        ("injected_transits_output1.csv", "snr", [0.5, 1.5, 2.5]),
        ("noninjected_transits_output1.csv", " snr", [0.25, 1.0, 2.0]),
    )
    for filename, column, values in input_rows:
        path = data_directory / filename
        print(f"Writing to {path}...")
        pd.DataFrame({column: values}).to_csv(path, index=False)
    capsys.readouterr()
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
        [sys.executable, "-c", "import snranalysis"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=True,
        text=True,
    )

    if verbosity == "0":
        assert result.stdout == ""
    else:
        assert result.stdout.splitlines() == [
            "Reading from data_outputs/injected_transits_output1.csv...",
            "2",
            "Reading from data_outputs/noninjected_transits_output1.csv...",
        ]