import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from tdpy.verbosity import print


@pytest.mark.parametrize("verbosity", ["0", "1"])
def test_histogram_narrates_csv_and_png_paths(tmp_path, capsys, verbosity):
    data_directory = tmp_path / "data_outputs"
    output_directory = data_directory / "run6analysis"
    output_directory.mkdir(parents=True)
    for filename, values in (
        ("injected_transits_output6.csv", [2.0, 3.0, 5.0]),
        ("noninjected_transits_output6.csv", [2.2, 3.2, 4.8]),
    ):
        path = data_directory / filename
        print(f"Writing to {path}...")
        pd.DataFrame({"snr": values}).to_csv(path, index=False)
    capsys.readouterr()
    repository_path = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    python_paths = [str(repository_path / "datavis_scripts")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "TDPY_VERBOSITY": verbosity,
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", "import histogram"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=True,
        text=True,
    )

    output_path = output_directory / "snr_histogram_unmarked.png"
    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    if verbosity == "0":
        assert result.stdout == ""
    else:
        assert result.stdout.splitlines() == [
            "Reading from data_outputs/injected_transits_output6.csv...",
            "Reading from data_outputs/noninjected_transits_output6.csv...",
            "Writing to data_outputs/run6analysis/snr_histogram_unmarked.png...",
        ]