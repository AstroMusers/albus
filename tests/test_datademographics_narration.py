import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from tdpy.verbosity import print


@pytest.mark.parametrize("verbosity", ["0", "1"])
def test_datademographics_narrates_input_and_output(tmp_path, verbosity):
    data_directory = tmp_path / "data_outputs"
    output_directory = data_directory / "run6analysis"
    output_directory.mkdir(parents=True)
    input_path = data_directory / "injected_transits_output6.csv"
    print(f"Writing to {input_path}...")
    pd.DataFrame(
        {
            "r_p": [0.1, 0.2, 0.3],
            "r_s": [1.0, 1.1, 1.2],
            "P_days": [2.0, 5.0, 12.0],
            "inc": [88.0, 89.0, 90.0],
        }
    ).to_csv(input_path, index=False)
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
        [sys.executable, "-c", "import datademographics"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=True,
        text=True,
    )

    output_path = output_directory / "histograms_radius_period_inclination6.png"
    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    if verbosity == "0":
        assert result.stdout == ""
    else:
        assert result.stdout.splitlines() == [
            "Reading from data_outputs/injected_transits_output6.csv...",
            "88.0 90.0",
            "Writing to data_outputs/run6analysis/histograms_radius_period_inclination6.png...",
        ]