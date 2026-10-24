import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from tdpy.verbosity import print


@pytest.mark.parametrize("verbosity", ["0", "1"])
def test_multiplot_narrates_csv_and_png_paths(tmp_path, verbosity):
    data_directory = tmp_path / "data_outputs"
    output_directory = data_directory / "run6analysis"
    output_directory.mkdir(parents=True)
    input_path = data_directory / "injected_transits_output6.csv"
    values = range(12)
    print(f"Writing to {input_path}...")
    pd.DataFrame(
        {
            "r_s": [10.0] * 12,
            "r_p": [1.0 + index / 10 for index in values],
            "P_days": [2.0 + index for index in values],
            "inc": [80.0 + index for index in values],
            "snr": [1.0 + index for index in values],
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
        [sys.executable, "-c", "import multiplot"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=True,
        text=True,
    )

    output_path = output_directory / "snr_vs_parameters_smooth_summary6.png"
    assert output_path.is_file()
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    if verbosity == "0":
        assert result.stdout == ""
    else:
        assert result.stdout.splitlines() == [
            "Reading from data_outputs/injected_transits_output6.csv...",
            "Loaded file: data_outputs/injected_transits_output6.csv  (rows: 12)",
            "Dropped 0 rows with invalid/missing snr or parameters. Remaining rows: 12",
            "Writing to data_outputs/run6analysis/snr_vs_parameters_smooth_summary6.png...",
            "Done: single figure with 6 subplots generated. Adjust input_csv, n_bins_1d, n_bins_2d, sigma as needed.",
        ]