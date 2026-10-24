import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from tdpy.verbosity import print


@pytest.mark.parametrize("verbosity", ["0", "1"])
def test_poster_plots_narrates_inputs_and_all_figures(tmp_path, capsys, verbosity):
    data_directory = tmp_path / "data_outputs"
    output_directory = data_directory / "run6poster"
    output_directory.mkdir(parents=True)
    columns = {
        "r_s": [10.0, 11.0, 12.0, 13.0],
        "r_p": [0.1, 0.2, 0.3, 0.4],
        "a": [0.02, 0.04, 0.06, 0.08],
        "P_days": [2.0, 4.0, 6.0, 8.0],
        "inc": [80.0, 82.0, 84.0, 86.0],
        "period": [2.0, 4.0, 6.0, 8.0],
        "median": [0.01, 0.02, 0.03, 0.04],
        "mean": [0.01, 0.02, 0.03, 0.04],
        "max_depth": [0.01, 0.02, 0.03, 0.04],
        "oot_variability": [0.01, 0.02, 0.03, 0.04],
    }
    input_paths = (
        data_directory / "injected_transits_output6.csv",
        data_directory / "noninjected_transits_output6.csv",
    )
    for index, path in enumerate(input_paths):
        print(f"Writing to {path}...")
        pd.DataFrame(
            columns | {"snr": [2.0, 3.0, 5.0, 8.0] if index == 0 else [1.5, 2.5, 4.0, 6.0]}
        ).to_csv(path, index=False)
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
        [sys.executable, "-c", "import poster_plots"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=True,
        text=True,
    )

    output_paths = [
        output_directory / filename
        for filename in (
            "precision_recall_vs_threshold.png",
            "precision_recall_curve.png",
            "snr_histograms.png",
            "recovery_fraction_heatmap.png",
            "corner_parameters.png",
        )
    ]
    assert all(path.is_file() for path in output_paths)
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in output_paths)
    if verbosity == "0":
        assert result.stdout == ""
    else:
        output_lines = result.stdout.splitlines()
        assert output_lines[:2] == [
            f"Reading from {path.relative_to(tmp_path)}..." for path in input_paths
        ]
        assert output_lines[2].startswith("[INFO] F1-optimal detection threshold:")
        assert output_lines[3:8] == [
            f"Writing to {path.relative_to(tmp_path)}..." for path in output_paths
        ]
        assert output_lines[8].startswith("[DONE] Saved figures:")