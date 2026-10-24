import sys
from pathlib import Path

import pandas as pd
from tdpy.verbosity import print

from datavis_scripts import statcurves


def test_statcurves_narrates_inputs_and_outputs(tmp_path, monkeypatch, capsys):
    positive_path = tmp_path / "positive.csv"
    negative_path = tmp_path / "negative.csv"
    output_directory = tmp_path / "results"
    for path, values in ((positive_path, [2.0, 8.0]), (negative_path, [1.0, 3.0])):
        print(f"Writing to {path}...")
        pd.DataFrame({"snr": values}).to_csv(path, index=False)
    capsys.readouterr()
    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "statcurves.py",
            "--pos", str(positive_path),
            "--neg", str(negative_path),
            "--outdir", str(output_directory),
            "--n-thresholds", "12",
        ],
    )

    statcurves.main()

    output_paths = [
        output_directory / "metrics_by_threshold.csv",
        output_directory / "f1_curve.png",
        output_directory / "precision_curve.png",
        output_directory / "recall_curve.png",
        output_directory / "precision_recall_curve_separate.png",
        output_directory / "pr_curve.png",
        output_directory / "roc_curve.png",
        output_directory / "snr_hist.png",
    ]
    output = capsys.readouterr().out.splitlines()
    assert output == [
        f"Reading from {positive_path}...",
        f"Reading from {negative_path}...",
        *(f"Writing to {path}..." for path in output_paths),
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in output_paths)
    assert all(
        path.suffix != ".png" or path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        for path in output_paths
    )