import os
import subprocess
import sys
from pathlib import Path

import pytest
from tdpy.verbosity import print


@pytest.mark.parametrize("verbosity", ["0", "1"])
def test_datasearch_narrates_input_and_result(tmp_path, capsys, verbosity):
    data_directory = tmp_path / "data_outputs"
    data_directory.mkdir()
    input_path = data_directory / "noninjected_transits_output6.csv"
    print(f"Writing to {input_path}...")
    input_path.write_text("name,snr\nlow,1.0\nhigh,5.0\n", encoding="utf-8")
    capsys.readouterr()
    repository_path = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    python_paths = [str(repository_path)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "TDPY_VERBOSITY": verbosity,
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", "import datasearch"],
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
            "Reading from data_outputs/noninjected_transits_output6.csv...",
            "{'name': 'high', 'snr': '5.0'}",
        ]