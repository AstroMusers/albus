import importlib
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest
from tdpy.verbosity import print

matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parents[1]
SHARED_NAMES = (
    'confusion_counts', 'f1_score', 'pr_from_conf', 'uniform_edges',
    'binned_metrics_uniform', 'binned_metrics_2d_uniform', 'confusion_matrix_plot',
    'basic_line_plot', 'calibration_curve_plot',
)


@pytest.fixture
def plot_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return tuple(importlib.import_module(f'datavis_scripts.{name}') for name in ('recallplots', 'recallplots_old'))


@pytest.mark.parametrize('name', SHARED_NAMES)
def test_older_workflow_reexports_the_current_numerical_helper(plot_modules, name):
    current, older = plot_modules
    assert getattr(older, name) is getattr(current, name)


@pytest.mark.parametrize(('truth', 'prediction', 'expected'), [
    ([1, 1, 0, 0], [1, 0, 1, 0], (1, 1, 1, 1)),
    ([], [], (0, 0, 0, 0)),
    ([1, 1], [1, 1], (2, 0, 0, 0)),
    ([0, 0], [1, 1], (0, 0, 2, 0)),
])
def test_confusion_counts_preserve_the_documented_order(plot_modules, truth, prediction, expected):
    current, _ = plot_modules
    assert current.confusion_counts(np.array(truth), np.array(prediction)) == expected


def test_one_and_two_dimensional_bins_keep_boundary_samples_and_metrics(plot_modules):
    current, _ = plot_modules
    truth = np.array([1, 1, 0, 0])
    prediction = np.array([1, 0, 1, 0])
    values = np.array([1.0, 2.0, 3.0, 4.0])

    table = current.binned_metrics_uniform(truth, prediction, values, 1, False)
    grid = current.binned_metrics_2d_uniform(truth, prediction, values, values, 1, False, False)

    assert table['count'].tolist() == [4]
    for lower, upper in [('tp', 'TP'), ('tn', 'TN'), ('fp', 'FP'), ('fn', 'FN')]:
        assert table[lower].tolist() == [1]
        np.testing.assert_array_equal(grid[upper], [[1]])
    for name in ('precision', 'recall', 'f1'):
        assert table[name].tolist() == [0.5]
        np.testing.assert_array_equal(grid[name], [[0.5]])
    assert current.pr_from_conf(0, 0, 0) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize('column', ['truth', 'prediction', 'x', 'y'])
@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize('log_scale', [False, True])
def test_binned_metrics_exclude_nonfinite_rows_without_misaligning_samples(plot_modules, column, value, log_scale):
    current, _ = plot_modules
    data = {
        'truth': np.array([1.0, 0.0, 1.0, 0.0]),
        'prediction': np.array([1.0, 0.0, 0.0, 1.0]),
        'x': np.array([1.0, 2.0, 3.0, 4.0]),
        'y': np.array([2.0, 4.0, 6.0, 8.0]),
    }
    data[column][0] = value
    valid = np.isfinite(data['truth']) & np.isfinite(data['prediction']) & np.isfinite(data['x'])

    actual = current.binned_metrics_uniform(data['truth'], data['prediction'], data['x'], 2, log_scale)
    expected = current.binned_metrics_uniform(data['truth'][valid], data['prediction'][valid], data['x'][valid], 2, log_scale)

    pd.testing.assert_frame_equal(actual, expected)
    assert actual['count'].sum() == (4 if column == 'y' else 3)
    valid &= np.isfinite(data['y'])
    actual_grid = current.binned_metrics_2d_uniform(data['truth'], data['prediction'], data['x'], data['y'], 2, log_scale, log_scale)
    expected_grid = current.binned_metrics_2d_uniform(*(data[name][valid] for name in ('truth', 'prediction', 'x', 'y')), 2, log_scale, log_scale)
    for name in actual_grid:
        np.testing.assert_array_equal(actual_grid[name], expected_grid[name])
    assert sum(actual_grid[name].sum() for name in ('TP', 'TN', 'FP', 'FN')) == 3


def test_all_nonfinite_coordinates_preserve_empty_metric_tables(plot_modules):
    current, _ = plot_modules
    truth = np.array([1.0, 0.0])
    values = np.array([np.inf, -np.inf])

    summary = current.binned_metrics_uniform(truth, truth, values, 2, False)
    grid = current.binned_metrics_2d_uniform(truth, truth, values, values, 2, False, False)

    assert summary['count'].tolist() == [0]
    for name in ('precision', 'recall', 'f1'):
        assert summary[name].tolist() == [0.0]
        np.testing.assert_array_equal(grid[name], [[0.0]])


def test_confusion_matrix_aligns_unequal_counts_with_true_and_predicted_labels(plot_modules, monkeypatch, tmp_path):
    current, older = plot_modules
    captured = {}
    save = current.plt.Figure.savefig

    def capture_and_save(figure, path, **kwargs):
        axis = figure.axes[0]
        captured['matrix'] = axis.images[0].get_array().copy()
        captured['horizontal'] = [label.get_text() for label in axis.get_xticklabels()]
        captured['vertical'] = [label.get_text() for label in axis.get_yticklabels()]
        captured['annotations'] = {text.get_position(): text.get_text() for text in axis.texts}
        return save(figure, path, **kwargs)

    monkeypatch.setattr(current.plt.Figure, 'savefig', capture_and_save)
    filename = tmp_path / 'unequal_confusion_counts.png'
    print(f'Writing to {filename}...')
    older.confusion_matrix_plot(tp=2, tn=3, fp=5, fn=7, out_path=filename)

    np.testing.assert_array_equal(captured['matrix'], [[2, 7], [5, 3]])
    assert captured['horizontal'] == ['Pred:1', 'Pred:0']
    assert captured['vertical'] == ['True:1', 'True:0']
    assert captured['annotations'] == {(0, 0): '2', (1, 0): '7', (0, 1): '5', (1, 1): '3'}
    print(f'Reading from {filename}...')
    image = current.plt.imread(filename)
    assert image.shape[0] > 100 and image.shape[1] > 100
    assert np.ptp(image[..., :3]) > 0


@pytest.mark.parametrize('script', ['recallplots', 'recallplots_old'])
@pytest.mark.parametrize('as_module', [False, True])
def test_both_command_line_entry_points_remain_import_safe(script, as_module, tmp_path):
    command = [sys.executable]
    if as_module:
        command += ['-m', f'datavis_scripts.{script}']
    else:
        command += [str(ROOT / 'datavis_scripts' / f'{script}.py')]
    result = subprocess.run(command + ['--help'], cwd=ROOT if as_module else tmp_path, capture_output=True, text=True, check=True)
    assert '--pos_csv' in result.stdout
    assert '--neg_csv' in result.stdout


@pytest.mark.parametrize(('truth', 'scores', 'expected_x', 'expected_y', 'probability'), [
    ([0, 1, 0, 1], [5, 5, 5, 5], [5], [0.5], False),
    ([0, 1, 0, 1], [2, 4, 6, 8], [3, 7], [0.5, 0.5], False),
    ([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], [0.15, 0.35], [0.5, 0.5], True),
    ([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], [0.15, 0.35], [0.5, 0.5], False),
    ([0, 1, 0, 1], [5, np.inf, np.nan, 5], [5], [0.5], False),
    ([0, np.inf, 1, 1], [1, 2, 3, 4], [2, 4], [0.5, 1], False),
    ([], [], [], [], False),
    ([0, 1], [np.inf, np.nan], [], [], False),
])
def test_score_plot_preserves_coordinates_and_explicit_probability_mode(plot_modules, monkeypatch, tmp_path, truth, scores, expected_x, expected_y, probability):
    current, older = plot_modules
    captured = {}
    save = current.plt.Figure.savefig

    def capture_and_save(figure, path, **kwargs):
        axis = figure.axes[0]
        captured['lines'] = [(line.get_xdata(), line.get_ydata()) for line in axis.lines]
        captured['xlabel'] = axis.get_xlabel()
        return save(figure, path, **kwargs)

    monkeypatch.setattr(current.plt.Figure, 'savefig', capture_and_save)
    filename = tmp_path / 'score_curve.png'
    print(f'Writing to {filename}...')
    older.calibration_curve_plot(np.asarray(truth), np.asarray(scores), 2, filename, score_is_probability=probability)

    assert len(captured['lines']) == (2 if probability else 1)
    np.testing.assert_allclose(captured['lines'][-1][0], expected_x)
    np.testing.assert_allclose(captured['lines'][-1][1], expected_y)
    assert captured['xlabel'] == ('Predicted probability' if probability else 'Detection score (SNR)')
    if probability:
        np.testing.assert_array_equal(captured['lines'][0], [[0, 1], [0, 1]])
    print(f'Reading from {filename}...')
    assert np.ptp(current.plt.imread(filename)[..., :3]) > 0


@pytest.mark.parametrize('n_bins', [0, -1])
def test_score_plot_rejects_nonpositive_bin_counts(plot_modules, tmp_path, n_bins):
    current, _ = plot_modules
    with pytest.raises(ValueError, match='n_bins must be positive'):
        current.calibration_curve_plot(np.array([0, 1]), np.array([2, 3]), n_bins, tmp_path / 'unused.png')


@pytest.mark.parametrize('score', [-0.1, 1.1])
def test_score_plot_rejects_scores_outside_explicit_probability_support(plot_modules, tmp_path, score):
    current, _ = plot_modules
    with pytest.raises(ValueError, match='probabilities must lie between zero and one'):
        current.calibration_curve_plot(np.array([0, 1]), np.array([score, 0.5]), 2, tmp_path / 'unused.png', score_is_probability=True)


@pytest.mark.parametrize('module_index', [0, 1])
@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('kind', ['line', 'heatmap', 'confusion', 'score', 'corner'])
def test_classifier_plots_render_requested_theme(plot_modules, monkeypatch, tmp_path, module_index, typeplotback, kind):
    import inspect
    from matplotlib.colors import to_rgba
    from PIL import Image
    from tdpy.plotting import plot_background_colors

    module = plot_modules[module_index]
    background, foreground = plot_background_colors(typeplotback)
    captured = []
    save = module.plt.Figure.savefig

    def capture_and_save(figure, path, **kwargs):
        captured.append(figure)
        assert figure.get_facecolor() == to_rgba(background)
        for axis in figure.axes:
            assert axis.get_facecolor() == to_rgba(background)
            assert axis.xaxis.label.get_color() == foreground
            assert axis.yaxis.label.get_color() == foreground
            assert all(spine.get_edgecolor() == to_rgba(foreground) for spine in axis.spines.values())
        return save(figure, path, **kwargs)

    monkeypatch.setattr(module.plt.Figure, 'savefig', capture_and_save)
    filename = tmp_path / 'theme.png'
    values = np.array([1.0, 2.0, 3.0, 4.0])
    truth = np.array([0, 1, 0, 1])
    options = {'typeplotback': typeplotback}
    if kind == 'line':
        module.basic_line_plot(values, values, 'x', 'y', 'Title', filename, **options)
    elif kind == 'heatmap':
        module.basic_heatmap(np.array([[0.5]]), [0, 1], [0, 1], 'x', 'y', 'Title', filename, **options)
    elif kind == 'confusion':
        module.confusion_matrix_plot(2, 3, 5, 7, filename, **options)
    elif kind == 'score':
        module.calibration_curve_plot(truth, values, 2, filename, **options)
    else:
        if 'cell_inches' in inspect.signature(module.corner_heatmap).parameters:
            options['cell_inches'] = 2.4
        module.corner_heatmap({'x': values, 'y': values}, truth, truth, 2, 'recall', filename, log_params=[], **options)
    assert len(captured) == 1
    print(f'Reading from {filename}...')
    with Image.open(filename) as image:
        pixels = np.asarray(image.convert('RGB'))
        expected = tuple(int(channel * 255) for channel in to_rgba(background)[:3])
        assert tuple(pixels[0, 0]) == expected
        assert np.any(pixels != expected)


@pytest.mark.parametrize('script', ['recallplots', 'recallplots_old'])
@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_classifier_cli_themes_every_generated_plot(tmp_path, script, typeplotback):
    from PIL import Image

    values = np.arange(1.0, 5.0)
    columns = {
        'r_p': values,
        'r_s': values * 1e7,
        'a': values,
        'inc': values / 10,
        'period': values,
        'duration': values / 100,
        'mean': values / 1000,
    }
    positive = tmp_path / 'positive.csv'
    negative = tmp_path / 'negative.csv'
    print(f'Writing to {positive}...')
    pd.DataFrame(columns | {'snr': [2, 4, 5, 1]}).to_csv(positive, index=False)
    print(f'Writing to {negative}...')
    pd.DataFrame(columns | {'snr': [5, 1, 2, 6]}).to_csv(negative, index=False)
    output = tmp_path / 'plots'
    environment = os.environ.copy()
    environment['TDPY_VERBOSITY'] = '1'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'datavis_scripts' / f'{script}.py'),
         '--pos_csv', str(positive), '--neg_csv', str(negative), '--out', str(output),
         '--bins', '2', '--typeplotback', typeplotback],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=True,
    )
    print(f'Reading from {output / "summary.txt"}...')
    assert 'TP=2, FP=2, FN=2, TN=2' in (output / 'summary.txt').read_text()
    images = sorted(output.glob('*.png'))
    assert len(images) == (32 if script == 'recallplots' else 52)
    for path in images:
        print(f'Reading from {path}...')
    output_lines = result.stdout.splitlines()
    assert output_lines[:2] == [f'Reading from {positive}...', f'Reading from {negative}...']
    written_paths = [line.removeprefix('Writing to ').removesuffix('...') for line in output_lines
                     if line.startswith('Writing to ')]
    assert len(written_paths) == len(list(output.iterdir()))
    assert set(written_paths) == {str(path) for path in output.iterdir()}
    expected = (0, 0, 0) if typeplotback == 'dark' else (255, 255, 255)
    for path in images:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB'))
            assert tuple(pixels[0, 0]) == expected, path.name
            assert np.any(pixels != expected), path.name
