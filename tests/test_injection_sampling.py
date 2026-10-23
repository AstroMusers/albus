import ast
import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from threading import Semaphore
from unittest.mock import Mock, call
import warnings
import subprocess
import sys
import csv
import json
import os
import copy
import inspect
from decimal import Decimal, localcontext
from math import isfinite
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pytest
from astropy import units
from astropy.time import Time
from astropy.timeseries import TimeSeries, BoxLeastSquares
from astropy.units import cds
from astropy.table import Table, MaskedColumn
from astropy.coordinates import SkyCoord
from astropy.io import fits
from scipy.signal import medfilt

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_bls_notebook_calls_pipeline_and_keeps_paths_private():
    path = ROOT / 'examples' / 'synthetic_bls_pipeline.ipynb'
    print('Reading from examples/synthetic_bls_pipeline.ipynb...')
    notebook = json.loads(path.read_text(encoding='utf-8'))
    code = '\n'.join(
        ''.join(cell['source'])
        for cell in notebook['cells']
        if cell['cell_type'] == 'code'
    )
    document = json.dumps(notebook)

    assert 'from BLSFit import BLSfit, BLSResults, FoldedLC' in code
    assert 'BLSfit(light_curve)' in code
    assert 'BLSResults(' in code
    assert 'FoldedLC(' in code
    assert 'repository_root / "examples" / "visuals"' in code
    assert 'from bokeh.layouts import column' in code
    assert 'albus_synthetic_bls_interactive.html' in code
    assert 'not a planet detection' in document
    assert '/' + 'Users/' not in document
    outputs = [output for cell in notebook['cells'] for output in cell.get('outputs', [])]
    assert sum('image/png' in output.get('data', {}) for output in outputs) == 2
    assert 'fractional_period_error' in code
    interactive_path = ROOT / 'examples' / 'visuals' / 'albus_synthetic_bls_interactive.html'
    print(f'Reading from {interactive_path.relative_to(ROOT)}...')
    interactive_html = interactive_path.read_text(encoding='utf-8')
    assert 'Synthetic BLS periodogram' in interactive_html
    assert 'Folded synthetic light curve' in interactive_html
    assert '/' + 'Users/' not in interactive_html


@pytest.fixture
def sampler(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    state = np.random.get_state()
    module = importlib.import_module('injection_sampling')
    yield module
    np.random.set_state(state)


@pytest.fixture
def selector_namespace():
    path = ROOT / 'preprocess.py'
    print(f'Reading from {path}...')
    node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'find_light_curve')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    namespace['_select_light_curve'] = namespace.pop('find_light_curve')
    return namespace


@pytest.mark.parametrize(('lower', 'upper', 'alpha', 'quantile', 'expected'), [
    (2.0, 6.0, 0.0, 0.25, 3.0),
    (1.0, 16.0, 1.0, 0.5, 4.0),
    (1.0, 4.0, 2.0, 0.5, 1.6),
    (1.0, 3.0, -1.0, 0.5, np.sqrt(5.0)),
])
def test_sampler_matches_closed_form_quantiles(sampler, monkeypatch, lower, upper, alpha, quantile, expected):
    monkeypatch.setattr(sampler.np.random, 'random', lambda size=None: quantile)
    assert sampler.sample_power_law(lower, upper, alpha) == pytest.approx(expected)


@pytest.mark.parametrize('size', [None, 0, 5, (2, 3)])
@pytest.mark.parametrize('alpha', [0.5, 1.0, 1.5])
def test_sample_shapes_and_inverse_cdf_match_the_seeded_uniform_draws(sampler, size, alpha):
    np.random.seed(35)
    samples = sampler.sample_power_law(0.5, 5.0, alpha, size=size)
    np.random.seed(35)
    uniform = np.random.random(size=size)

    assert np.shape(samples) == np.shape(uniform)
    assert np.isscalar(samples) == (size is None)
    assert np.all(np.asarray(samples) >= 0.5)
    assert np.all(np.asarray(samples) <= 5.0)
    if alpha == 1.0:
        quantiles = np.log(samples / 0.5) / np.log(10.0)
    else:
        quantiles = (samples ** (1.0 - alpha) - 0.5 ** (1.0 - alpha)) / (5.0 ** (1.0 - alpha) - 0.5 ** (1.0 - alpha))
    np.testing.assert_allclose(quantiles, uniform, rtol=0, atol=2e-15)


@pytest.mark.parametrize('alpha', [0.5, 1.0, 1.5])
def test_equal_bounds_return_the_same_value(sampler, alpha):
    np.testing.assert_array_equal(sampler.sample_power_law(2.0, 2.0, alpha, size=(2, 3)), np.full((2, 3), 2.0))


@pytest.mark.parametrize('quantile', [0.0, 0.25, 0.5, 0.75, 1.0])
def test_log_uniform_sampler_avoids_overflow_for_finite_wide_bounds(sampler, monkeypatch, quantile):
    monkeypatch.setattr(sampler.np.random, 'random', lambda size=None: quantile)
    with np.errstate(all='raise'):
        actual = sampler.sample_power_law(1e-200, 1e200, 1.0)
    expected = 10.0 ** (-200.0 + 400.0 * quantile)
    assert np.isscalar(actual)
    assert np.isfinite(actual)
    assert 1e-200 <= actual <= 1e200
    assert actual == pytest.approx(expected, rel=5e-13, abs=0)
    if quantile == 0:
        assert actual == 1e-200
    elif quantile == 1:
        assert actual == 1e200


@pytest.mark.parametrize('size', [None, 0, 7, (2, 3)])
def test_wide_log_uniform_draws_preserve_shapes_quantiles_and_rng_state(sampler, size):
    np.random.seed(62)
    expected_uniform = np.random.random(size=size)
    expected_next = np.random.random()
    np.random.seed(62)

    with np.errstate(all='raise'):
        actual = sampler.sample_power_law(1e-200, 1e200, 1.0, size=size)

    assert np.shape(actual) == np.shape(expected_uniform)
    assert np.isscalar(actual) == (size is None)
    assert np.isfinite(actual).all()
    assert np.all(np.asarray(actual) >= 1e-200)
    assert np.all(np.asarray(actual) <= 1e200)
    recovered = (np.log10(actual) + 200.0) / 400.0
    np.testing.assert_allclose(recovered, expected_uniform, rtol=0, atol=5e-16)
    assert np.random.random() == expected_next


@pytest.mark.parametrize(('lower', 'upper', 'alpha'), [(0.5, 5.0, 1100.0), (0.5, 5.0, -1100.0), (1e-200, 1e200, 3.0), (1e-200, 1e200, -2.0)])
@pytest.mark.parametrize('quantile', [0.0, 0.25, 0.5, 0.75, 1.0])
def test_extreme_bounded_power_law_quantiles_match_decimal_reference(sampler, monkeypatch, lower, upper, alpha, quantile):
    monkeypatch.setattr(sampler.np.random, 'random', lambda size=None: quantile)
    with localcontext() as context:
        context.prec = 90
        minimum, maximum = Decimal(str(lower)), Decimal(str(upper))
        exponent = Decimal(1) - Decimal(str(alpha))
        probability = Decimal(str(quantile))
        expected = float(((1 - probability) * minimum**exponent + probability * maximum**exponent) ** (1 / exponent))

    with np.errstate(all='raise'):
        actual = sampler.sample_power_law(lower, upper, alpha)

    assert np.isscalar(actual)
    assert np.isfinite(actual)
    assert lower <= actual <= upper
    assert actual == pytest.approx(expected, rel=5e-13, abs=0)
    if quantile == 0:
        assert actual == lower
    elif quantile == 1:
        assert actual == upper


@pytest.mark.parametrize('alpha', [1100.0, -1100.0])
@pytest.mark.parametrize('size', [None, 0, 7, (2, 3)])
def test_extreme_power_law_samples_preserve_shape_and_rng_draws(sampler, alpha, size):
    np.random.seed(63)
    uniform = np.random.random(size=size)
    expected_next = np.random.random()
    np.random.seed(63)
    with np.errstate(all='raise'):
        samples = sampler.sample_power_law(0.5, 5.0, alpha, size=size)
    assert np.shape(samples) == np.shape(uniform)
    assert np.isscalar(samples) == (size is None)
    assert np.isfinite(samples).all()
    assert np.all(np.asarray(samples) >= 0.5)
    assert np.all(np.asarray(samples) <= 5.0)
    assert np.random.random() == expected_next


@pytest.mark.parametrize('alpha', [np.nextafter(1.0, 0.0), np.nextafter(1.0, 2.0), 1.0 - 1e-12, 1.0 + 1e-12, 1.0 - 1e-6, 1.0 + 1e-6])
@pytest.mark.parametrize('quantile', [0.0, 0.25, 0.5, 0.75, 1.0])
def test_near_log_uniform_quantiles_avoid_cancellation(sampler, monkeypatch, alpha, quantile):
    monkeypatch.setattr(sampler.np.random, 'random', lambda size=None: quantile)
    with localcontext() as context:
        context.prec = 90
        lower, upper = Decimal('0.5'), Decimal(5)
        exponent = Decimal(1) - Decimal.from_float(float(alpha))
        probability = Decimal.from_float(quantile)
        expected = float(((1 - probability) * lower**exponent + probability * upper**exponent) ** (1 / exponent))

    with np.errstate(all='raise'):
        actual = sampler.sample_power_law(0.5, 5.0, alpha)

    assert np.isfinite(actual)
    assert 0.5 <= actual <= 5.0
    assert actual == pytest.approx(expected, rel=5e-15, abs=0)
    if quantile in (0.0, 1.0):
        assert actual == (0.5 if quantile == 0.0 else 5.0)


@pytest.mark.parametrize('alpha', [1.0 - 1e-12, 1.0 + 1e-12])
@pytest.mark.parametrize('size', [None, 0, 7, (2, 3)])
def test_near_log_uniform_sampling_preserves_shape_and_random_stream(sampler, alpha, size):
    np.random.seed(64)
    uniform = np.random.random(size=size)
    expected_next = np.random.random()
    np.random.seed(64)
    with np.errstate(all='raise'):
        samples = sampler.sample_power_law(0.5, 5.0, alpha, size=size)
    assert np.shape(samples) == np.shape(uniform)
    assert np.isscalar(samples) == (size is None)
    assert np.isfinite(samples).all()
    expected = sampler._stable_power_quantile(0.5, 5.0, 1.0 - alpha, uniform)
    np.testing.assert_array_equal(samples, expected)
    assert np.random.random() == expected_next


@pytest.mark.parametrize('lower', [1e-300, 1.0, 1e300])
@pytest.mark.parametrize('steps', [1, 16, 1024])
@pytest.mark.parametrize('alpha', [0.5, 1.0, 1.5])
def test_narrow_power_law_draws_remain_supported_and_match_decimal(sampler, monkeypatch, lower, steps, alpha):
    upper = lower
    for _ in range(steps):
        upper = np.nextafter(upper, np.inf)
    quantiles = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    monkeypatch.setattr(sampler.np.random, 'random', lambda size=None: quantiles)
    expected = []
    with localcontext() as context:
        context.prec = 90
        minimum, maximum = Decimal.from_float(lower), Decimal.from_float(float(upper))
        exponent = Decimal(1) - Decimal.from_float(alpha)
        for quantile in quantiles:
            probability = Decimal.from_float(float(quantile))
            if exponent == 0:
                value = ((1 - probability) * minimum.ln() + probability * maximum.ln()).exp()
            else:
                value = ((1 - probability) * minimum**exponent + probability * maximum**exponent) ** (1 / exponent)
            expected.append(float(value))

    with np.errstate(all='raise'):
        samples = sampler.sample_power_law(lower, upper, alpha, size=quantiles.shape)

    assert np.isfinite(samples).all()
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)
    assert samples[0] == lower
    assert samples[-1] == upper
    np.testing.assert_array_max_ulp(samples, expected, maxulp=1)


@pytest.mark.parametrize('alpha', [0.5, 1.0, 1.5])
@pytest.mark.parametrize('size', [None, 0, 7, (2, 3)])
def test_narrow_power_law_draws_preserve_shape_and_random_stream(sampler, alpha, size):
    lower = 1e300
    upper = np.nextafter(lower, np.inf)
    np.random.seed(65)
    uniform = np.random.random(size=size)
    expected_next = np.random.random()
    np.random.seed(65)

    with np.errstate(all='raise'):
        samples = sampler.sample_power_law(lower, upper, alpha, size=size)

    assert np.shape(samples) == np.shape(uniform)
    assert np.isscalar(samples) == (size is None)
    assert np.isfinite(samples).all()
    assert np.all(np.asarray(samples) >= lower)
    assert np.all(np.asarray(samples) <= upper)
    assert np.random.random() == expected_next


@pytest.mark.parametrize(('lower', 'upper', 'alpha'), [
    (0.0, 5.0, 1.5), (-1.0, 5.0, 1.5), (5.0, 1.0, 1.5),
    (1.0, np.inf, 1.5), (np.nan, 5.0, 1.5),
    (1.0, 5.0, np.nan), (1.0, 5.0, np.inf),
])
def test_invalid_power_laws_raise_before_drawing(sampler, lower, upper, alpha):
    np.random.seed(35)
    expected_next = np.random.random()
    np.random.seed(35)
    with pytest.raises(ValueError, match='bounds must be finite'):
        sampler.sample_power_law(lower, upper, alpha)
    assert np.random.random() == expected_next


@pytest.mark.parametrize('script', ['main.py', 'main_testing.py', 'main_parallel.py'])
def test_all_three_drivers_reexport_the_shared_sampler_without_loading_light_curves(sampler, script):
    path = ROOT / script
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    assert not any(isinstance(node, ast.FunctionDef) and node.name == 'sample_power_law' for node in tree.body)
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == 'injection_sampling']
    namespace = {}
    exec(compile(ast.Module(body=imports, type_ignores=[]), str(path), 'exec'), namespace)
    assert namespace['sample_power_law'] is sampler.sample_power_law


@pytest.mark.parametrize('script', ['main.py', 'main_parallel.py'])
@pytest.mark.parametrize('failures', [0, 1, 9, None])
@pytest.mark.parametrize('failure_kind', ['exception', 'missing'])
def test_target_selection_retries_failed_preprocessing_and_preserves_metadata(selector_namespace, script, failures, failure_kind):
    path = ROOT / script
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'find_light_curve')
    assert any(
        isinstance(item, ast.ImportFrom) and item.module == 'preprocess'
        and any(alias.name == 'find_light_curve' and alias.asname == '_select_light_curve' for alias in item.names)
        for item in tree.body
    )
    targets = pd.DataFrame({'MassH': [0.6, 0.8], 'E_MassH': [0.03, 0.04], 'Target ID': [123, 456]})
    marker = object()
    attempts = 10 if failures is None else failures + 1
    failure = RuntimeError('synthetic preprocessing failure') if failure_kind == 'exception' else None
    effects = [failure] * (10 if failures is None else failures)
    if failures is not None:
        effects.append(marker)
    preprocess = Mock(side_effect=effects)
    rng = SimpleNamespace(randint=Mock(side_effect=[0, 1] * 5))
    namespace = selector_namespace
    namespace.update(random=rng, df=targets, preprocess=preprocess)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)

    result = namespace['find_light_curve']() if script == 'main.py' else namespace['find_light_curve'](targets)

    assert preprocess.call_args_list == [call([123, 456][index % 2], TICID=True) for index in range(attempts)]
    assert rng.randint.call_count == attempts
    if failures is None:
        assert result is None
    else:
        selected = targets.iloc[(attempts - 1) % 2]
        assert result[0] == selected['Target ID']
        assert result[1] is marker
        assert result[2] == selected['MassH']
        assert result[3] == selected['MassH'] * 1.989e30
        assert result[4] == np.cbrt(result[3] / (1e9 * (4 / 3 * np.pi)))
        assert result[5] == result[4] * (1 / 3) * selected['E_MassH'] / selected['MassH']


def test_empty_target_catalog_does_not_draw_or_preprocess(selector_namespace):
    rng = SimpleNamespace(randint=Mock())
    preprocess = Mock()
    selector_namespace.update(random=rng, preprocess=preprocess)

    assert selector_namespace['_select_light_curve'](pd.DataFrame()) is None
    rng.randint.assert_not_called()
    preprocess.assert_not_called()


@pytest.mark.parametrize('max_tries', [0, 1, 3])
def test_parallel_selector_forwards_retry_limit_and_explicit_rng(selector_namespace, max_tries):
    path = ROOT / 'main_parallel.py'
    print(f'Reading from {path}...')
    node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'find_light_curve')
    default_rng = SimpleNamespace(randint=Mock())
    injected_rng = SimpleNamespace(randint=Mock(return_value=1))
    preprocess = Mock(side_effect=RuntimeError('synthetic failure'))
    selector_namespace.update(random=default_rng, preprocess=preprocess)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), selector_namespace)
    targets = pd.DataFrame({'MassH': [0.6, 0.8], 'E_MassH': [0.03, 0.04], 'Target ID': [123, 456]})

    assert selector_namespace['find_light_curve'](targets, max_tries=max_tries, rng=injected_rng) is None

    default_rng.randint.assert_not_called()
    assert injected_rng.randint.call_count == max_tries
    assert preprocess.call_args_list == [call(456, TICID=True)] * max_tries


@pytest.mark.parametrize(('unit', 'scale'), [
    (units.electron / units.s, 1.0),
    (units.dimensionless_unscaled, 0.01),
    (cds.ppm, 10000.0),
])
@pytest.mark.parametrize('injection', [False, True])
@pytest.mark.parametrize('missing_errors', [False, True])
def test_actual_preprocessor_uses_shared_flux_normalization_and_retains_uncertainties(unit, scale, injection, missing_errors):
    path = ROOT / 'preprocess.py'
    print(f'Reading from {path}...')
    node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'preprocess')
    time = Time('2026-10-17') + np.arange(3) * units.day
    flux = np.array([100.0, 101.0, 99.0]) * scale * unit
    error = np.full(3, np.nan) if missing_errors else np.array([2.0, 3.0, 4.0])
    flux_err = error * scale * unit
    median = np.median(flux)
    normalized = SimpleNamespace(flux=flux / median, flux_err=flux_err / median, time=time)
    normalized.remove_outliers = Mock(return_value=normalized)
    original = SimpleNamespace(flux=flux.copy(), flux_err=flux_err.copy(), time=time)
    original.remove_nans = Mock(return_value=original)
    original.flatten = Mock(return_value=original)
    original.normalize = Mock(return_value=normalized)
    download = Mock(return_value=[original])
    stitch = Mock(return_value=normalized)
    construct = Mock(side_effect=lambda time, flux, flux_err: TimeSeries(time=time, data={'flux': flux, 'flux_err': flux_err}))
    backend = SimpleNamespace(
        search_lightcurve=Mock(return_value=SimpleNamespace(download_all=download)),
        LightCurveCollection=Mock(return_value=SimpleNamespace(stitch=stitch)),
        LightCurve=construct,
    )
    namespace = {'np': np, 'lk': backend, 'medfilt': Mock(return_value=np.ones(3))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)

    result = namespace['preprocess'](123, injection=injection)

    original.normalize.assert_called_once_with()
    backend.search_lightcurve.assert_called_once_with('TIC 123', mission='TESS')
    backend.LightCurveCollection.assert_called_once_with([normalized])
    np.testing.assert_allclose(result['flux'], np.array([1.0, 1.01, 0.99]) - (1 if injection else 0))
    np.testing.assert_allclose(result['flux_err'], error / 100, equal_nan=True)
    assert result['flux_err'].unit == units.dimensionless_unscaled
    np.testing.assert_array_equal(result.time.jd, time.jd)
    np.testing.assert_array_equal(original.flux, flux)
    np.testing.assert_array_equal(original.flux_err, flux_err)


@pytest.mark.parametrize(('value', 'expected'), [
    (12.3, 12.3), ('12.3', 12.3), (0, 0.0),
    (None, None), (np.nan, None), (np.inf, None), (-np.inf, None),
    (np.ma.masked, None), ('invalid', None),
])
def test_shared_catalog_scalar_normalizer_returns_finite_values_without_mask_warnings(monkeypatch, value, expected):
    monkeypatch.syspath_prepend(str(ROOT))
    module = importlib.import_module('catalog_values')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert module.finite_or_none(value) == expected


@pytest.mark.parametrize('script', ['readfits.py', 'readfits_parallel.py'])
@pytest.mark.parametrize('magnitude', [12.3, np.inf, np.nan, None])
@pytest.mark.parametrize('response_kind', ['single', 'nearest_finite', 'no_coordinates'])
def test_actual_tic_matchers_share_finite_magnitude_handling_for_astropy_rows(monkeypatch, script, magnitude, response_kind):
    monkeypatch.syspath_prepend(str(ROOT))
    path = ROOT / script
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == 'catalog_values']
    assert len(imports) == 1
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_tic_match')
    response = Table({'ID': ['123'], 'ra': [10.0], 'dec': [-20.0]})
    response['Tmag'] = MaskedColumn([0.0 if magnitude is None else magnitude], mask=[magnitude is None])
    if response_kind == 'nearest_finite':
        response = Table({'ID': ['123', '456', '789'], 'ra': [10.0005, 10.0001, 10.0002], 'dec': [-20.0] * 3})
        response['Tmag'] = MaskedColumn([0.0 if magnitude is None else magnitude, 0.0, 14.0], mask=[magnitude is None, True, False])
    elif response_kind == 'no_coordinates':
        response.remove_columns(['ra', 'dec'])
    query = Mock(return_value=response)
    backoff = Mock()
    namespace = {
        'np': np, 'SkyCoord': SkyCoord, 'u': units, 'RADIUS_ARCSEC': 2.0,
        'RETRIES': 1, 'sem_tic': Semaphore(1), 'backoff_sleep': backoff,
        'Catalogs': SimpleNamespace(query_region=query),
        'with_retries': lambda function, *args, **kwargs: function(*args, **{key: value for key, value in kwargs.items() if key != 'retries'}),
    }
    exec(compile(ast.Module(body=imports + [definition], type_ignores=[]), str(path), 'exec'), namespace)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        target_id, actual = namespace['get_tic_match'](10.0, -20.0)

    if script == 'readfits_parallel.py' and response_kind == 'nearest_finite':
        assert target_id == '789'
        assert actual == 14.0
    else:
        assert target_id == '123'
        assert actual == (12.3 if magnitude == 12.3 else None)
    query.assert_called_once()
    backoff.assert_not_called()


def test_progress_upsert_distinguishes_added_updated_and_unchanged_entries():
    path = ROOT / 'sync_csv_to_progress.py'
    print(f'Reading from {path}...')
    node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'upsert')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    progress = {}
    upsert = namespace['upsert']

    assert upsert(progress, '10.000000,-20.000000', 'bright_lc', '123', 12.3) == 'added'
    assert progress == {'10.000000,-20.000000': {'status': 'done', 'category': 'bright_lc', 'tic_id': '123', 'Tmag': 12.3}}
    assert upsert(progress, '10.000000,-20.000000', 'bright_lc', '123', 12.3) == 'skipped'
    progress['10.000000,-20.000000']['existing_metadata'] = 'keep'
    assert upsert(progress, '10.000000,-20.000000', 'faint_lc', '123', 17.0) == 'updated'
    assert progress['10.000000,-20.000000']['Tmag'] == 17.0
    assert progress['10.000000,-20.000000']['existing_metadata'] == 'keep'


@pytest.fixture
def synchronizer(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module('sync_csv_to_progress')


@pytest.mark.parametrize('coordinate', ['inf', '-inf', 'nan', ''])
def test_progress_csv_skips_invalid_coordinates_and_counts_valid_additions(synchronizer, tmp_path, coordinate):
    filename = tmp_path / 'catalog.csv'
    print(f'Writing to {filename}...')
    with filename.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([' ra_deg ', ' dec_deg ', ' tic_id ', ' Tmag '])
        writer.writerow([coordinate, '-20', '999', '12.3'])
        writer.writerow(['10', '-20', '123', 'inf'])
    progress = {}

    counts = synchronizer.process_csv(filename, 'bright_lc', progress)

    assert counts == (1, 0, 0, 0, 1)
    assert progress == {'10.000000,-20.000000': {'status': 'done', 'category': 'bright_lc', 'tic_id': '123', 'Tmag': None}}
    assert synchronizer.process_csv(filename, 'bright_lc', progress) == (0, 0, 1, 0, 1)


def test_progress_sync_cli_runs_without_third_party_site_packages(tmp_path):
    result = subprocess.run(
        [sys.executable, '-S', str(ROOT / 'sync_csv_to_progress.py'), '--help'],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert '--progress' in result.stdout
    assert '--no_update_existing' in result.stdout
    assert 'backup of the JSON before syncing' in ' '.join(result.stdout.split())


@pytest.mark.parametrize('preserve_category', [False, True])
@pytest.mark.parametrize('migrate', [False, True])
def test_real_progress_sync_cli_honors_category_and_presync_backup_flags(tmp_path, preserve_category, migrate):
    progress_file = tmp_path / 'progress.json'
    old_key = '9.000000,-20.000000' if migrate else '10.000000,-20.000000'
    new_key = '10.000000,-20.000000'
    original = {old_key: {'status': 'done', 'category': 'bright_lc', 'tic_id': '123', 'Tmag': 12.3, 'metadata': 'keep'}}
    print(f'Writing to {progress_file}...')
    progress_file.write_text(json.dumps(original))
    csv_file = tmp_path / 'faint.csv'
    print(f'Writing to {csv_file}...')
    with csv_file.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ra_deg', 'dec_deg', 'tic_id', 'Tmag'])
        writer.writerow([10, -20, 123, 17.3])
        writer.writerow([11, -21, 456, 18.0])
    command = [
        sys.executable, '-S', str(ROOT / 'sync_csv_to_progress.py'),
        '--progress', str(progress_file), '--bright_csv', str(tmp_path / 'absent.csv'),
        '--faint_csv', str(csv_file), '--backup',
    ]
    if preserve_category:
        command.append('--no_update_existing')

    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)

    print(f'Reading from {progress_file}...')
    actual = json.loads(progress_file.read_text())
    assert actual[new_key] == {'status': 'done', 'category': 'bright_lc' if preserve_category else 'faint_lc', 'tic_id': '123', 'Tmag': 17.3, 'metadata': 'keep'}
    assert actual['11.000000,-21.000000']['category'] == 'faint_lc'
    assert actual['11.000000,-21.000000']['tic_id'] == '456'
    assert len(actual) == 2
    backups = list(tmp_path.glob('progress-before-tic-sync-*.json'))
    assert len(backups) == 1
    print(f'Reading from {backups[0]}...')
    assert json.loads(backups[0].read_text()) == original
    assert f'added=1, updated={0 if migrate else 1}' in result.stdout
    assert f'migrated={int(migrate)}' in result.stdout


@pytest.mark.parametrize('content', [
    None, '', 'ra_deg,dec_deg,tic_id\n',
    ' ra_deg , dec_deg , tic_id \n10.123456789,-20.98765, 123 \ninf,-20,456\n1,nan,nan\n11,-21,none\n',
])
def test_actual_resume_csv_reader_handles_empty_and_nonfinite_rows(monkeypatch, tmp_path, content):
    monkeypatch.syspath_prepend(str(ROOT))
    path = ROOT / 'readfits_parallel.py'
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module in {'catalog_values', 'progress_io'}]
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'load_csv_seen')
    namespace = {'os': os}
    exec(compile(ast.Module(body=imports + [definition], type_ignores=[]), str(path), 'exec'), namespace)
    filename = tmp_path / 'resume.csv'
    if content is not None:
        print(f'Writing to {filename}...')
        filename.write_text(content)

    coordinates, targets = namespace['load_csv_seen'](filename)

    if content and '10.123456789' in content:
        assert coordinates == {'10.123457,-20.987650', '11.000000,-21.000000'}
        assert targets == {'123', '456'}
    else:
        assert coordinates == set()
        assert targets == set()


@pytest.mark.parametrize('case', ['no_migration', 'free_destination', 'other_target', 'same_target', 'new_target_collision'])
def test_progress_migrations_preserve_existing_coordinate_records(synchronizer, tmp_path, capsys, case):
    old_key = '9.000000,-20.000000'
    new_key = '10.000000,-20.000000'
    progress = {old_key: {'status': 'done', 'category': 'bright_lc', 'tic_id': '123', 'metadata': 'original'}}
    if case in ('other_target', 'same_target', 'new_target_collision'):
        progress[new_key] = {'status': 'done', 'category': 'faint_lc', 'tic_id': '123' if case == 'same_target' else '456', 'metadata': 'destination'}
    if case == 'new_target_collision':
        del progress[old_key]
    original = copy.deepcopy(progress)
    filename = tmp_path / 'catalog.csv'
    print(f'Writing to {filename}...')
    with filename.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ra_deg', 'dec_deg', 'tic_id', 'Tmag'])
        writer.writerow([10, -20, 123, 12.3])
    capsys.readouterr()

    counts = synchronizer.process_csv(filename, 'bright_lc', progress, migrate_keys=case != 'no_migration')

    output = capsys.readouterr().out
    if case == 'free_destination':
        assert counts == (0, 0, 0, 1, 0)
        assert old_key not in progress
        assert progress[new_key] == {'status': 'done', 'category': 'bright_lc', 'tic_id': '123', 'metadata': 'original', 'Tmag': 12.3}
    else:
        assert counts == (0, 0, 1, 0, 0)
        assert progress == original
        assert ('is occupied; skipping TIC 123' in output) == (case != 'no_migration')


@pytest.mark.parametrize('disable_migration', [False, True])
def test_real_sync_cli_does_not_overwrite_occupied_progress_keys(tmp_path, disable_migration):
    filename = tmp_path / 'progress.json'
    original = {
        '9.000000,-20.000000': {'status': 'done', 'category': 'bright_lc', 'tic_id': '123', 'metadata': 'original'},
        '10.000000,-20.000000': {'status': 'done', 'category': 'faint_lc', 'tic_id': '456', 'metadata': 'destination'},
    }
    print(f'Writing to {filename}...')
    filename.write_text(json.dumps(original))
    catalog = tmp_path / 'catalog.csv'
    print(f'Writing to {catalog}...')
    with catalog.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ra_deg', 'dec_deg', 'tic_id', 'Tmag'])
        writer.writerow([10, -20, 123, 12.3])
    command = [sys.executable, '-S', str(ROOT / 'sync_csv_to_progress.py'), '--progress', str(filename), '--bright_csv', str(catalog), '--faint_csv', str(tmp_path / 'absent.csv')]
    if disable_migration:
        command.append('--no_migrate_keys')

    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)

    print(f'Reading from {filename}...')
    assert json.loads(filename.read_text()) == original
    assert 'added=0, updated=0, skipped=1, migrated=0' in result.stdout


@pytest.fixture
def progress_files(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module('progress_io')


@pytest.mark.parametrize('content', ['[1,2]', 'null', '{"target":null}', '{invalid'])
def test_invalid_progress_is_preserved_before_default_recovery(progress_files, tmp_path, content):
    filename = tmp_path / 'progress.json'
    print(f'Writing to {filename}...')
    filename.write_text(content)
    fallback = {'existing': {}}

    assert progress_files.load_progress_json(filename, fallback) is fallback

    backups = list(tmp_path.glob('progress.json.bad-*'))
    assert len(backups) == 1
    print(f'Reading from {backups[0]}...')
    assert backups[0].read_text() == content
    assert not filename.exists()


def test_progress_file_round_trip_preserves_records_permissions_and_narration(progress_files, tmp_path, capsys):
    filename = tmp_path / 'progress.json'
    expected = {'10.000000,-20.000000': {'status': 'done', 'tic_id': '123', 'Tmag': 12.3, 'extra': {'keep': True}}}
    print(f'Writing to {filename}...')
    filename.write_text('{}')
    filename.chmod(0o640)
    capsys.readouterr()

    progress_files.save_progress_json(filename, expected)
    actual = progress_files.load_progress_json(filename)

    assert actual == expected
    assert filename.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.iterdir()) == [filename]
    output = capsys.readouterr().out
    assert f'Writing to {filename}...' in output
    assert f'Reading from {filename}...' in output


def test_failed_progress_serialization_preserves_original_and_cleans_temporary_file(progress_files, tmp_path):
    filename = tmp_path / 'progress.json'
    original = '{"target": {"status": "done"}}'
    print(f'Writing to {filename}...')
    filename.write_text(original)

    with pytest.raises(TypeError):
        progress_files.save_progress_json(filename, {'invalid': object()})

    print(f'Reading from {filename}...')
    assert filename.read_text() == original
    assert list(tmp_path.iterdir()) == [filename]


@pytest.mark.parametrize('content', [None, '', '  \n'])
@pytest.mark.parametrize('fallback', [None, {'existing': {}}, []])
def test_progress_reader_preserves_caller_default_for_missing_and_empty_files(progress_files, tmp_path, content, fallback):
    filename = tmp_path / 'progress.json'
    if content is not None:
        print(f'Writing to {filename}...')
        filename.write_text(content)
    assert progress_files.load_progress_json(filename, fallback) is fallback


def test_progress_read_errors_do_not_quarantine_readable_content(progress_files, monkeypatch, tmp_path):
    filename = tmp_path / 'progress.json'
    original = '{"target": {"status": "done"}}'
    print(f'Writing to {filename}...')
    filename.write_text(original)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', Mock(side_effect=PermissionError('synthetic read failure')))
        with pytest.raises(PermissionError, match='synthetic read failure'):
            progress_files.load_progress_json(filename)
    print(f'Reading from {filename}...')
    assert filename.read_text() == original
    assert list(tmp_path.iterdir()) == [filename]


@pytest.mark.parametrize('operation', ['quarantine', 'replace'])
def test_failed_progress_replacement_preserves_original_and_cleans_owned_temporary_files(progress_files, monkeypatch, tmp_path, operation):
    filename = tmp_path / 'progress.json'
    original = '{invalid' if operation == 'quarantine' else '{"target": {"status": "done"}}'
    print(f'Writing to {filename}...')
    filename.write_text(original)
    monkeypatch.setattr(progress_files.os, 'replace', Mock(side_effect=PermissionError('synthetic replacement failure')))

    with pytest.raises(PermissionError, match='synthetic replacement failure'):
        if operation == 'quarantine':
            progress_files.load_progress_json(filename)
        else:
            progress_files.save_progress_json(filename, {'new': {}})

    print(f'Reading from {filename}...')
    assert filename.read_text() == original
    assert list(tmp_path.iterdir()) == [filename]


def test_repeated_invalid_progress_never_overwrites_earlier_preserved_files(progress_files, monkeypatch, tmp_path):
    filename = tmp_path / 'progress.json'
    prior = tmp_path / 'progress.json.bad-20261018-000000'
    print(f'Writing to {prior}...')
    prior.write_text('earlier preserved content')
    monkeypatch.setattr(progress_files.time, 'strftime', lambda *args: '20261018-000000')
    for content in ('{invalid', '[]'):
        print(f'Writing to {filename}...')
        filename.write_text(content)
        assert progress_files.load_progress_json(filename) == {}
    backups = list(tmp_path.glob('progress.json.bad-*'))
    assert len(backups) == 3
    contents = set()
    for backup in backups:
        print(f'Reading from {backup}...')
        contents.add(backup.read_text())
    assert contents == {'earlier preserved content', '{invalid', '[]'}


@pytest.mark.parametrize('script', ['readfits_parallel.py', 'sync_csv_to_progress.py'])
def test_progress_io_wrappers_preserve_signatures_and_real_file_round_trips(progress_files, tmp_path, script):
    path = ROOT / script
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    load_name, save_name = ('load_json', 'save_json') if script == 'readfits_parallel.py' else ('safe_load_json', 'safe_save_json')
    nodes = [node for node in tree.body if (isinstance(node, ast.ImportFrom) and node.module == 'progress_io') or (isinstance(node, ast.FunctionDef) and node.name in {load_name, save_name})]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    filename = tmp_path / 'progress.json'
    if script == 'readfits_parallel.py':
        assert namespace[load_name](filename, None) is None
    else:
        assert namespace[load_name](filename) == {}
    expected = {'target': {'tic_id': '123', 'status': 'done'}}

    namespace[save_name](filename, expected)

    actual = namespace[load_name](filename, {}) if script == 'readfits_parallel.py' else namespace[load_name](filename)
    assert actual == expected


def test_serial_catalog_helpers_import_without_data_reads_or_writes(progress_files, monkeypatch, tmp_path):
    path = ROOT / 'readfits.py'
    astroquery = ModuleType('astroquery')
    mast = ModuleType('astroquery.mast')
    mast.Catalogs = SimpleNamespace(query_region=Mock())
    mast.Observations = SimpleNamespace(query_criteria=Mock())
    astroquery.mast = mast
    monkeypatch.setitem(sys.modules, 'astroquery', astroquery)
    monkeypatch.setitem(sys.modules, 'astroquery.mast', mast)
    blocked_read = Mock(side_effect=AssertionError('Data read during helper import'))
    blocked_write = Mock(side_effect=AssertionError('Data write during helper import'))
    monkeypatch.setattr(progress_files, 'load_progress_json', blocked_read)
    monkeypatch.setattr(progress_files, 'save_progress_json', blocked_write)
    monkeypatch.setattr(fits, 'open', blocked_read)
    monkeypatch.setattr('builtins.open', blocked_write)
    monkeypatch.chdir(tmp_path)
    specification = importlib.util.spec_from_file_location('albus_serial_import_test', path)
    module = importlib.util.module_from_spec(specification)

    specification.loader.exec_module(module)

    assert callable(module.get_tic_match)
    assert callable(module.has_tess_lightcurve)
    blocked_read.assert_not_called()
    blocked_write.assert_not_called()
    mast.Catalogs.query_region.assert_not_called()
    mast.Observations.query_criteria.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('progress_state', ['missing', 'existing', 'existing_no_csv', 'invalid', 'pending', 'failed', 'missing_status', 'tic_outage', 'tess_outage', 'duplicate_rows', 'rounded_duplicates'])
@pytest.mark.parametrize('script', ['readfits.py', 'readfits_parallel.py'])
def test_actual_catalog_workflows_preserve_fits_csv_and_progress_outputs(progress_files, tmp_path, progress_state, script):
    input_file = tmp_path / 'input.fits'
    source_data = Table({'ra': [10.0, 11.0, 12.0, 13.0], 'dec': [-20.0, -21.0, -22.0, -23.0], 'Pwd': [0.99, 0.98, 0.89, 0.99], 'bright_N_flag': [0, 0, 0, 1]})
    duplicate = progress_state in ('duplicate_rows', 'rounded_duplicates')
    if duplicate:
        offset = 2e-8 if progress_state == 'rounded_duplicates' else 0.0  # [deg]
        source_data.add_row([10.0 + offset, -20.0 + offset, 0.97, 0])
    selected_first_index = len(source_data) - 1 if duplicate and script == 'readfits_parallel.py' else 0
    selected_first_ra = float(source_data['ra'][selected_first_index])  # [deg]
    selected_first_dec = float(source_data['dec'][selected_first_index])  # [deg]
    print(f'Writing to {input_file}...')
    source_data.write(input_file, format='fits')
    progress_file = tmp_path / 'progress.json'
    bright_csv = tmp_path / 'bright.csv'
    faint_csv = tmp_path / 'faint.csv'
    bright_fits = tmp_path / 'bright.fits'
    faint_fits = tmp_path / 'faint.fits'
    if progress_state in ('existing', 'existing_no_csv'):
        print(f'Writing to {progress_file}...')
        progress_file.write_text(json.dumps({'10.000000,-20.000000': {'status': 'done', 'category': 'bright_lc', 'tic_id': '123', 'Tmag': 12.3, 'metadata': 'keep'}}))
        if progress_state == 'existing':
            print(f'Writing to {bright_csv}...')
            with bright_csv.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['ra_deg', 'dec_deg', 'tic_id', 'Tmag'])
                writer.writerow([10, -20, 123, 12.3])
    elif progress_state == 'invalid':
        print(f'Writing to {progress_file}...')
        progress_file.write_text('[]')
    elif progress_state in ('pending', 'failed', 'missing_status'):
        entry = {'category': 'unresolved', 'tic_id': 'stale', 'Tmag': 99.0, 'metadata': 'keep'}
        if progress_state != 'missing_status':
            entry['status'] = progress_state
        print(f'Writing to {progress_file}...')
        progress_file.write_text(json.dumps({'10.000000,-20.000000': entry}))
    path = ROOT / script
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == 'progress_io']
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'save_progress', 'load_json', 'save_json', 'load_csv_seen', 'key_ra6dec6_from_vals'}]
    guard = next(node for node in tree.body if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__')
    query_error = importlib.import_module('catalog_values').CatalogQueryError

    def query_result(ra, dec):
        if progress_state == 'tic_outage' and ra == 10.0:
            raise query_error('synthetic exhausted TIC request')
        return ('123', 12.3) if ra < 10.5 else ('456', 17.3)

    def lightcurve_result(tic_id, **kwargs):
        if progress_state == 'tess_outage' and str(tic_id) == '123':
            raise query_error('synthetic exhausted TESS request')
        return True

    query = Mock(side_effect=query_result)
    lightcurves = Mock(side_effect=lightcurve_result)
    namespace = {
        '__name__': '__main__', 'fits': fits, 'Table': Table, 'csv': csv, 'os': os, 'isfinite': isfinite,
        'finite_or_none': importlib.import_module('catalog_values').finite_or_none,
        '_finite_or_none': importlib.import_module('catalog_values').finite_or_none,
        'CatalogQueryError': query_error,
        'ThreadPoolExecutor': ThreadPoolExecutor, 'as_completed': as_completed,
        'Counter': Counter, 'WORKERS': 2, 'SAVE_EVERY': 1,
        'INPUT_FITS': input_file, 'LOG_FILE': progress_file,
        'CSV_BRIGHT_LC': bright_csv, 'CSV_FAINT_LC': faint_csv,
        'FITS_BRIGHT_LC': bright_fits, 'FITS_FAINT_LC': faint_fits,
        'TMAG_BRIGHT_CUTOFF': 16.0, 'get_tic_match': query, 'has_tess_lightcurve': lightcurves,
        'tqdm': lambda values, **kwargs: values,
    }

    exec(compile(ast.Module(body=imports + definitions + [guard], type_ignores=[]), str(path), 'exec'), namespace)

    expected_queries = [call(11.0, -21.0)] if progress_state in ('existing', 'existing_no_csv') else [call(selected_first_ra, selected_first_dec), call(11.0, -21.0)]
    assert query.call_count == len(expected_queries)
    query.assert_has_calls(expected_queries, any_order=script == 'readfits_parallel.py')
    assert lightcurves.call_count == len(expected_queries) - int(progress_state == 'tic_outage')
    progress = progress_files.load_progress_json(progress_file)
    outage = progress_state in ('tic_outage', 'tess_outage')
    assert set(progress) == ({'11.000000,-21.000000'} if outage else {'10.000000,-20.000000', '11.000000,-21.000000'})
    if not outage:
        assert progress['10.000000,-20.000000']['category'] == 'bright_lc'
        assert progress['10.000000,-20.000000']['status'] == 'done'
        assert progress['10.000000,-20.000000']['tic_id'] == '123'
        assert progress['10.000000,-20.000000']['Tmag'] == 12.3
    assert progress['11.000000,-21.000000']['category'] == 'faint_lc'
    if progress_state in ('existing', 'existing_no_csv', 'pending', 'failed', 'missing_status'):
        assert progress['10.000000,-20.000000']['metadata'] == 'keep'
    if progress_state == 'invalid':
        backups = list(tmp_path.glob('progress.json.bad-*'))
        assert len(backups) == 1
        print(f'Reading from {backups[0]}...')
        assert backups[0].read_text() == '[]'
    for filename, target, magnitude in [(bright_csv, '123', 12.3), (faint_csv, '456', 17.3)]:
        print(f'Reading from {filename}...')
        with filename.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if outage and target == '123':
            assert rows == []
        else:
            assert len(rows) == 1
            assert rows[0]['tic_id'] == target
            assert float(rows[0]['Tmag']) == magnitude
            if target == '123':
                assert float(rows[0]['ra_deg']) == selected_first_ra
                assert float(rows[0]['dec_deg']) == selected_first_dec
    for filename, right_ascension in [(bright_fits, 10.0), (faint_fits, 11.0)]:
        if outage and right_ascension == 10.0:
            assert not filename.exists()
            continue
        print(f'Reading from {filename}...')
        restored = Table.read(filename)
        assert restored.colnames == source_data.colnames
        index = selected_first_index if right_ascension == 10.0 else 1
        selected_rows = [index]
        if duplicate and script == 'readfits.py' and right_ascension == 10.0:
            selected_rows.append(len(source_data) - 1)
        for column in source_data.colnames:
            np.testing.assert_array_equal(restored[column], source_data[column][selected_rows])


@pytest.mark.parametrize('script', ['readfits.py', 'readfits_parallel.py'])
@pytest.mark.parametrize('count', [1, 2, 4])
def test_both_catalog_exporters_write_selected_fits_records_as_rows(tmp_path, script, count):
    input_file = tmp_path / 'input.fits'
    expected = Table({'ra': np.arange(count) + 10.0, 'dec': -np.arange(count) - 20.0, 'Pwd': np.full(count, 0.99), 'bright_N_flag': np.zeros(count, dtype=np.int64)})
    print(f'Writing to {input_file}...')
    expected.write(input_file, format='fits')
    print(f'Reading from {input_file}...')
    with fits.open(input_file) as hdus:
        data = hdus[1].data.copy()
    path = ROOT / script
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    branches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Call) and isinstance(node.test.left.func, ast.Name)
        and node.test.left.func.id == 'len' and isinstance(node.test.left.args[0], ast.Name)
        and node.test.left.args[0].id in {'bright_full', 'faint_full'}
    ]
    assert len(branches) == 2
    namespace = {
        'Table': Table, 'data': data, 'names': data.names,
        'bright_full': [data[index] for index in range(count)],
        'faint_full': [data[index] for index in range(count)],
        'FITS_BRIGHT_LC': tmp_path / 'bright.fits',
        'FITS_FAINT_LC': tmp_path / 'faint.fits',
    }

    exec(compile(ast.Module(body=branches, type_ignores=[]), str(path), 'exec'), namespace)

    for name in ('FITS_BRIGHT_LC', 'FITS_FAINT_LC'):
        filename = namespace[name]
        print(f'Reading from {filename}...')
        actual = Table.read(filename)
        assert actual.colnames == expected.colnames
        for column in expected.colnames:
            np.testing.assert_array_equal(actual[column], expected[column])


@pytest.mark.parametrize('initial', [None, '', 'ra_deg,dec_deg,tic_id,Tmag\n'])
def test_shared_catalog_append_writes_one_row_and_updates_seen_after_success(progress_files, tmp_path, initial):
    filename = tmp_path / 'catalog.csv'
    if initial is not None:
        print(f'Writing to {filename}...')
        filename.write_text(initial)
    seen = set()

    assert progress_files.append_catalog_row(filename, 10.123456789, -20.0, '123', 12.3, seen)
    assert not progress_files.append_catalog_row(filename, 10.123456789, -20.0, '123', 12.3, seen)

    assert seen == {'10.123457,-20.000000'}
    print(f'Reading from {filename}...')
    with filename.open(newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames == ['ra_deg', 'dec_deg', 'tic_id', 'Tmag']
    assert rows == [{'ra_deg': '10.123456789', 'dec_deg': '-20.0', 'tic_id': '123', 'Tmag': '12.3'}]


def test_failed_catalog_append_does_not_mark_coordinate_seen(progress_files, monkeypatch, tmp_path):
    filename = tmp_path / 'catalog.csv'
    seen = set()
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', Mock(side_effect=PermissionError('synthetic CSV failure')))
        with pytest.raises(PermissionError, match='synthetic CSV failure'):
            progress_files.append_catalog_row(filename, 10.0, -20.0, '123', 12.3, seen)
    assert seen == set()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('script', ['readfits.py', 'readfits_parallel.py'])
@pytest.mark.parametrize('service', ['tic', 'tess'])
@pytest.mark.parametrize('outcome', ['empty', 'failed', 'retry_empty', 'retry_match'])
def test_actual_catalog_retry_helpers_distinguish_failures_from_empty_results(monkeypatch, script, service, outcome):
    monkeypatch.syspath_prepend(str(ROOT))
    catalog = importlib.import_module('catalog_values')
    path = ROOT / script
    print(f'Reading from {path}...')
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if (isinstance(node, ast.ImportFrom) and node.module == 'catalog_values') or (isinstance(node, ast.FunctionDef) and node.name in {'with_retries', 'get_tic_match', 'has_tess_lightcurve'})]
    empty = Table({'ID': [], 'Tmag': [], 'ra': [], 'dec': []})
    match = Table({'ID': ['123'], 'Tmag': [12.3], 'ra': [10.0], 'dec': [-20.0]})
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        if outcome == 'failed' or (outcome.startswith('retry_') and len(calls) == 1):
            raise ConnectionError('synthetic catalog outage')
        return match if outcome == 'retry_match' else empty

    query = Mock(side_effect=request)
    namespace = {
        'np': np, 'SkyCoord': SkyCoord, 'u': units, 'RADIUS_ARCSEC': 2.0, 'RETRIES': 2,
        'Catalogs': SimpleNamespace(query_region=query), 'Observations': SimpleNamespace(query_criteria=query),
        'sem_tic': Semaphore(1), 'sem_lc': Semaphore(1),
        'backoff_sleep': Mock(), 'time': SimpleNamespace(sleep=Mock()),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    function = namespace['get_tic_match'] if service == 'tic' else namespace['has_tess_lightcurve']
    arguments = {'ra_deg': 10.0, 'dec_deg': -20.0}
    if service == 'tess':
        arguments['tic_id'] = '123'

    if outcome == 'failed':
        with pytest.raises(catalog.CatalogQueryError) as caught:
            function(**arguments)
        assert isinstance(caught.value.__cause__, ConnectionError)
        assert len(calls) == (3 if script == 'readfits.py' else 2)
    else:
        result = function(**arguments)
        expected = ('123', 12.3) if service == 'tic' else True
        assert result == (expected if outcome == 'retry_match' else (None, None) if service == 'tic' else False)
        if outcome.startswith('retry_'):
            assert len(calls) >= 2


@pytest.fixture
def transit_mask():
    path = ROOT / 'BLStests.py'
    print(f'Reading from {path}...')
    node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'create_transit_mask_manual')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['create_transit_mask_manual']


@pytest.mark.parametrize('unit', [units.day, units.hour, units.minute, units.s, units.ms])
def test_transit_mask_is_invariant_to_equivalent_quantity_time_units(transit_mask, unit):
    days = np.array([0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0])
    time = (days * units.day).to(unit)
    original = time.copy()

    actual = transit_mask(time, 24 * units.hour, 0 * units.s, 12 * units.hour)

    np.testing.assert_array_equal(actual, [True, True, False, False, False, True, True])
    np.testing.assert_array_equal(actual, transit_mask(days, 1.0, 0.0, 0.5))
    np.testing.assert_array_equal(time, original)
    assert time.unit == unit


def test_transit_mask_retains_absolute_time_epochs_and_strict_boundaries(transit_mask):
    epoch = Time('2026-10-18T00:00:00', scale='utc')
    time = epoch + np.array([0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0]) * units.day

    actual = transit_mask(time, 1 * units.day, epoch, 0.5 * units.day)

    np.testing.assert_array_equal(actual, [True, True, False, False, False, True, True])


@pytest.mark.parametrize('shape', [(), (0,), (2, 3)])
def test_transit_mask_preserves_scalar_empty_and_multidimensional_shapes(transit_mask, shape):
    time = np.full(shape, 10800.0) * units.s
    actual = transit_mask(time, 1 * units.day, 0 * units.day, 0.5 * units.day)
    assert np.shape(actual) == shape
    assert np.asarray(actual).dtype == np.bool_
    assert np.asarray(actual).all()


def test_transit_mask_rejects_incompatible_time_units(transit_mask):
    with pytest.raises(units.UnitConversionError):
        transit_mask(np.array([1.0, 2.0]) * units.m, 1.0, 0.0, 0.5)


@pytest.mark.parametrize(('parameter', 'value', 'message'), [
    ('period', 0.0, 'periods must be finite and positive'),
    ('period', -1.0, 'periods must be finite and positive'),
    ('period', np.inf, 'periods must be finite and positive'),
    ('period', np.nan, 'periods must be finite and positive'),
    ('t0', np.inf, 'epochs must be finite'),
    ('t0', np.nan, 'epochs must be finite'),
    ('duration', -0.1, 'durations must be finite and nonnegative'),
    ('duration', np.inf, 'durations must be finite and nonnegative'),
    ('duration', np.nan, 'durations must be finite and nonnegative'),
])
@pytest.mark.parametrize('quantity', [False, True])
def test_transit_mask_rejects_invalid_phase_parameters(transit_mask, parameter, value, message, quantity):
    arguments = {'period': 1.0, 't0': 0.0, 'duration': 0.5}
    arguments[parameter] = value
    if quantity:
        arguments = {name: number * units.day for name, number in arguments.items()}
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match=message):
            transit_mask(np.array([0.0, 0.125, 0.5]), **arguments)


@pytest.mark.parametrize('quantity', [False, True])
def test_transit_mask_preserves_the_zero_duration_empty_mask(transit_mask, quantity):
    duration = 0.0 * units.s if quantity else 0.0
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        actual = transit_mask(np.array([0.0, 0.125, 0.5]), 1.0, 0.0, duration)
    np.testing.assert_array_equal(actual, [False, False, False])


def test_transit_mask_preserves_valid_quantity_parameter_broadcasting(transit_mask):
    time = np.array([[0.125, 0.5], [0.875, 1.0]]) * units.day
    period = np.array([1.0, 2.0]) * units.day
    duration = np.array([0.5, 0.25]) * units.day

    actual = transit_mask(time, period, np.array([0.0, 1.0]) * units.day, duration)

    np.testing.assert_array_equal(actual, [[True, False], [True, True]])


@pytest.fixture
def bls_helpers():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    path = ROOT / 'BLSFit.py'
    print(f'Reading from {path}...')
    definitions = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in {'BLSfit', 'BLSResults', 'FoldedLC', 'BLSTestOutputs'}]
    namespace = {'np': np, 'medfilt': medfilt, 'plt': plt, 'csv': csv}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, plt


@pytest.fixture
def noninjection_loop():
    path = ROOT / 'main_noninject.py'
    print(f'Reading from {path}...')
    statements = ast.parse(path.read_text()).body
    initialization = next(node for node in statements if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'completed_ids' for target in node.targets))
    loop = next(node for node in statements if isinstance(node, ast.For))
    return compile(ast.Module(body=[initialization, loop], type_ignores=[]), str(path), 'exec')


@pytest.mark.parametrize('mode', ['', 'save', 'show'])
@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_actual_bls_candidate_selection_is_independent_of_plot_mode(bls_helpers, monkeypatch, tmp_path, mode, typeplotback):
    from matplotlib.colors import to_rgba
    from tdpy.plotting import plot_background_colors

    namespace, plt = bls_helpers
    calculate = namespace['BLSResults']
    periods = np.geomspace(1.0, 10.0, 2048) * units.day
    power = np.zeros(2048)
    indices = np.array([20, 500, 1000, 1500])
    power[indices] = [10.0, 9.0, 8.0, 7.0]
    epoch = Time('2026-10-18T00:00:00')
    periodogram = SimpleNamespace(period=periods, power=power * units.dimensionless_unscaled, transit_time=epoch + np.linspace(0.0, 1.0, 2048) * units.day, duration=periods * 0.01)
    background, foreground = plot_background_colors(typeplotback)

    def check_displayed_theme():
        figure = plt.gcf()
        assert figure.get_facecolor() == to_rgba(background)
        assert figure.axes[0].xaxis.label.get_color() == foreground

    show = Mock(side_effect=check_displayed_theme)
    monkeypatch.setattr(plt, 'show', show)

    candidates, powers, best_period, transit_time, duration = calculate(periodogram, folder=str(tmp_path), ID='synthetic', plot=mode, typeplotback=typeplotback)

    np.testing.assert_array_equal(candidates, periods[indices].value)
    np.testing.assert_array_equal(powers, [10.0, 9.0, 8.0, 7.0])
    assert best_period == periodogram.period[20]
    assert transit_time == periodogram.transit_time[20]
    assert duration == periodogram.duration[20]
    assert show.call_count == int(mode == 'show')
    if mode == 'save':
        filename = tmp_path / 'synthetic_blsplot.png'
        print(f'Reading from {filename}...')
        image = plt.imread(filename)
        assert image.shape[0] > 100 and image.shape[1] > 100
        assert np.ptp(image[..., :3]) > 0
        np.testing.assert_array_equal(image[0, 0], to_rgba(background))
    else:
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('mode', ['', 'save', 'show'])
@pytest.mark.parametrize(('raw_index', 'secondary_index', 'other_indices'), [
    (1000, 20, (500, 1500)),
    (500, 1500, (20, 1000)),
    (1500, 500, (20, 1000)),
])
def test_actual_bls_keeps_distinct_strongest_detrended_peak(bls_helpers, monkeypatch, tmp_path, mode, raw_index, secondary_index, other_indices):
    namespace, plt = bls_helpers
    periods = np.geomspace(1.0, 10.0, 2048) * units.day
    power = np.zeros(2048)
    power[raw_index - 80:raw_index + 81] = 8.0
    indices = [raw_index, secondary_index, *other_indices]
    power[indices] = [10.0, 9.0, 7.0, 6.0]
    epoch = Time('2026-10-18T00:00:00')
    periodogram = SimpleNamespace(period=periods, power=power * units.dimensionless_unscaled, transit_time=epoch + np.linspace(0.0, 1.0, 2048) * units.day, duration=periods * 0.01)
    show = Mock()
    monkeypatch.setattr(plt, 'show', show)
    assert np.argmax(power) == raw_index
    assert np.argmax(power - medfilt(power, kernel_size=101)) == secondary_index

    candidates, powers, best_period, transit_time, duration = namespace['BLSResults'](periodogram, folder=str(tmp_path), ID='distinct', plot=mode)

    np.testing.assert_array_equal(candidates, periods[indices].value)
    np.testing.assert_array_equal(powers, [10.0, 9.0, 7.0, 6.0])
    assert best_period == periodogram.period[raw_index]
    assert transit_time == periodogram.transit_time[raw_index]
    assert duration == periodogram.duration[raw_index]
    assert show.call_count == int(mode == 'show')
    if mode == 'save':
        filename = tmp_path / 'distinct_blsplot.png'
        print(f'Reading from {filename}...')
        image = plt.imread(filename)
        assert image.shape[0] > 100 and image.shape[1] > 100
        assert np.ptp(image[..., :3]) > 0
    else:
        assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('absolute_time', [False, True])
@pytest.mark.parametrize('candidate_count', [1, 4])
def test_actual_bls_candidate_ephemerides_follow_selected_trial_indices(bls_helpers, absolute_time, candidate_count):
    namespace, _ = bls_helpers
    periods = (np.full(2048, 2.0) if candidate_count == 1 else np.geomspace(1.0, 10.0, 2048)) * units.day
    power = np.zeros(2048)
    indices = [20, 500, 1000, 1500]
    power[indices] = [10.0, 9.0, 8.0, 7.0]
    offsets = np.linspace(0.0, 1.0, 2048) * units.day
    epochs = Time('2026-10-18T00:00:00') + offsets if absolute_time else offsets
    durations = np.linspace(0.01, 0.2, 2048) * units.day
    result = SimpleNamespace(period=periods, power=power * units.dimensionless_unscaled, transit_time=epochs, duration=durations)

    default = namespace['BLSResults'](result)
    aligned = namespace['BLSResults'](result, candidate_ephemerides=True)

    assert len(default) == len(aligned) == 5
    assert default[3] == epochs[20]
    assert default[4] == durations[20]
    for actual, expected in zip(aligned[:3], default[:3]):
        np.testing.assert_array_equal(actual, expected)
    selected = indices[:candidate_count]
    assert aligned[3].shape == aligned[4].shape == (candidate_count,)
    np.testing.assert_array_equal(aligned[3] == epochs[selected], True)
    np.testing.assert_array_equal(aligned[4], durations[selected])
    assert isinstance(aligned[3], Time if absolute_time else units.Quantity)
    assert aligned[4].unit == units.day


@pytest.mark.parametrize(('minimum', 'maximum'), [(1.0, 10.0), (2.0, 3.0), (2.0, 2.0), (15.0, 20.0)])
def test_actual_bls_plot_skips_empty_summary_bins_without_changing_results(bls_helpers, monkeypatch, tmp_path, minimum, maximum):
    namespace, plt = bls_helpers
    calculate = namespace['BLSResults']
    periods = np.linspace(minimum, maximum, 101) * units.day
    power = np.ones(101)
    power[50] = 10.0
    result = SimpleNamespace(period=periods, power=power * units.dimensionless_unscaled, transit_time=np.zeros(101) * units.day, duration=periods * 0.01)
    captured = {}
    original_plot = plt.semilogx

    def capture_plot(horizontal, vertical, *args, **kwargs):
        if kwargs.get('label') == 'Median':
            captured['centers'] = np.asarray(horizontal)
            captured['medians'] = np.asarray(vertical)
        return original_plot(horizontal, vertical, *args, **kwargs)

    monkeypatch.setattr(plt, 'semilogx', capture_plot)
    expected = calculate(result, plot='')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual = calculate(result, folder=str(tmp_path), ID='sparse', plot='save')

    for returned, baseline in zip(actual, expected):
        np.testing.assert_array_equal(returned, baseline)
    edges = np.logspace(0, 1, 49)
    occupied = [(periods.value >= edges[index]) & (periods.value < edges[index + 1]) for index in range(48)]
    expected_centers = [np.median(periods.value[mask]) for mask in occupied if mask.any()]
    np.testing.assert_array_equal(captured['centers'], expected_centers)
    assert np.isfinite(captured['medians']).all()
    filename = tmp_path / 'sparse_blsplot.png'
    print(f'Reading from {filename}...')
    image = plt.imread(filename)
    assert image.shape[0] > 100 and image.shape[1] > 100
    assert np.ptp(image[..., :3]) > 0


@pytest.mark.parametrize('mode', ['', 'save', 'show'])
@pytest.mark.parametrize('binned', [False, True])
@pytest.mark.parametrize('output', [False, True])
@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_actual_folded_curve_honors_plotting_and_preserves_return_contract(bls_helpers, monkeypatch, tmp_path, mode, binned, output, typeplotback):
    """Check actual control flow with explicit Lightkurve interfaces and real plots."""
    from matplotlib.colors import to_rgba
    from tdpy.plotting import plot_background_colors

    namespace, plt = bls_helpers
    phase = np.linspace(-1.0, 1.0, 25)
    flux = np.ones(25)
    flux[10:15] = 0.98

    def draw(**kwargs):
        return plt.plot(phase, flux, **kwargs)

    binned_curve = SimpleNamespace(plot=Mock(side_effect=draw))
    folded_curve = SimpleNamespace(bin=Mock(return_value=binned_curve), scatter=Mock(side_effect=draw))
    light_curve = SimpleNamespace(fold=Mock(return_value=folded_curve))
    period = 2.0 * units.day
    epoch = Time('2026-10-18T00:00:00')
    step = 0.001 * units.day
    background, foreground = plot_background_colors(typeplotback)

    def check_displayed_theme():
        assert plt.gcf().get_facecolor() == to_rgba(background)
        assert plt.gca().xaxis.label.get_color() == foreground

    show = Mock(side_effect=check_displayed_theme)
    monkeypatch.setattr(plt, 'show', show)

    result = namespace['FoldedLC'](light_curve, period, epoch, plot=mode, ID='synthetic', folder=str(tmp_path), bin=binned, time_bin_size=step, output=output, typeplotback=typeplotback)

    light_curve.fold.assert_called_once_with(period=period, epoch_time=epoch)
    if binned:
        folded_curve.bin.assert_called_once_with(time_bin_size=step)
        assert result is binned_curve
    else:
        folded_curve.bin.assert_not_called()
        assert result is (folded_curve if output else None)
    assert binned_curve.plot.call_count == int(binned and mode != '')
    assert folded_curve.scatter.call_count == int(not binned and mode != '')
    assert show.call_count == int(mode == 'show')
    if mode == 'save':
        filename = tmp_path / 'synthetic_2.0_foldedlc.png'
        print(f'Reading from {filename}...')
        image = plt.imread(filename)
        assert image.shape[0] > 100 and image.shape[1] > 100
        assert np.ptp(image[..., :3]) > 0
        np.testing.assert_array_equal(image[0, 0], to_rgba(background))
    else:
        assert list(tmp_path.iterdir()) == []


def test_binned_fold_without_plotting_preserves_existing_figures(bls_helpers):
    namespace, plt = bls_helpers
    binned_curve = object()
    folded_curve = SimpleNamespace(bin=Mock(return_value=binned_curve))
    light_curve = SimpleNamespace(fold=Mock(return_value=folded_curve))
    figure = plt.figure()
    try:
        assert namespace['FoldedLC'](light_curve, 2.0, 0.0, bin=True) is binned_curve
        assert plt.fignum_exists(figure.number)
    finally:
        plt.close(figure)


@pytest.fixture
def bls_parameter_extractors():
    path = ROOT / 'BLStests.py'
    print(f'Reading from {path}...')
    definitions = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in {'test_period', 'test_duration'}]
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize(('name', 'attribute'), [('test_period', 'period_at_max_power'), ('test_duration', 'duration_at_max_power')])
@pytest.mark.parametrize('value', [0.5, 2.0 * units.day, 12.0 * units.hour])
def test_bls_parameter_extractors_preserve_values_and_units(bls_parameter_extractors, name, attribute, value):
    periodogram = SimpleNamespace(**{attribute: value})
    assert bls_parameter_extractors[name](periodogram) is value


def test_bls_parameter_extraction_supports_real_astropy_bls_results(bls_parameter_extractors, transit_mask):
    time = np.arange(0.0, 12.0, 0.02) * units.day
    true_period = 2.0 * units.day
    duration = 0.2 * units.day
    mask = transit_mask(time, true_period, 0.0 * units.day, duration)
    flux = np.ones(time.size)
    flux[mask] -= 0.01
    model = BoxLeastSquares(time, flux)
    result = model.power(np.linspace(1.9, 2.1, 101) * units.day, duration)
    index = np.argmax(result.power)
    periodogram = SimpleNamespace(period_at_max_power=result.period[index], duration_at_max_power=result.duration[index])

    extracted_period = bls_parameter_extractors['test_period'](periodogram)
    extracted_duration = bls_parameter_extractors['test_duration'](periodogram)

    assert extracted_period is periodogram.period_at_max_power
    assert extracted_duration is periodogram.duration_at_max_power
    assert abs((extracted_period - true_period).to_value(units.day)) < 0.01
    actual_mask = transit_mask(time, extracted_period, result.transit_time[index], extracted_duration)
    assert actual_mask.dtype == np.bool_
    assert actual_mask.any()


@pytest.mark.parametrize('mode', ['', 'save'])
@pytest.mark.parametrize('case', ['empty', 'nan_power', 'inf_power', 'nan_period', 'zero_period', 'negative_period', 'mismatched', 'matrix'])
def test_bls_results_reject_invalid_trial_arrays_before_filtering_or_plotting(bls_helpers, tmp_path, mode, case):
    namespace, _ = bls_helpers
    periods = np.geomspace(1.0, 10.0, 101)
    powers = np.ones(101)
    if case == 'empty':
        periods = np.array([])
        powers = np.array([])
    elif case == 'nan_power':
        powers[20] = np.nan
    elif case == 'inf_power':
        powers[20] = np.inf
    elif case == 'nan_period':
        periods[20] = np.nan
    elif case == 'zero_period':
        periods[20] = 0.0
    elif case == 'negative_period':
        periods[20] = -1.0
    elif case == 'mismatched':
        powers = powers[:-1]
    else:
        periods = periods.reshape(1, -1)
        powers = powers.reshape(1, -1)
    result = SimpleNamespace(period=periods * units.day, power=powers * units.dimensionless_unscaled, transit_time=np.zeros(101) * units.day, duration=np.ones(101) * units.day)
    filtered = Mock(side_effect=AssertionError('Invalid periodogram reached smoothing'))
    namespace['medfilt'] = filtered

    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match='BLS periods'):
            namespace['BLSResults'](result, folder=str(tmp_path), ID='invalid', plot=mode)

    filtered.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('script', ['main.py', 'main_testing.py', 'main_parallel.py'])
@pytest.mark.parametrize('absolute_time', [False, True])
@pytest.mark.parametrize('candidate_count', [1, 4])
def test_actual_fit_drivers_use_each_candidates_own_ephemeris(bls_helpers, tmp_path, script, absolute_time, candidate_count):
    helpers, plt = bls_helpers
    path = ROOT / script
    print(f'Reading from {path}...')
    node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'fit_fold_and_test')
    periods = (np.full(2048, 2.0) if candidate_count == 1 else np.geomspace(1.0, 10.0, 2048)) * units.day
    indices = [20, 500, 1000, 1500]
    power = np.zeros(2048)
    power[indices] = [10.0, 9.0, 8.0, 7.0]
    offsets = np.linspace(0.0, 1.0, 2048) * units.day
    epochs = Time('2026-10-18T00:00:00') + offsets if absolute_time else offsets
    durations = np.full(2048, 0.01) * units.day
    durations[indices] = [0.1, 0.2, 0.4, 0.8] * units.day
    periodogram = SimpleNamespace(period=periods, power=power * units.dimensionless_unscaled, transit_time=epochs, duration=durations)
    phases = np.array([-0.6, -0.45, -0.2, -0.1, -0.04, 0.0, 0.04, 0.1, 0.2, 0.45, 0.6]) * units.day
    folded = Table({'time': phases, 'flux': np.full(len(phases), 0.9) * units.dimensionless_unscaled})
    fold = Mock(return_value=folded)
    depth = Mock(return_value=(0.1, 0.1, 0.1))
    vshape = Mock(return_value=0.0)
    snr = Mock(return_value=10.0)
    plotter = Mock()
    figure = plt.figure()
    plotter.gcf.return_value = figure
    helpers['plt'] = plotter
    selector = Mock(wraps=helpers['BLSResults'])
    namespace = {
        'os': os, 'np': np, 'plt': plotter, 'csv': csv, 'print': Mock(),
        'BLSfit': Mock(return_value=periodogram), 'BLSResults': selector,
        'FoldedLC': fold, 'test_out_of_transit_variability': Mock(return_value=0.01),
        'test_depth': depth, 'test_v_shape': vshape, 'test_snr': snr,
        '_to_float': lambda value: float(getattr(value, 'value', value)),
        'ID': 'synthetic', 'tic_id': 123, 'r_s': 1.0, 'e_r_s': 0.1,
        'r_p': 1.0, 'a': 1.0, 'P_days': 1.0, 'inc': 90.0,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    function = namespace['fit_fold_and_test']
    arguments = {name: namespace.get(name) for name, parameter in inspect.signature(function).parameters.items() if parameter.default is inspect.Parameter.empty}
    output_path = tmp_path / 'candidate_rows.csv'
    if 'output_file' in arguments:
        arguments['output_file'] = str(output_path)
    arguments.update(lc=object(), folder=str(tmp_path))

    try:
        rows = function(**arguments)
    finally:
        plt.close(figure)

    assert selector.call_args.kwargs['candidate_ephemerides'] is True
    assert fold.call_count == depth.call_count == vshape.call_count == snr.call_count == candidate_count
    for position, index in enumerate(indices[:candidate_count]):
        assert fold.call_args_list[position].args[1] == periods[index].value
        assert fold.call_args_list[position].args[2] == epochs[index]
        expected_mask = np.abs(phases.value) < 0.6 * durations[index].value
        np.testing.assert_array_equal(depth.call_args_list[position].args[2], expected_mask)
        np.testing.assert_array_equal(vshape.call_args_list[position].args[2], expected_mask)
        np.testing.assert_array_equal(snr.call_args_list[position].args[1], expected_mask)
    if script == 'main.py':
        print(f'Reading from {output_path}...')
        with output_path.open(newline='') as stream:
            rows = list(csv.reader(stream))
    if script != 'main_testing.py':
        assert len(rows) == candidate_count
        for row, index in zip(rows, indices[:candidate_count]):
            assert float(row[8]) == periods[index].value
            recorded_duration = units.Quantity(row[9]).to_value(units.day) if script == 'main.py' else row[9]  # [day]
            assert recorded_duration == durations[index].value


@pytest.mark.parametrize('absolute_time', [False, True])
@pytest.mark.parametrize('candidate_count', [1, 4])
@pytest.mark.parametrize('repeat_target', [False, True])
def test_actual_noninjection_loop_uses_candidate_ephemerides_and_one_shared_mask(bls_helpers, noninjection_loop, transit_mask, tmp_path, absolute_time, candidate_count, repeat_target):
    helpers, _ = bls_helpers
    periods = (np.full(2048, 2.0) if candidate_count == 1 else np.geomspace(1.0, 10.0, 2048)) * units.day
    indices = [20, 500, 1000, 1500]
    power = np.zeros(2048)
    power[indices] = [10.0, 9.0, 8.0, 7.0]
    base_epoch = Time('2026-10-18T00:00:00')
    offsets = np.linspace(0.0, 1.0, 2048) * units.day
    epochs = base_epoch + offsets if absolute_time else offsets
    durations = np.full(2048, 0.01) * units.day
    durations[indices] = [0.1, 0.2, 0.4, 0.8] * units.day
    periodogram = SimpleNamespace(period=periods, power=power * units.dimensionless_unscaled, transit_time=epochs, duration=durations)
    observation_offsets = np.linspace(0.0, 20.0, 401) * units.day
    curve = {'time': base_epoch + observation_offsets if absolute_time else observation_offsets, 'flux': np.full(401, 0.9) * units.dimensionless_unscaled}
    selector = Mock(wraps=helpers['BLSResults'])
    mask = Mock(wraps=transit_mask)
    fold = Mock()
    depth = Mock(return_value=(0.1, 0.1, 0.1))
    vshape = Mock(return_value=0.0)
    snr = Mock(return_value=10.0)
    scatter = Mock(return_value=0.01)
    output_path = tmp_path / 'noninjection_candidates.csv'
    namespace = {
        'df': pd.DataFrame({'Target ID': [0, 100, 200]}),
        'out': pd.DataFrame({'ID': [200]}),
        'random': SimpleNamespace(randrange=Mock(side_effect=[1, 1, 2, 1, 2] if repeat_target else [1, 2, 2, 2, 2])),
        'tqdm': lambda values: values, 'print': Mock(),
        'preprocess': Mock(return_value=curve), 'BLSfit': Mock(return_value=periodogram),
        'BLSResults': selector, 'FoldedLC': fold, 'create_transit_mask_manual': mask,
        'test_depth': depth, 'test_v_shape': vshape, 'test_snr': snr,
        'test_out_of_transit_variability': scatter, 'BLSTestOutputs': helpers['BLSTestOutputs'],
        'output_file': str(output_path),
    }

    print(f'Writing to {output_path}...')
    exec(noninjection_loop, namespace)

    assert selector.call_args.kwargs['candidate_ephemerides'] is True
    namespace['preprocess'].assert_called_once_with(100, TICID=True)
    assert namespace['completed_ids'] == {100, 200}
    assert namespace['print'].call_args_list.count(call('outputted')) == 1
    assert fold.call_count == mask.call_count == depth.call_count == vshape.call_count == snr.call_count == scatter.call_count == candidate_count
    print(f'Reading from {output_path}...')
    with output_path.open(newline='') as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == candidate_count
    for position, index in enumerate(indices[:candidate_count]):
        assert fold.call_args_list[position].args[1] == periods[index].value
        assert fold.call_args_list[position].args[2] == epochs[index]
        assert mask.call_args_list[position].args[2] == epochs[index]
        assert mask.call_args_list[position].args[3] == durations[index]
        expected = transit_mask(curve['time'], periods[index], epochs[index], durations[index])
        actual = depth.call_args_list[position].args[2]
        np.testing.assert_array_equal(actual, expected)
        assert vshape.call_args_list[position].args[2] is actual
        assert snr.call_args_list[position].args[1] is actual
        assert scatter.call_args_list[position].args[1] is actual
        assert len(rows[position]) == 8
        assert rows[position][:2] == ['100', '100']
        assert float(rows[position][2]) == periods[index].value
        assert units.Quantity(rows[position][3]) == durations[index]


@pytest.mark.parametrize('row_count', [1, 2, 1291, 1292, 1500])
@pytest.mark.parametrize('custom_index', [False, True])
@pytest.mark.parametrize('endpoint', ['first', 'last'])
def test_actual_noninjection_sampling_uses_catalog_positions(noninjection_loop, row_count, custom_index, endpoint):
    ids = np.arange(10000, 10000 + row_count)
    frame = pd.DataFrame({'Target ID': ids}, index=np.arange(row_count) * 2 + 11 if custom_index else None)
    position = 0 if endpoint == 'first' else row_count - 1
    draw = Mock(return_value=position)
    preprocess = Mock(return_value=None)
    namespace = {
        'df': frame, 'out': pd.DataFrame({'ID': []}),
        'random': SimpleNamespace(randrange=draw), 'preprocess': preprocess,
        'tqdm': lambda values: values, 'print': Mock(),
    }

    exec(noninjection_loop, namespace)

    assert draw.call_args_list == [call(row_count)] * 5
    assert preprocess.call_args_list == [call(int(ids[position]), TICID=True)] * 5
    assert namespace['completed_ids'] == set()
    assert namespace['print'].call_args_list == [call(int(ids[position]))] * 5


def test_actual_noninjection_sampling_skips_empty_catalog_without_drawing(noninjection_loop):
    draw = Mock(side_effect=AssertionError('Empty catalog must not draw a position'))
    preprocess = Mock(side_effect=AssertionError('Empty catalog must not preprocess'))
    namespace = {
        'df': pd.DataFrame({'Target ID': []}), 'out': pd.DataFrame({'ID': []}),
        'random': SimpleNamespace(randrange=draw), 'preprocess': preprocess,
        'tqdm': lambda values: values, 'print': Mock(),
    }

    exec(noninjection_loop, namespace)

    draw.assert_not_called()
    preprocess.assert_not_called()


@pytest.mark.parametrize('completed_writes', [0, 1])
def test_actual_noninjection_loop_does_not_mark_completed_after_write_error(noninjection_loop, completed_writes):
    original = OSError('synthetic candidate write failure')
    writer = Mock(side_effect=[None] * completed_writes + [original])
    preprocess = Mock(return_value={'time': np.array([0.0, 0.5]) * units.day, 'flux': np.array([0.9, 1.0]) * units.dimensionless_unscaled})
    namespace = {
        'df': pd.DataFrame({'Target ID': [100]}), 'out': pd.DataFrame({'ID': [200]}),
        'random': SimpleNamespace(randrange=Mock(return_value=0)), 'tqdm': lambda values: values,
        'print': Mock(), 'preprocess': preprocess, 'BLSfit': Mock(),
        'BLSResults': Mock(return_value=([1.0, 2.0], [5.0, 4.0], 1.0 * units.day, np.array([0.0, 0.5]) * units.day, np.array([0.1, 0.2]) * units.day)),
        'FoldedLC': Mock(), 'create_transit_mask_manual': Mock(return_value=np.array([True, False])),
        'test_depth': Mock(return_value=(0.1, 0.1, 0.1)), 'test_v_shape': Mock(return_value=0.0),
        'test_snr': Mock(return_value=10.0), 'test_out_of_transit_variability': Mock(return_value=0.01),
        'BLSTestOutputs': writer, 'output_file': 'unused',
    }

    with pytest.raises(OSError) as caught:
        exec(noninjection_loop, namespace)

    assert caught.value is original
    assert namespace['completed_ids'] == {200}
    assert writer.call_count == completed_writes + 1
    assert call('outputted') not in namespace['print'].call_args_list


def test_actual_bls_fit_preserves_search_grid_and_successful_result(bls_helpers):
    namespace, _ = bls_helpers
    periodogram = object()
    curve = SimpleNamespace(to_periodogram=Mock(return_value=periodogram))

    assert namespace['BLSfit'](curve) is periodogram

    curve.to_periodogram.assert_called_once()
    assert curve.to_periodogram.call_args.args == ()
    arguments = curve.to_periodogram.call_args.kwargs
    assert set(arguments) == {'method', 'period', 'duration', 'frequency_factor'}
    assert arguments['method'] == 'bls'
    assert arguments['frequency_factor'] == 5000
    expected = np.logspace(0.0, 1.0, 2048)
    np.testing.assert_array_equal(arguments['period'], expected)
    np.testing.assert_array_equal(arguments['duration'], expected * 0.01)


@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, OSError])
def test_actual_bls_fit_propagates_original_error_instead_of_returning_text(bls_helpers, error_type):
    namespace, _ = bls_helpers
    original = error_type('synthetic BLS fitting failure')
    curve = SimpleNamespace(to_periodogram=Mock(side_effect=original))

    with pytest.raises(error_type) as caught:
        namespace['BLSfit'](curve)

    assert caught.value is original
    curve.to_periodogram.assert_called_once()


@pytest.mark.parametrize('script', ['main.py', 'main_testing.py', 'main_parallel.py'])
def test_actual_fit_drivers_stop_before_result_processing_after_fit_failure(bls_helpers, tmp_path, script):
    helpers, _ = bls_helpers
    path = ROOT / script
    print(f'Reading from {path}...')
    node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'fit_fold_and_test')
    result_reader = Mock(side_effect=AssertionError('Failed fit reached result processing'))
    namespace = {'os': os, 'BLSfit': helpers['BLSfit'], 'BLSResults': result_reader, 'ID': 'synthetic', 'print': Mock()}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    original = ValueError('synthetic native BLS failure')
    curve = SimpleNamespace(to_periodogram=Mock(side_effect=original))
    function = namespace['fit_fold_and_test']
    arguments = {name: object() for name, parameter in inspect.signature(function).parameters.items() if parameter.default is inspect.Parameter.empty}
    arguments.update(lc=curve, folder=str(tmp_path))

    with pytest.raises(ValueError) as caught:
        function(**arguments)

    assert caught.value is original
    curve.to_periodogram.assert_called_once()
    result_reader.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def transit_measurements():
    path = ROOT / 'BLStests.py'
    print(f'Reading from {path}...')
    names = {'test_depth', 'test_snr', 'test_out_of_transit_variability'}
    definitions = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'np': np}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('case', ['empty', 'no_transit', 'no_baseline'])
@pytest.mark.parametrize('quantity', [False, True])
def test_undefined_transit_measurements_return_nan_without_warnings(transit_measurements, case, quantity):
    flux = np.array([]) if case == 'empty' else np.array([1.0, 1.01, 0.99])
    if quantity:
        flux = flux * units.dimensionless_unscaled
    mask = np.full(flux.size, case == 'no_baseline', dtype=bool)
    with warnings.catch_warnings(), np.errstate(all='raise'):
        warnings.simplefilter('error')
        depth = transit_measurements['test_depth'](np.arange(flux.size), flux, mask)
        snr = transit_measurements['test_snr'](flux, mask)
        scatter = transit_measurements['test_out_of_transit_variability'](flux, mask)

    assert np.isnan(snr)
    if case in ('empty', 'no_transit'):
        assert np.isnan(depth).all()
    else:
        np.testing.assert_allclose(depth, [0.0, 0.0, 0.01], atol=1e-15)
    if case in ('empty', 'no_baseline'):
        assert np.isnan(scatter)
    else:
        assert scatter == np.std(flux)


@pytest.mark.parametrize('quantity', [False, True])
def test_defined_transit_measurements_preserve_depth_scatter_and_snr(transit_measurements, quantity):
    flux = np.array([1.01, 0.99, 0.97, 0.96])
    if quantity:
        flux = flux * units.dimensionless_unscaled
    original = flux.copy()
    mask = np.array([False, False, True, True])
    expected_depth = tuple(1 - function(flux[mask]) for function in (np.median, np.mean, np.min))
    expected_scatter = np.std(flux[~mask])

    actual_depth = transit_measurements['test_depth'](np.arange(4), flux, mask)
    actual_scatter = transit_measurements['test_out_of_transit_variability'](flux, mask)
    actual_snr = transit_measurements['test_snr'](flux, mask)

    np.testing.assert_array_equal(actual_depth, expected_depth)
    assert actual_scatter == expected_scatter
    assert actual_snr == expected_depth[1] / expected_scatter
    np.testing.assert_allclose(actual_depth, [0.035, 0.035, 0.04])
    assert float(actual_snr) == pytest.approx(3.5)
    np.testing.assert_array_equal(flux, original)
    if quantity:
        assert all(value.unit == units.dimensionless_unscaled for value in actual_depth)
        assert actual_scatter.unit == units.dimensionless_unscaled
        assert actual_snr.unit == units.dimensionless_unscaled
