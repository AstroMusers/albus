"""Behavioral comparisons for the analysis routines shared with Miletos."""

import sys
from pathlib import Path
import importlib
import pickle
from unittest.mock import Mock

from miletos.light_curve import LightCurve
import numpy as np
import pytest
from tdpy.verbosity import print

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BLSFit import BLSfit


@pytest.mark.parametrize('legacy_module,legacy_name,shared_module,shared_name', [
    ('BLSFit', 'BLSResults', 'box_least_squares', 'lightkurve_bls_results'),
    ('BLSFit', 'FoldedLC', 'box_least_squares', 'fold_lightkurve'),
    ('preprocess', 'preprocess', 'photometry', 'preprocess_tess_lightkurve'),
    ('BLStests', 'test_period', 'transit_metrics', 'period_at_max_power'),
    ('BLStests', 'test_duration', 'transit_metrics', 'duration_at_max_power'),
    ('BLStests', 'test_depth', 'transit_metrics', 'transit_depth'),
    ('BLStests', 'test_v_shape', 'transit_metrics', 'transit_v_shape'),
    ('BLStests', 'test_snr', 'transit_metrics', 'transit_snr'),
    ('BLStests', 'test_out_of_transit_variability', 'transit_metrics', 'out_of_transit_variability'),
    ('BLStests', 'transit_depth_quantile_phase', 'transit_metrics', 'transit_depth_quantile_phase'),
    ('BLStests', '_sigma_clip', 'transit_metrics', '_sigma_clip'),
    ('BLStests', '_wrap_centered_phase', 'transit_metrics', '_wrap_centered_phase'),
    ('BLStests', 'create_transit_mask_manual', 'transit_search', 'transit_mask'),
    ('injections', 'calc_a', 'simulation', 'semimajor_axis_from_period'),
    ('injections', 'minimum_transit_inclination', 'simulation', 'minimum_transit_inclination'),
    ('injections', 'generate_lightcurve', 'simulation', 'generate_batman_light_curve'),
    ('injections', 'inject_transit', 'simulation', 'inject_batman_transit'),
    ('injection_sampling', 'sample_power_law', 'simulation', 'sample_power_law'),
    ('injection_sampling', '_log_bound_ratio', 'simulation', '_log_bound_ratio'),
    ('injection_sampling', '_stable_power_quantile', 'simulation', '_stable_power_quantile'),
])
def test_legacy_analysis_names_are_shared_imports_and_remain_picklable(
    legacy_module, legacy_name, shared_module, shared_name,
):
    legacy = getattr(importlib.import_module(legacy_module), legacy_name)
    shared = getattr(importlib.import_module(f'miletos.{shared_module}'), shared_name)
    assert legacy is shared
    assert pickle.loads(pickle.dumps(legacy)) is shared


def test_blsfit_keeps_legacy_keyword_and_uses_native_miletos_backend(monkeypatch):
    shared = importlib.import_module('miletos.box_least_squares')
    result, curve = object(), object()
    fit = Mock(return_value=result)
    monkeypatch.setattr(shared, 'fixed_grid_box_least_squares', fit)
    assert BLSfit(flatlc=curve) is result
    fit.assert_called_once_with(curve)


def test_dataframe_transit_injection_preserves_legacy_flux_model():
    import setuptools
    import pandas as pd

    from injections import inject_transit_df, generate_lightcurve

    time = np.linspace(0.0, 4.0, 200)  # [day]
    baseline = np.full(time.size, 2.0)
    frame = pd.DataFrame({'Flux': baseline.copy()})
    _, transit_flux, _ = generate_lightcurve(
        radius_star=0.01,
        mass_star=0.6,
        radius_planet=0.005,
        albedo_planet=0.1,
        period=2.0,
        inclination=90.0,
        time_array=time,
    )

    result = inject_transit_df(
        frame,
        radius_star=0.01,
        mass_star=0.6,
        radius_planet=0.005,
        luminosity_star=0.001,
        albedo_planet=0.1,
        period=2.0,
        inclination=90.0,
        time_array=time,
    )

    assert result is frame
    np.testing.assert_array_equal(result['Flux'], baseline + transit_flux)


@pytest.mark.parametrize('tic_id', [False, True])
@pytest.mark.parametrize('outcome', ['empty', 'failure', 'success'])
def test_shared_download_wrapper_retains_targets_options_and_failure_behavior(monkeypatch, tic_id, outcome):
    from preprocess import preprocess
    import miletos.photometry as shared

    failure = shared.TessDownloadError('download failure')
    collection, expected = object(), object()
    download = Mock(return_value=None if outcome == 'empty' else collection)
    if outcome == 'failure':
        download.side_effect = failure
    prepare = Mock(return_value=expected)
    monkeypatch.setattr(shared, 'download_tess_light_curves', download)
    monkeypatch.setattr(shared, 'prepare_light_curve_collection', prepare)
    actual = preprocess('123', TICID=tic_id, injection=True, plots=True, report=True)
    download.assert_called_once_with('123', tic_id=tic_id)
    if outcome == 'success':
        assert actual is expected
        prepare.assert_called_once_with(collection, target_label='123', injection=True, plots=True, report=True)
    else:
        assert actual is None
        prepare.assert_not_called()


def test_shared_download_wrapper_propagates_search_failures(monkeypatch):
    from preprocess import preprocess
    import miletos.photometry as shared

    failure = RuntimeError('search failure')
    monkeypatch.setattr(shared, 'download_tess_light_curves', Mock(side_effect=failure))
    with pytest.raises(RuntimeError) as caught:
        preprocess(123)
    assert caught.value is failure


@pytest.mark.parametrize('injection', [False, True])
@pytest.mark.parametrize('plots', [False, True])
def test_real_collection_preprocessing_preserves_flux_errors_and_figures(monkeypatch, tmp_path, injection, plots):
    from scipy.signal import medfilt
    from preprocess import preprocess

    time = np.arange(1200) / 50  # [day]
    flux = 100 + 0.2 * np.sin(time)
    flux[100:105] -= 2
    curves = [LightCurve(time=time + offset, flux=flux, flux_err=np.full(time.size, 0.3))
              for offset in (0, 30)]
    expected_curves = []
    for curve in curves:
        normalized = curve.remove_nans()
        expected_curves.append(normalized.normalize().remove_outliers())
    expected_curves = [curve.normalize() for curve in expected_curves]
    expected = LightCurve(np.concatenate([curve.time for curve in expected_curves]),
                         np.concatenate([curve.flux for curve in expected_curves]),
                         np.concatenate([curve.flux_err for curve in expected_curves]))
    trend = medfilt(expected.flux, kernel_size=501)
    expected_flux = expected.flux - trend if injection else expected.flux - trend + 1

    monkeypatch.setattr('miletos.photometry.download_tess_light_curves',
                        lambda *args, **kwargs: [curve.copy() for curve in curves])
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'presentation_plots').mkdir()
    actual = preprocess(123, injection=injection, plots=plots, report=True)

    np.testing.assert_array_equal(actual.time.value, expected.time.value)
    np.testing.assert_array_equal(actual.flux, expected_flux)
    np.testing.assert_array_equal(actual.flux_err, expected.flux_err)
    output = sorted((tmp_path / 'presentation_plots').glob('*.png'))
    assert [filename.name for filename in output] == (
        ['123_detrendedlc.png', '123_flattenedlc.png', '123_rawlc.png'] if plots else []
    )
    if plots:
        import matplotlib.pyplot as plt
        for filename in output:
            print(f'Reading from {filename}...')
            image = plt.imread(filename)
            assert image.shape[0] > 100 and np.ptp(image[..., :3]) > 0


@pytest.mark.parametrize('include_edges', [False, True])
def test_shared_mask_retains_both_boundary_conventions(include_edges):
    from miletos.transit_search import transit_mask
    from BLStests import create_transit_mask_manual

    time = np.array([0, 0.125, 0.25, 0.5, 0.75, 0.875, 1])  # [day]
    expected = [True, True, include_edges, False, include_edges, True, True]
    np.testing.assert_array_equal(transit_mask(time, 1, 0, 0.5, include_edges=include_edges), expected)
    np.testing.assert_array_equal(create_transit_mask_manual(time, 1, 0, 0.5),
                                  transit_mask(time, 1, 0, 0.5))


def test_fixed_grid_bls_preserves_lightkurve_results():
    pytest.importorskip('lightkurve')
    from miletos.benchmarks import compare_lightkurve_bls
    from astropy.time import Time

    time = np.linspace(0, 20, 100)  # [day]
    flux = np.ones(time.size)
    flux[np.abs((time + 1.7) % 3.4 - 1.7) < 0.04] -= 0.02
    curve = LightCurve(time=time, flux=flux, flux_err=np.full(time.size, 0.001))
    comparison = compare_lightkurve_bls(curve)
    actual, expected = comparison['native'], comparison['lightkurve']
    for name in ('period', 'power', 'transit_time', 'duration', 'depth'):
        reference = getattr(expected, name)
        observed = getattr(actual, name)
        np.testing.assert_allclose(getattr(observed, 'value', observed),
                                   getattr(reference, 'value', reference), rtol=3e-15, atol=0)


def test_fixed_grid_bls_propagates_native_exceptions(monkeypatch):
    failure = RuntimeError('native BLS failure')

    monkeypatch.setattr('miletos.box_least_squares.BoxLeastSquares', Mock(side_effect=failure))
    with pytest.raises(RuntimeError) as caught:
        BLSfit(LightCurve([0., 1., 2.], [1., 1., 1.]))
    assert caught.value is failure