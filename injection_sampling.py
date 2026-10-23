import numpy as np


def _log_bound_ratio(min_val, max_val):
    """Retain narrow relative widths without subtracting large logarithms."""
    with np.errstate(over='ignore', under='ignore'):
        relative_width = (max_val - min_val) / min_val
        if np.isfinite(relative_width):
            return np.log1p(relative_width)
    return np.log(max_val) - np.log(min_val)


def _stable_power_quantile(min_val, max_val, exponent, uniform):
    """Evaluate the inverse CDF in logs, anchored at the dominant endpoint."""
    log_lower, log_upper = np.log(min_val), np.log(max_val)
    log_width = _log_bound_ratio(min_val, max_val)
    with np.errstate(over='ignore', under='ignore', divide='ignore'):
        if exponent == 0:
            log_samples = (1 - uniform) * log_lower + uniform * log_upper
            anchor = min_val
            log_offset = uniform * log_width
        else:
            extent = abs(exponent) * log_width
            weight = uniform if exponent < 0 else 1 - uniform
            if extent <= 1:
                log_weight = np.log1p(weight * np.expm1(-extent))
            else:
                log_weight = np.logaddexp(np.log1p(-weight), np.log(weight) - extent)
            anchor = min_val if exponent < 0 else max_val
            log_offset = log_weight / exponent
            log_samples = (log_lower if exponent < 0 else log_upper) + log_offset
        if log_width <= 1:
            samples = anchor + anchor * np.expm1(log_offset)
        else:
            samples = np.exp(log_samples)
    samples = np.where(uniform == 0, min_val, np.where(uniform == 1, max_val, samples))
    return samples[()] if samples.ndim == 0 else samples


def sample_power_law(min_val, max_val, alpha, size=None):
    """Draw from a bounded density proportional to x**(-alpha)."""
    if not np.isfinite([min_val, max_val, alpha]).all() or not 0 < min_val <= max_val:
        raise ValueError('Power-law bounds must be finite, positive, ordered, and the slope finite')
    uniform = np.random.random(size=size)
    if min_val == max_val:
        return min_val + 0 * uniform
    if alpha == 1.0:
        with np.errstate(over='ignore'):
            ratio = max_val / min_val
        if np.isfinite(ratio) and _log_bound_ratio(min_val, max_val) > 1:
            return min_val * ratio ** uniform
        return _stable_power_quantile(min_val, max_val, 0, uniform)
    exponent = 1.0 - alpha
    with np.errstate(over='ignore', under='ignore'):
        extent = abs(exponent) * _log_bound_ratio(min_val, max_val)
    if extent <= 1:
        return _stable_power_quantile(min_val, max_val, exponent, uniform)
    try:
        with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
            lower_power = min_val**exponent
            upper_power = max_val**exponent
            if lower_power > 0 and upper_power > 0 and lower_power != upper_power and np.isfinite([lower_power, upper_power]).all():
                samples = (lower_power + (upper_power - lower_power) * uniform) ** (1.0 / exponent)
                if np.isfinite(samples).all() and np.all(samples >= min_val) and np.all(samples <= max_val):
                    return samples
    except (OverflowError, ZeroDivisionError, FloatingPointError):
        pass
    return _stable_power_quantile(min_val, max_val, exponent, uniform)
