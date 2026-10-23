> [!WARNING]
> **Currently under active development.**

# albus
Could a planet passing in front of a white dwarf leave a recognizable dip in its light curve? Albus gives us a place to test that question with simulated transit injections and light-curve analysis using [Astropy](https://github.com/astropy/astropy), [batman](https://github.com/lkreidberg/batman), and [Lightkurve](https://github.com/lightkurve/lightkurve). The pipeline is still under active development.

## Synthetic BLS example

The [synthetic BLS notebook](examples/synthetic_bls_pipeline.ipynb) sends a
seeded toy transit through `BLSfit()`, `BLSResults()`, and `FoldedLC()`, then
saves the periodogram and folded curve under `visuals/`. Run it with a Python
environment that includes Lightkurve and TDpy. The recovered period illustrates
the pipeline on known synthetic input; it is not a planet detection or a survey
completeness result.

Our `paths.py` module reads `ALBUS_PATH`, which defaults to this checkout. It exposes local `data/` and `visuals/` directories. Git ignores both.

We draw planet radii for injections with a shared bounded power-law sampler.
Its `size=None` default returns one value, while an integer or shape tuple returns
an array. A slope of one gives a log-uniform distribution. These draws use NumPy's
global random state, so `np.random.seed(...)` remains effective.
A shared log-space inverse CDF handles failed direct calculations and small
exponent spans, including slopes near one. This avoids cancellation while
retaining one uniform random draw per sample.
Narrow-bound draws use relative widths and endpoint offsets to preserve intervals
at very large or small absolute scales without clipping samples to the bounds.

Our injection drivers retry target selection when preprocessing raises an exception
or returns no light curve, up to the configured attempt limit. The serial injection driver uses ten
attempts. We retain a fixed-density stellar-radius approximation with mean
density $10^9\,\mathrm{kg\,m^{-3}}$ and propagate only the catalog mass uncertainty.

Our non-injection driver makes up to five draws, sampling uniformly by position
from all current catalog rows. Empty catalogs need no draws. Targets without a
preprocessed light curve remain eligible for later attempts. We skip IDs already
in the resume CSV and targets completed during the current run, recording
completion only after all candidate rows are written. Interrupted targets can
still leave partial CSV rows.

We use Lightkurve's median-flux normalization for both flux and its reported
uncertainty. Returned detrended curves retain those normalized per-point errors,
including missing values. They do not include uncertainty in the estimated
median-filter trend or the correlations introduced by detrending.

We convert Astropy time quantities to days before computing manual transit masks.
Plain numeric times and epochs are interpreted as days with a common origin,
while Astropy `Time` inputs use Julian dates. Transit boundaries remain excluded.
Invalid periods, nonfinite epochs, and negative or nonfinite durations raise
explicit errors. Periods must be finite and positive. A zero duration is valid
and produces an empty mask.
Extracted best-fit periods and transit durations retain their supplied units.
Empty in-transit selections yield `NaN` depth estimates. SNR is undefined when
either sample set is empty, and out-of-transit scatter is `NaN` without baseline
samples. These cases do not fabricate zero depths or noise estimates.

We select BLS period candidates before rendering. Disabling, showing, or saving
the periodogram plot returns the same candidates and best-fit transit parameters.
We keep the raw-power maximum first and add distinct periods from the full
detrended ranking, up to four candidates. Earlier runs could omit the strongest
detrended peak when the raw and detrended maxima differed.
By default, `BLSResults()` keeps its five-value return with the raw best-fit epoch
and duration. Our injection and non-injection drivers request
`candidate_ephemerides=True` to receive epochs and durations aligned with the
returned periods. Each candidate is folded and
tested with its own fitted parameters. Earlier runs reused the raw best-fit epoch
and duration for every candidate, which could shift secondary transit windows.
Failed BLS fits propagate their original exceptions rather than returning text
that downstream code could mistake for a periodogram.
BLS periods and powers must be nonempty, aligned one-dimensional arrays, with
finite powers and finite positive periods. Invalid trials raise an error before
smoothing or selecting a maximum. No powers are silently filled or dropped.
Period-bin summaries omit empty bins while retaining the full periodogram curve.
Earlier runs with plotting disabled may have searched fewer period candidates.
Folded-curve plots honor save/show requests for both binned and unbinned curves.
Binned calls retain their existing return value regardless of the `output` flag.
Unbinned calls return the folded curve when `output=True`.

`BLSResults()` and `FoldedLC()` use white backgrounds with black axes by default.
Pass `typeplotback='dark'` for black backgrounds with white axes. Both classifier
plotting scripts accept `--typeplotback dark`, including the older workflow's
error histograms and corner plots. These plotting helpers require TDpy.

We treat masked, nonfinite, and unreadable TIC magnitudes as missing values.
Serial matching keeps its first-result policy, while parallel matching prefers
the nearest source with finite `Tmag` when source coordinates are available.
We schedule each six-decimal coordinate key only once in parallel. For repeated
keys, the retained final catalog row supplies both the query coordinates and the
parallel FITS export. Serial FITS exports retain all matching catalog rows.

Our CSV-to-progress synchronizer skips rows with nonfinite coordinates and
reports additions separately from updates. Its `--no_update_existing` option
preserves existing categories while refreshing other fields. With `--backup`,
the saved JSON contains the progress state before either CSV is processed.
Empty resume CSVs contribute no completed targets. Nonfinite coordinates are
excluded from coordinate keys while valid TIC IDs remain available for matching.
With `--no_migrate_keys`, existing TIC entries at other coordinates are skipped.
Occupied destination keys are reported and left unchanged during migration.

We preserve malformed progress JSON and invalid record structures in uniquely
named `.bad-*` files before starting fresh. Read failures and failed preservation
stop processing. Progress writes use atomic replacement and retain existing file
permissions. They do not merge concurrent updates, so each progress file needs
one writer at a time.

Importing the serial query helpers does not start catalog processing or create
output files. The serial script loads its inputs when run directly and uses the
same validated progress reader and atomic writer as the parallel workflow.
Both catalog workflows restore missing CSV rows for completed targets still in
the selected input catalog when saved TIC IDs and finite magnitudes are available.
They use the same coordinate-deduplicating CSV writer without repeating queries.
Only `status='done'` marks a JSON query record complete. Retrying an unfinished
record refreshes query fields and preserves unrelated saved metadata.
Exhausted TIC or MAST request exceptions leave targets unfinished for later runs.
Successful empty responses remain valid negative results. Previously completed
records are retained and may need reevaluation if earlier service failures were
mistaken for empty responses.
