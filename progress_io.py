import csv
import json
import os
from pathlib import Path
import stat
import tempfile
import time
from catalog_values import finite_or_none

_DEFAULT = object()
_CATALOG_COLUMNS = ('ra_deg', 'dec_deg', 'tic_id', 'Tmag')


def is_completed_record(record):
    """Only an explicit done status marks a catalog query as completed."""
    return isinstance(record, dict) and record.get('status') == 'done'


def read_catalog_csv_keys(filename):
    """Return independent coordinate and TIC-ID sets from a catalog CSV."""
    coordinates, targets = set(), set()
    path = Path(filename)
    if not path.exists():
        return coordinates, targets
    print(f'Reading from {path}...')
    with path.open(newline='', encoding='utf-8') as stream:
        for raw in csv.DictReader(stream):
            row = {(name.strip() if isinstance(name, str) else name): value for name, value in raw.items()}
            ra = finite_or_none(row.get('ra_deg'))
            dec = finite_or_none(row.get('dec_deg'))
            if ra is not None and dec is not None:
                coordinates.add(f'{ra:.6f},{dec:.6f}')
            target = row.get('tic_id')
            if target is not None:
                target = str(target).strip()
                if target and target.lower() not in ('none', 'nan'):
                    targets.add(target)
    return coordinates, targets


def append_catalog_row(filename, ra, dec, tic_id, magnitude, seen_coordinates):
    """Append one coordinate-deduplicated row, recording success only after writing."""
    key = f'{float(ra):.6f},{float(dec):.6f}'
    if key in seen_coordinates:
        return False
    path = Path(filename)
    needs_header = not path.exists() or path.stat().st_size == 0
    print(f'Writing to {path}...')
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if needs_header:
            writer.writerow(_CATALOG_COLUMNS)
        writer.writerow([ra, dec, tic_id, magnitude])
    seen_coordinates.add(key)
    return True


def load_progress_json(filename, default=_DEFAULT):
    """Load progress records, preserving malformed input before returning a default."""
    path = Path(filename)
    fallback = {} if default is _DEFAULT else default
    print(f'Reading from {path}...')
    try:
        with path.open('r', encoding='utf-8') as stream:
            source = stream.read().strip()
        if not source:
            return fallback
        progress = json.loads(source)
        if not isinstance(progress, dict) or not all(isinstance(record, dict) for record in progress.values()):
            raise ValueError('Progress must be a JSON object of record objects')
        return progress
    except FileNotFoundError:
        return fallback
    except ValueError as error:
        descriptor, backup = tempfile.mkstemp(prefix=f'{path.name}.bad-{time.strftime("%Y%m%d-%H%M%S")}-', dir=path.parent)
        os.close(descriptor)
        try:
            os.replace(path, backup)
        except OSError:
            os.unlink(backup)
            raise
        print(f'[warn] Invalid progress ({error}) preserved in {backup}.')
        return fallback


def save_progress_json(filename, progress):
    """Replace progress atomically, retaining the previous file if writing fails."""
    path = Path(filename)
    print(f'Writing to {path}...')
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            if path.exists():
                os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
            json.dump(progress, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
