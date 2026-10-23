#!/usr/bin/env python3
"""
sync_csv_to_progress.py

Goal:
- For each row in the bright/faint CSVs, ensure an entry exists in wd_progress.json
  keyed as "RA6,DEC6" (rounded to 6 dp from the CSV's RA/DEC).
- Use tic_id to find/migrate existing JSON entries (move from old key to the new "RA6,DEC6" key).
- Update fields: status='done', category='bright_lc' or 'faint_lc', tic_id, Tmag.

Usage (defaults match your layout):
    python sync_csv_to_progress.py \
      --progress data_inputs/wd_progress.json \
      --bright_csv data_inputs/wd_bright_lc_summary.csv \
      --faint_csv  data_inputs/wd_faint_lc_summary.csv \
      --backup
"""

import os, json, csv, argparse, time
from catalog_values import finite_or_none
from progress_io import load_progress_json, save_progress_json

def safe_load_json(path):
    return load_progress_json(path)

def safe_save_json(path, obj):
    return save_progress_json(path, obj)

def parse_float_or_none(x):
    return finite_or_none(x)

def strip_keys(d):
    """Return a copy of row dict with header names stripped of whitespace."""
    return {(k.strip() if isinstance(k, str) else k): v for k, v in d.items()}

def make_key_ra6dec6(ra, dec):
    """Key format: '351.608864,-61.105014' (6 dp)."""
    return f"{float(ra):.6f},{float(dec):.6f}"

def tic_index(progress):
    """Build mapping tic_id -> current JSON key(s)."""
    idx = {}
    for k, v in progress.items():
        tid = v.get("tic_id")
        if tid is None:
            continue
        # keep the first occurrence; if duplicates exist, prefer not to overwrite
        idx.setdefault(str(tid), k)
    return idx

def upsert(progress, key, category, tic_id, tmag):
    """Create or update a JSON entry at 'key' with given fields."""
    existed = key in progress
    entry = progress.get(key, {})
    changed = False

    if entry.get("status") != "done":
        entry["status"] = "done"; changed = True
    if entry.get("category") != category:
        entry["category"] = category; changed = True

    if entry.get("tic_id") != tic_id:
        entry["tic_id"] = tic_id; changed = True

    if tmag is not None and entry.get("Tmag") != tmag:
        entry["Tmag"] = tmag; changed = True
    elif "Tmag" not in entry:
        # write None explicitly if not present
        entry["Tmag"] = tmag
        changed = True

    if changed or not existed:
        progress[key] = entry
        return "updated" if existed else "added"
    return "skipped"

def process_csv(csv_path, category, progress, update_existing=True, migrate_keys=True):
    """
    For each CSV row:
      - read ra/dec/tic_id/Tmag (headers may have whitespace)
      - compute key = 'RA6,DEC6'
      - if tic_id exists in JSON under a different key, move it to the new key (if migrate_keys)
      - upsert fields (status/category/tic_id/Tmag)
    """
    if not os.path.exists(csv_path):
        print(f"[info] CSV not found, skipping: {csv_path}")
        return (0, 0, 0, 0, 0)  # added, updated, skipped, migrated, skipped_rows

    added = updated = skipped = migrated = skipped_rows = 0

    # Build tic_id -> key index once per CSV to keep things O(n)
    idx = tic_index(progress)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = strip_keys(raw_row)

            # Extract fields
            ra = parse_float_or_none(row.get("ra_deg"))
            dec = parse_float_or_none(row.get("dec_deg"))
            tic_raw = row.get("tic_id")
            tic_id = None
            if tic_raw is not None:
                s = str(tic_raw).strip()
                if s and s.lower() not in ("none", "nan"):
                    tic_id = s

            tmag = parse_float_or_none(row.get("Tmag"))

            # Must have RA/DEC & tic_id to act
            if ra is None or dec is None or tic_id is None:
                skipped_rows += 1
                continue

            new_key = make_key_ra6dec6(ra, dec)

            # If this tic_id already exists in JSON under a different key, migrate
            old_key = idx.get(tic_id)
            moving = old_key is not None and old_key != new_key
            if moving and not migrate_keys:
                skipped += 1
                continue
            if new_key in progress:
                destination_tic = progress[new_key].get("tic_id")
                if moving or (destination_tic is not None and str(destination_tic) != tic_id):
                    print(f"[warn] Coordinate key {new_key} is occupied; skipping TIC {tic_id}.")
                    skipped += 1
                    continue
            if moving:
                # Move old entry to new key (merge fields conservatively)
                old_entry = progress.get(old_key, {})
                # Prepare merged entry
                merged = dict(old_entry)
                merged["status"] = "done"
                merged["category"] = category if update_existing else old_entry.get("category", category)
                merged["tic_id"] = tic_id
                if tmag is not None:
                    merged["Tmag"] = tmag
                elif "Tmag" not in merged:
                    merged["Tmag"] = None
                # Write to new key
                progress[new_key] = merged
                # Remove old key
                del progress[old_key]
                idx[tic_id] = new_key
                migrated += 1
            else:
                # Normal upsert at new_key
                entry_category = category
                if not update_existing and new_key in progress:
                    entry_category = progress[new_key].get("category", category)
                res = upsert(progress, new_key, entry_category, tic_id, tmag)
                if res == "added":
                    added += 1
                    idx.setdefault(tic_id, new_key)
                elif res == "updated":
                    updated += 1
                    idx[tic_id] = new_key
                else:
                    skipped += 1

    return (added, updated, skipped, migrated, skipped_rows)

def main():
    ap = argparse.ArgumentParser(description="Sync CSV detections into wd_progress.json keyed as 'RA6,DEC6' using tic_id.")
    ap.add_argument("--progress", default="data_inputs/wd_progress.json", help="Path to progress JSON")
    ap.add_argument("--bright_csv", default="data_inputs/wd_bright_lc_summary.csv", help="Bright LC CSV path")
    ap.add_argument("--faint_csv",  default="data_inputs/wd_faint_lc_summary.csv",  help="Faint  LC CSV path")
    ap.add_argument("--no_update_existing", action="store_true",
                    help="Do NOT change category for existing entries; only set for new ones")
    ap.add_argument("--no_migrate_keys", action="store_true",
                    help="Do NOT move existing tic_id entries to the new RA6,DEC6 key; just update in place if keys match")
    ap.add_argument("--backup", action="store_true", help="Save a timestamped backup of the JSON before syncing")
    args = ap.parse_args()

    progress_path = args.progress
    progress = safe_load_json(progress_path)

    if args.backup:
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = f"{os.path.splitext(progress_path)[0]}-before-tic-sync-{ts}.json"
        safe_save_json(bak, progress)
        print(f"[info] Backup written to: {bak}")

    added_b, updated_b, skipped_b, migrated_b, skipped_rows_b = process_csv(
        args.bright_csv, "bright_lc", progress,
        update_existing=(not args.no_update_existing),
        migrate_keys=(not args.no_migrate_keys)
    )
    added_f, updated_f, skipped_f, migrated_f, skipped_rows_f = process_csv(
        args.faint_csv, "faint_lc", progress,
        update_existing=(not args.no_update_existing),
        migrate_keys=(not args.no_migrate_keys)
    )

    safe_save_json(progress_path, progress)

    print("\n=== TIC-ID Sync summary ===")
    print(f"Bright CSV : added={added_b}, updated={updated_b}, skipped={skipped_b}, migrated={migrated_b}, skipped_rows={skipped_rows_b}")
    print(f"Faint  CSV : added={added_f}, updated={updated_f}, skipped={skipped_f}, migrated={migrated_f}, skipped_rows={skipped_rows_f}")
    print(f"TOTAL      : added={added_b+added_f}, updated={updated_b+updated_f}, "
          f"skipped={skipped_b+skipped_f}, migrated={migrated_b+migrated_f}, "
          f"skipped_rows={skipped_rows_b+skipped_rows_f}")
    print(f"Wrote: {progress_path}")

if __name__ == "__main__":
    main()
