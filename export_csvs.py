"""Export PERSON.csv, TRIPS.csv, HOUSEHOLDS.csv from each .db in a directory.

Cleans raw data:
- Parses WKT POINT geometries into x/y columns
- Parses transit_activity into departure_time / arrival_time
- Parses location_mapping dict into per-purpose x/y columns
- Extracts work location coordinates from location_work
- Drops bulky route WKT and redundant activity_sequence from TRIPS
"""

import argparse
import re
import sqlite3
from pathlib import Path

import pandas as pd


TRIPS_QUERY = """
SELECT
    od.origin,
    od.destination,
    od.mode,
    od.transit_activity,
    od.origin_purpose,
    od.destination_purpose,
    od.bline_distance,
    od.calculated_duration,
    od.sampled_duration,
    od.uuid_person,
    p.age,
    p.sex,
    p.age_group,
    p.primary_status,
    p.has_car,
    p.type_household
FROM od_matrix od
LEFT JOIN person p ON od.uuid_person = p.uuid
"""

PERSON_QUERY = """
SELECT
    p.uuid,
    p.uuid_household,
    p.age,
    p.sex,
    p.age_group,
    p.primary_status,
    p.has_car,
    p.has_child,
    p.child_count,
    p.is_head,
    p.is_child,
    p.type_household,
    p.type_house,
    p.origin,
    p.location_work,
    p.location_mapping
FROM person p
"""

HOUSEHOLD_QUERY = """
SELECT
    h.uuid,
    h.name_category,
    h.count_children,
    h.count_adults,
    h.count_members,
    h.count_cars,
    h.type_house,
    h.head_of_household
FROM household h
"""

# ── helpers ──────────────────────────────────────────────────────────

_POINT_RE = re.compile(r"POINT\s*\(\s*([\d.+-]+)\s+([\d.+-]+)\s*\)")
_TIME_RE = re.compile(r"Start Time:\s*([\d:]+),\s*End Time:\s*([\d:]+)")
_WORK_POINT_RE = re.compile(r"POINT\s*\(\s*([\d.+-]+)\s+([\d.+-]+)\s*\)")
_LOC_MAP_RE = re.compile(r"'(\w[\w/ ]*?)':\s*(?:<POINT\s*\(([\d.]+)\s+([\d.]+)\)>|None)")


def _parse_point(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Parse WKT POINT column into {prefix}_x, {prefix}_y."""
    extracted = series.str.extract(_POINT_RE)
    return pd.DataFrame({
        f"{prefix}_x": pd.to_numeric(extracted[0], errors="coerce"),
        f"{prefix}_y": pd.to_numeric(extracted[1], errors="coerce"),
    })


def _parse_transit_activity(series: pd.Series) -> pd.DataFrame:
    """Extract departure_time and arrival_time from transit_activity text."""
    extracted = series.str.extract(_TIME_RE)
    return pd.DataFrame({
        "departure_time": extracted[0],
        "arrival_time": extracted[1],
    })


def _parse_location_work(series: pd.Series) -> pd.DataFrame:
    """Extract x/y from 'WORK (...) - Work @ POINT (x y)'."""
    extracted = series.str.extract(_WORK_POINT_RE)
    return pd.DataFrame({
        "work_x": pd.to_numeric(extracted[0], errors="coerce"),
        "work_y": pd.to_numeric(extracted[1], errors="coerce"),
    })


def _parse_location_mapping(series: pd.Series) -> pd.DataFrame:
    """Parse the Python dict repr into per-purpose x/y columns."""
    purposes = ["Home", "Work", "Education", "Shopping", "Grocery",
                 "Leisure", "Healthcare", "Pickup/Dropoff child", "Other"]
    cols = {}
    for p in purposes:
        key = p.lower().replace("/", "_").replace(" ", "_")
        cols[f"loc_{key}_x"] = pd.Series(dtype="float64")
        cols[f"loc_{key}_y"] = pd.Series(dtype="float64")

    rows_x = {p: [] for p in purposes}
    rows_y = {p: [] for p in purposes}

    for val in series:
        found = {}
        if pd.notna(val):
            for m in _LOC_MAP_RE.finditer(str(val)):
                found[m.group(1)] = (m.group(2), m.group(3))
        for p in purposes:
            xy = found.get(p)
            rows_x[p].append(float(xy[0]) if xy and xy[0] else None)
            rows_y[p].append(float(xy[1]) if xy and xy[1] else None)

    result = {}
    for p in purposes:
        key = p.lower().replace("/", "_").replace(" ", "_")
        result[f"loc_{key}_x"] = rows_x[p]
        result[f"loc_{key}_y"] = rows_y[p]
    return pd.DataFrame(result)


# ── cleaning ─────────────────────────────────────────────────────────

def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    # Parse origin / destination points
    o = _parse_point(df["origin"], "origin")
    d = _parse_point(df["destination"], "destination")
    # Parse times from transit_activity
    times = _parse_transit_activity(df["transit_activity"])

    result = pd.concat([
        df[["uuid_person"]],
        o, d,
        df[["mode", "origin_purpose", "destination_purpose"]],
        times,
        df[["bline_distance", "calculated_duration", "sampled_duration"]],
        df[["age", "sex", "age_group", "primary_status", "has_car", "type_household"]],
    ], axis=1)

    result = result.rename(columns={"bline_distance": "distance_m"})
    return result


def clean_person(df: pd.DataFrame) -> pd.DataFrame:
    # Parse home origin
    home = _parse_point(df["origin"], "home")
    # Parse work location
    work = _parse_location_work(df["location_work"])
    # Parse full location mapping
    loc_map = _parse_location_mapping(df["location_mapping"])

    result = pd.concat([
        df[["uuid", "uuid_household", "age", "sex", "age_group",
            "primary_status", "has_car", "has_child", "child_count",
            "is_head", "is_child", "type_household", "type_house"]],
        home, work, loc_map,
    ], axis=1)
    return result


def clean_households(df: pd.DataFrame) -> pd.DataFrame:
    return df  # already clean


# ── export ───────────────────────────────────────────────────────────

EXPORTS = [
    ("TRIPS", TRIPS_QUERY, clean_trips),
    ("PERSON", PERSON_QUERY, clean_person),
    ("HOUSEHOLDS", HOUSEHOLD_QUERY, clean_households),
]


def export_db(db_path: Path, output_dir: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        for name, query, cleaner in EXPORTS:
            df = pd.read_sql_query(query, conn)
            df = cleaner(df)
            out = output_dir / f"{name}.csv"
            df.to_csv(out, index=False)
            print(f"  {name}.csv  ({len(df):,} rows, {len(df.columns)} cols)")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_dir", nargs="?",
                        default="data/processed/20241017_GAPSIM_v0_2_0",
                        help="Directory containing .db files")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as each .db)")
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    db_files = sorted(db_dir.glob("*.db"))
    if not db_files:
        print(f"No .db files found in {db_dir}")
        return

    for db_path in db_files:
        print(f"\n=== {db_path.name} ===")
        out_dir = Path(args.output_dir) if args.output_dir else db_path.parent / db_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        export_db(db_path, out_dir)
        print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
