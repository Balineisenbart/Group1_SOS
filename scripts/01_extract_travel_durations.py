#!/usr/bin/env python3
"""
01_extract_travel_durations.py

Wireframe for a script that extracts travel durations from input data and writes
the result to an output file.

Usage:
  python 01_extract_travel_durations.py --input data/input.csv --output data/output.csv
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Any
from src.utils import setup_logging
import requests
from dotenv import load_dotenv
import os
import pandas as pd
from tqdm import tqdm


load_dotenv()

maps_api_key = os.getenv("MAPS_API_KEY")


LOG = logging.getLogger(__name__)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Extract travel durations from dataset")
  p.add_argument("-i", "--input", type=Path, required=False, help="Path to input file (.csv).", default=Path("data/input.csv"))
  p.add_argument("-o", "--output", type=Path, required=False, help="Path to output file (.csv).", default=Path("data/travel_times.csv"))
  p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
  return p.parse_args(argv)

def load_input(path: Path) -> pd.DataFrame:
  LOG.debug("Loading input from %s", path)
  if not path.exists():
    LOG.error("Input file does not exist: %s", path)
    raise FileNotFoundError(path)
    
  suffix = path.suffix.lower()

  if suffix in {".csv"}:
    return pd.read_csv(path)
  raise ValueError(f"Unsupported input file type: {suffix}")


def extract_travel_durations(df: pd.DataFrame) -> pd.DataFrame:
  """
  Placeholder for the core extraction logic.

  Expected behavior:
  - Identify origin/destination/time columns in df.
  - Compute travel duration per trip or per segment.
  - Return a DataFrame with canonical columns (e.g. trip_id, start, end, duration_s).

  Replace the example implementation with project-specific rules.
  """
  LOG.debug("Extracting all locations from dataframe with %d rows", len(df))

  required = {"osm_id", "lat", "lon"}
  if not required.issubset(df.columns):
    LOG.warning("Input missing expected columns %s; returning empty result", required)
    return pd.DataFrame(columns=["trip_id", "start_time", "end_time", "duration_s"])

  df = df.copy()

  # for testing purposes only keep first two rows
  # df = df.head(2)

  # loop all lines of df for each line extract travel times to all other lines
  results = []
  n = len(df)
  total = n * (n - 1)
  pbar = tqdm(total=total, desc="computing travel times", unit="pair")

  for idx_from, row_from in df.iterrows():
    fromId = row_from["osm_id"]
    for idx_to, row_to in df.iterrows():
      toId = row_to["osm_id"]
      if fromId == toId:
        continue

      # as number where "," is the comma
      latFrom = normalize_coord(row_from["lat"])
      lonFrom = normalize_coord(row_from["lon"])
      latTo = normalize_coord(row_to["lat"])
      lonTo = normalize_coord(row_to["lon"])

      walk_time = travel_time("WALK", latFrom, lonFrom, latTo, lonTo)
      transit_time = travel_time("TRANSIT", latFrom, lonFrom, latTo, lonTo)

      travelTimes = {
        "from": fromId,
        "to": toId,
        "walking": walk_time,
        "public_transport": transit_time
      }

      results.append(travelTimes)
      pbar.update(1)

  pbar.close()

  dfOut = pd.DataFrame(results)
  return dfOut

def travel_time(mode, lat1, lon1, lat2, lon2):
    # print("Fetching travel time from", lat1, lon1, "to", lat2, lon2, "by", mode)
    url = (
        f"https://maps.googleapis.com/maps/api/directions/json?"
        f"origin={lat1},{lon1}&destination={lat2},{lon2}"
        f"&mode={mode}&departure_time=now&key={maps_api_key}"
    )
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    try:
      return int(data["routes"][0]["legs"][0]["duration"]["value"])
    except (KeyError, IndexError, TypeError, ValueError):
      return -1

def normalize_coord(value):
    """Convert coordinate string with comma to float with dot."""
    if isinstance(value, str):
        value = value.replace(",", ".")
    return float(value)

# def osrm_walk_time(lat1, lon1, lat2, lon2):
#     url = f"https://router.project-osrm.org/route/v1/foot/{lon1},{lat1};{lon2},{lat2}?overview=false"
#     r = requests.get(url).json()
#     return r["routes"][0]["duration"] 

# def navitia_transit_time(lat1, lon1, lat2, lon2, api_key):
#     url = f"https://api.navitia.io/v1/journeys?from={lon1};{lat1}&to={lon2};{lat2}&datetime=20251108T090000"
#     headers = {"Authorization": api_key}
#     r = requests.get(url, headers=headers).json()
#     return r["journeys"][0]["duration"]

def save_output(df: pd.DataFrame, path: Path, overwrite: bool = False) -> None:
  if path.exists() and not overwrite:
    LOG.error("Output file already exists: %s (use --overwrite to replace)", path)
    raise FileExistsError(path)
  
  LOG.debug("Saving output to %s.csv", path)
  path.parent.mkdir(parents=True, exist_ok=True)
  
  df.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  setup_logging()

  try:
    df_in = load_input(args.input)
    df_out = extract_travel_durations(df_in)
    save_output(df_out, args.output, overwrite=args.overwrite)
    LOG.info("Extraction completed: wrote %d rows to %s", len(df_out), args.output)
    return 0
  except Exception as exc:
    LOG.exception("Failed to extract travel durations: %s", exc)
    return 2

if __name__ == "__main__":
  raise SystemExit(main())