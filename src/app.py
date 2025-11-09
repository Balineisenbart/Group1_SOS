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
from aco import ACO
from data import Item, TravelTime
from utils import setup_logging
import requests
from dotenv import load_dotenv
import os
import pandas as pd
from tqdm import tqdm

load_dotenv()

maps_api_key = os.getenv("MAPS_API_KEY")

LOG = logging.getLogger(__name__)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Run the ACO for the Seidl Rally.")
  p.add_argument("-iloc", "--input-locations", type=Path, required=False, help="Path to input file containing the location infos (.csv).", default=Path("data/input.csv"))
  p.add_argument("-idur", "--input-durations", type=Path, required=False, help="Path to input file containing the travel durations (.csv).", default=Path("data/travel_times.csv"))
  p.add_argument("-v", "--visualize", action="store_true", required=False, help="Visualize the graph.", default=False)
  return p.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  setup_logging()

  try:

    # load input locations
    locations_df = pd.read_csv(args.input_locations)
    items = []
    for _, row in locations_df.iterrows():
      items.append(
          Item(
              id=str(row["osm_id"]),
              name=str(row["name"]),
              district=int(row["district_no"]),
          )
      )
    
    # load input travel durations
    durations_df = pd.read_csv(args.input_durations)
    travel_times = []

    for _, row in durations_df.iterrows():
      travel_times.append(
        TravelTime(
          from_id=str(row['from']),
          to_id=str(row['to']),
          duration=int(min(row['walking'], row['public_transport']))
        )
      )

    aco = ACO(items=items, travel_times=travel_times)
    aco.construct_graph()

    path = aco.find_best_path()

    if args.visualize:
      aco.graph.visualize(shortest_path=path)

    return 0
  except Exception as exc:
    LOG.exception("Failed to run ACO: %s", exc)
    return 2

if __name__ == "__main__":
  raise SystemExit(main())