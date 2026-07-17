"""Download the Sachs 2005 flow cytometry dataset and save as sachs_real.csv.

The canonical source is the bnlearn book-CRC supplement (gzipped,
space-separated).  A plain-text fallback is provided in case the primary
URL is unavailable.

Usage
-----
    python3 data/download_sachs.py
    python3 data/download_sachs.py --output /path/to/sachs_real.csv

The saved CSV has a header row with column names matching VARS and uses
comma as the delimiter.  All 11 columns are standardised to zero mean and
unit variance.
"""

import argparse
import gzip
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Column names must match VARS in run_experiment.py exactly.
VARS = ["Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA", "PKC", "P38", "Jnk"]

# Plain TSV (854 obs, tab-separated, confirmed working 2026-03-27)
PRIMARY_URL  = "https://raw.githubusercontent.com/snarles/causal/master/bnlearn_files/sachs.data.txt"
# Gzipped fallback from bnlearn book supplement
FALLBACK_URL = "https://www.bnlearn.com/book-crc/code/sachs.data.gz"


def _fetch(url: str) -> bytes:
    """Download *url* and return raw bytes."""
    import urllib.request
    print(f"  Fetching {url} ...")
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def _parse_bytes(raw: bytes, compressed: bool) -> pd.DataFrame:
    """Decompress if needed, then parse a whitespace- or comma-separated table."""
    if compressed:
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="replace")
    # Try comma first, then whitespace
    for sep in (",", r"\s+"):
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
            if df.shape[1] >= 11:
                return df
        except Exception:
            continue
    raise ValueError("Could not parse downloaded data as a tabular file with >= 11 columns.")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to match VARS using case-insensitive prefix matching."""
    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for var in VARS:
        key = var.lower()
        # Exact match first
        if key in cols_lower:
            rename_map[cols_lower[key]] = var
            continue
        # Prefix match
        candidates = [c for c in cols_lower if c.startswith(key)]
        if len(candidates) == 1:
            rename_map[cols_lower[candidates[0]]] = var
            continue
        raise KeyError(
            f"Cannot map variable '{var}' to any column in {list(df.columns)}. "
            "Please rename columns manually."
        )
    df = df.rename(columns=rename_map)
    return df[VARS]


def download_sachs(output_path: str) -> None:
    """Download, parse, standardise, and save the Sachs dataset.

    Parameters
    ----------
    output_path : str
        Destination file path (written as comma-separated CSV with header).
    """
    raw = None
    compressed = False

    # Try primary (plain TSV, not compressed)
    try:
        raw = _fetch(PRIMARY_URL)
        compressed = False
        print("  Primary URL succeeded.")
    except Exception as exc:
        print(f"  Primary URL failed ({exc}). Trying fallback ...")

    if raw is None:
        try:
            raw = _fetch(FALLBACK_URL)
            compressed = True   # bnlearn fallback is gzipped
            print("  Fallback URL succeeded.")
        except Exception as exc:
            sys.exit(f"Both download URLs failed. Last error: {exc}")

    df = _parse_bytes(raw, compressed)
    print(f"  Parsed table: {df.shape[0]} rows x {df.shape[1]} columns.")

    df = _normalise_columns(df)
    print(f"  Columns mapped to VARS: {list(df.columns)}")

    # Standardise each column
    data = df.values.astype(np.float64)
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    df_out = pd.DataFrame(data, columns=VARS)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    print(f"  Saved {len(df_out)} observations to {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Sachs 2005 flow cytometry data to data/sachs_real.csv."
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "sachs_real.csv"),
        help="Destination CSV path (default: data/sachs_real.csv in the repo root).",
    )
    args = parser.parse_args()
    download_sachs(args.output)


if __name__ == "__main__":
    main()
