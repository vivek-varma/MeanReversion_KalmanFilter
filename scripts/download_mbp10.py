#!/usr/bin/env python
"""Download a single year of MBP-10 depth with Databento **v0.59+**."""
import os, yaml, pathlib, databento as db
from dotenv import load_dotenv
load_dotenv("config/api_key.env")  # Load API key from .env file

CFG = yaml.safe_load(open("config/mbp10.yaml"))
DATASET  = CFG["dataset"]
SCHEMA   = CFG["schema"]
SYMS     = CFG["symbols"]
START    = CFG["start"]
END      = CFG["end"]
OUTDIR   = pathlib.Path(CFG["out_dir"])
COMPRESS = CFG.get("compression", None)  # e.g. "zstd"

key = os.getenv("DATABENTO_API_KEY")
if not key:
    raise RuntimeError("Set DATABENTO_API_KEY in env / Codespace secrets")

client = db.Historical(key)                 # 0.59 “v1” interface

print("→ streaming MBP-10 depth …")
stream = client.timeseries.get_range(
    dataset   = DATASET,
    schema    = SCHEMA,
    symbols   = SYMS,
    stype_in  = "parent",
    start     = START,
    end       = END,
)                                            # always DBN on the wire

OUTDIR.mkdir(parents=True, exist_ok=True)
fname = OUTDIR / f"ZNZF_mbp10_{START[:4]}.dbn"
if COMPRESS:
    fname = fname.with_suffix(".dbn.zst")

stream.to_file(fname, compression=COMPRESS)  # v0.59 helper
print(f"✅ saved {fname}  ({fname.stat().st_size/1e9:,.2f} GB)")
