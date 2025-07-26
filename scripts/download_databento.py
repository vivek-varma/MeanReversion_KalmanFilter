#!/usr/bin/env python
"""
Safely download 1-minute OHLCV for ZN + ZF from Databento:
• Auto-discovers the dataset's last available date
• Adds adjust_continuous only if the SDK version supports it
• Stops before burning credits on non-existent data
"""
import os, yaml, inspect, datetime as dt, pathlib, tqdm, databento as db
from dotenv import load_dotenv
load_dotenv("config/api_key.env")  # Load API key from .env file

# ---------- config ---------------------------------------------------
CFG = yaml.safe_load(open("config/databento.yaml"))
DATASET, SCHEMA   = CFG["dataset"], CFG["schema"]
SYMS, ADJ         = CFG["symbols"], CFG["adjust"]
OUTDIR            = pathlib.Path(CFG["out_dir"])
START_DATE        = dt.datetime.fromisoformat(CFG["start"])
REQ_END_DATE      = dt.datetime.fromisoformat(CFG["end"])

# ---------- auth -----------------------------------------------------
API_KEY = os.getenv("DATABENTO_API_KEY")
if not API_KEY:
    raise RuntimeError("❌  Set DATABENTO_API_KEY in your env or Codespace secrets")

client = db.Historical(API_KEY)
client.set_limits(max_bytes=120_000_000)  # 120 MB cap

# ---------- discover real dataset end date ---------------------------
meta = client.metadata.get_dataset_range(dataset=DATASET)
DATASET_END = dt.datetime.fromisoformat(meta["end"])
if DATASET_END < REQ_END_DATE:
    print(f"ℹ️  {DATASET} currently ends {DATASET_END:%Y-%m-%d}. "
          f"Clipping requests after that date.")
FINAL_END = min(REQ_END_DATE, DATASET_END)

# ---------- helper ---------------------------------------------------
HAS_ADJUST = "adjust_continuous" in inspect.signature(
                 client.timeseries.get_range).parameters
print(f"SDK adjust_continuous support: {HAS_ADJUST}")

def fetch_year(year):
    start = dt.datetime(year, 1, 2)
    end   = dt.datetime(year, 12, 31, 23, 59)
    if end < START_DATE or start > FINAL_END:
        return                                  # outside desired range
    if end > FINAL_END:
        end = FINAL_END                         # clip last partial year
    kwargs = dict(
        dataset = DATASET,
        schema  = SCHEMA,
        symbols = SYMS,
        stype_in= "parent",
        start   = start.isoformat(),
        end     = end.isoformat(),
        encoding="dbn",
    )
    if "adjust_continuous" in inspect.signature(client.timeseries.get_range).parameters:
        KWARGS["adjust_continuous"] = ADJ
    print(f"→ {year}: {kwargs['start']} → {kwargs['end']}")
    frame = client.timeseries.get_range(**kwargs)
    path  = OUTDIR / f"ZNZF_{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_df().to_parquet(path)
    print(f"   saved {path.name:14} "
          f"({path.stat().st_size/1e6:,.1f} MB)")

# ---------- loop -----------------------------------------------------
for yr in tqdm.tqdm(range(START_DATE.year, FINAL_END.year + 1)):
    fetch_year(yr)

print("✅  download complete")
