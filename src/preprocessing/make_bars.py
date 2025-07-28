#!/usr/bin/env python
"""
Stream-merge ZN + ZF minute bars into ONE parquet without OOM.
Works with Polars ≥ 0.20.x
"""

import glob, pathlib, pyarrow.parquet as pq, polars as pl

RAW_FILES = sorted(glob.glob("data/raw/databento/ZNZF_*.parquet"))
print("Raw files:", RAW_FILES)

out_path = pathlib.Path("data/processed/ZNZF_1m.parquet")
out_path.parent.mkdir(parents=True, exist_ok=True)

writer = None
COL_ORDER = ["ts_event", "ZN", "ZF"]        # fixed order for all chunks

for src in RAW_FILES:
    print("→", src)
    df = pl.read_parquet(src)

    # root symbol: "ZN", "ZF", "UB", "UD", ...
    df = df.with_columns(
            pl.col("symbol")
              .str.slice(0, 2)
              .alias("root")
        )

    # keep only ZN & ZF rows
    df = df.filter(pl.col("root").is_in(["ZN", "ZF"]))

    wide = (df
            .select(["ts_event", "root", "close"])
            .group_by(["ts_event", "root"])
            .agg(pl.col("close").last())
            .pivot(values="close", index="ts_event", on="root")
            .sort("ts_event"))

    # ensure both columns exist and fixed order
    for col in ["ZN", "ZF"]:
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(None).alias(col))

    wide = (wide
            .with_columns([
                pl.col("ZN").fill_null(strategy="forward"),
                pl.col("ZF").fill_null(strategy="forward")
            ])
            .select(COL_ORDER))

    tbl = wide.to_arrow()

    if writer is None:
        writer = pq.ParquetWriter(out_path, tbl.schema,
                                  compression="zstd", use_dictionary=True)

    writer.write_table(tbl)

if writer:
    writer.close()
    print("✅  wrote", out_path, f"({out_path.stat().st_size/1e6:,.1f} MB)")
else:
    print("⚠️  no data written – check RAW_FILES paths.")
