# Databricks notebook source
# MAGIC %md
# MAGIC # 00_common: shared settings, paths and rules
# MAGIC Used by every job task through `%run ./00_common`. Not a job task itself.
# MAGIC
# MAGIC **Adapted from** the DSAI3202 Lab-5 (bronze to silver) and Lab-6 (quality gate) notebooks:
# MAGIC same ideas (widgets/job parameters, one folder per run, refuse to overwrite a run,
# MAGIC row accounting, gate verdict saved as JSON). **Changes for this project:** Dubai Ejari rent
# MAGIC columns and rules from our Phase-0 proposal, CSV source in `raw/`, and a content
# MAGIC fingerprint next to the lab's full `exceptAll` content comparison for reruns.

# COMMAND ----------

import json, time, datetime
from pyspark.sql import functions as F, Window
spark.conf.set("spark.sql.session.timeZone", "UTC")   # same dates/years on every cluster (as in Lab-6)

# ---------- job parameters (the job passes these; widgets are for manual runs) ----------
dbutils.widgets.text("lake", "amazonlake60306080")
dbutils.widgets.text("catalog", "dsai3202_databricks")
dbutils.widgets.text("schema", "s_60306080")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("input_path", "processed/dld_rents/landing/v1/DLD_Rents.csv")
dbutils.widgets.text("compare_to", "")
dbutils.widgets.text("max_reject_share", "0.80")

LAKE       = dbutils.widgets.get("lake").strip()
CATALOG    = dbutils.widgets.get("catalog").strip()
SCHEMA     = dbutils.widgets.get("schema").strip()
RUN_ID     = dbutils.widgets.get("run_id").strip()
INPUT_REL  = dbutils.widgets.get("input_path").strip().lstrip("/")
COMPARE_TO = dbutils.widgets.get("compare_to").strip()
MAX_REJECT_SHARE = float(dbutils.widgets.get("max_reject_share"))

if not RUN_ID or "{{" in RUN_ID:
    raise ValueError("run_id is empty. In the job, set run_id = {{job.run_id}}. "
                     "For a manual test, type e.g. manual-01 in the run_id widget.")

def abfss(rel):
    """'container/path' -> abfss URL in our lake."""
    container, _, path = rel.partition("/")
    return f"abfss://{container}@{LAKE}.dfs.core.windows.net/{path}"

# ---------- lake layout ----------
INPUT_PATH   = abfss(INPUT_REL)
RUN_DIR      = abfss(f"processed/dld_rents/runs/run_id={RUN_ID}")
BRONZE_PATH  = f"{RUN_DIR}/bronze"           # Parquet, 16 of 28 columns
SILVER_PATH  = f"{RUN_DIR}/silver"           # Delta, accepted rows of this run
REJECTS_PATH = f"{RUN_DIR}/rejects"          # Parquet, rejected rows + reason
QUALITY_DIR  = abfss(f"curated/dld_rents/quality/run_id={RUN_ID}")
GOLD_PATH    = abfss("curated/dld_rents/gold")   # Delta, the published table
GOLD_TABLE   = f"{CATALOG}.{SCHEMA}.dld_rents_gold"

# ---------- data contract (Phase-0 rules) ----------
SOURCE_COLUMNS = 28
KEEP_COLUMNS = [            # "reduce early": drop landmark/metro/mall/project-number columns
    "contract_id", "contract_reg_type_en", "contract_start_date", "contract_end_date",
    "contract_amount", "annual_amount", "no_of_prop", "line_number", "is_free_hold",
    "ejari_property_type_en", "ejari_property_sub_type_en", "property_usage_en",
    "area_id", "area_name_en", "actual_area", "tenant_type_en",
]
DATE_FMT = "dd-MM-yyyy"           # e.g. 07-04-2019
YEAR_MIN, YEAR_MAX = 2010, 2023   # 2024 is partial; later years are entry errors
DUR_MIN, DUR_MAX = 330, 400       # roughly one-year contracts
MIN_SIZE_M2 = 10                  # size must be > 10 m2
MIN_ANNUAL_AED = 5000
EXCLUDED_TYPE_REGEX = r"(?i)(labou?r|portacabin|camp)"   # labour camps, camp rooms, portacabins
MODEL_FIELDS = ["area_name_en", "ejari_property_type_en",
                "ejari_property_sub_type_en", "contract_reg_type_en"]
TRAIN_YEARS, TEST_YEARS = (2010, 2021), (2022, 2023)

# Reject reasons, in the order they are tested. A row gets its FIRST failed rule only.
REASONS = ["missing_contract_id", "bad_start_date", "bad_annual_amount", "not_residential",
           "excluded_type", "year_out_of_range", "bad_duration", "bad_size", "low_rent",
           "missing_model_field", "duplicate_contract_id"]

# ---------- helpers ----------
def path_exists(p):
    try:
        dbutils.fs.ls(p)
        return True
    except Exception:
        return False

def write_json(path, obj, overwrite=False):
    dbutils.fs.mkdirs(path.rsplit("/", 1)[0])
    dbutils.fs.put(path, json.dumps(obj, indent=2, default=str), overwrite)
    print(f"saved {path}")

def read_json(path):
    return json.loads(dbutils.fs.head(path, 1_000_000))

def fingerprint(df, cols):
    """Order-independent content fingerprint: row count + sum of 64-bit row hashes."""
    r = df.select(F.count(F.lit(1)).alias("n"),
                  F.sum(F.xxhash64(*[F.col(c) for c in sorted(cols)]).cast("decimal(38,0)")).alias("h")
                 ).collect()[0]
    return {"rows": int(r["n"]), "hash_sum": str(r["h"])}

def same_content(left, right):
    """Lab-6 method: all rows equal in both directions (exceptAll keeps duplicates). Order is irrelevant."""
    assert set(left.columns) == set(right.columns), "Column names differ."
    cols = sorted(left.columns)
    a, b = left.select(*cols), right.select(*cols)
    assert a.schema.simpleString() == b.schema.simpleString(), "Column types differ."
    return a.exceptAll(b).limit(1).count() == 0 and b.exceptAll(a).limit(1).count() == 0

def now_utc():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

print(f"run_id={RUN_ID}\ninput={INPUT_PATH}\nrun_dir={RUN_DIR}\nquality={QUALITY_DIR}\ngold={GOLD_TABLE}")
