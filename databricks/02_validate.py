# Databricks notebook source
# MAGIC %md
# MAGIC # 02_validate: quality gate (task 2 of 3)
# MAGIC Checks this run's silver data. Saves the verdict to `gate.json`.
# MAGIC **If any FAIL rule fails, this task raises an error, so `03_publish` does not run**
# MAGIC and the published gold table stays unchanged.
# MAGIC
# MAGIC | Policy | Meaning |
# MAGIC |---|---|
# MAGIC | FAIL | stop publication |
# MAGIC | WARN | publish, but record it for the README/EDA |

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

clean = read_json(f"{QUALITY_DIR}/clean.json")
silver = spark.read.format("delta").load(SILVER_PATH)
n = silver.count()

EXPECTED = {"contract_id", "start_date", "contract_year", "contract_month", "reg_type", "area_name_en",
            "property_type", "property_sub_type", "size_m2", "annual_amount_aed", "monthly_rent_aed"}
year_counts = {r["contract_year"]: r["count"] for r in silver.groupBy("contract_year").count().collect()}
missing_years = [y for y in range(YEAR_MIN, YEAR_MAX + 1) if year_counts.get(y, 0) == 0]
thin_years = [y for y in range(YEAR_MIN, YEAR_MAX + 1) if 0 < year_counts.get(y, 0) < 10_000]

agg = silver.agg(F.countDistinct("contract_id").alias("distinct_ids")).collect()[0]
null_model = silver.filter(" OR ".join(f"{c} IS NULL OR trim({c}) = ''" for c in
                           ["area_name_en", "property_type", "property_sub_type", "reg_type"])
                          + " OR size_m2 IS NULL OR monthly_rent_aed IS NULL OR start_date IS NULL").count()
out_of_rule = silver.filter(
    f"contract_year NOT BETWEEN {YEAR_MIN} AND {YEAR_MAX} OR size_m2 <= {MIN_SIZE_M2} "
    f"OR annual_amount_aed < {MIN_ANNUAL_AED} OR duration_days NOT BETWEEN {DUR_MIN} AND {DUR_MAX}").count()
extreme = silver.filter("monthly_rent_aed > 1000000").count()

checks = [
  # name, policy, passed, detail
  ("schema_complete", "FAIL", EXPECTED.issubset(set(silver.columns)),
       f"missing={sorted(EXPECTED - set(silver.columns))}"),
  ("not_empty", "FAIL", n > 0, f"accepted_rows={n}"),
  ("accounting_balanced", "FAIL", bool(clean["balanced"]) and clean["accepted_rows"] == n,
       f"input={clean['input_rows']} accepted={clean['accepted_rows']} rejected={clean['rejected_rows']}"),
  ("contract_id_unique", "FAIL", agg["distinct_ids"] == n, f"rows={n} distinct_ids={agg['distinct_ids']}"),
  ("model_fields_present", "FAIL", null_model == 0, f"rows_with_missing_fields={null_model}"),
  ("phase0_rules_hold", "FAIL", out_of_rule == 0, f"rows_breaking_rules={out_of_rule}"),
  ("reject_share_max", "FAIL", (clean["reject_share"] or 1.0) <= MAX_REJECT_SHARE,
       f"reject_share={clean['reject_share']} limit={MAX_REJECT_SHARE} "
       "(a sudden jump means the source format changed)"),
  ("all_years_present", "FAIL", not missing_years,
       f"missing_years={missing_years} (train {TRAIN_YEARS}, test {TEST_YEARS} must all exist)"),
  ("thin_years", "WARN", not thin_years, f"years_under_10k_rows={thin_years}"),
  ("extreme_rent", "WARN", extreme == 0, f"rows_monthly_rent_over_1M_AED={extreme}"),
]

failed = [c[0] for c in checks if c[1] == "FAIL" and not c[2]]
verdict = {
  "run_id": RUN_ID, "checked_utc": now_utc(), "accepted_rows": n,
  "verdict": "FAIL" if failed else "PASS", "failed_rules": failed,
  "checks": [{"rule": c[0], "policy": c[1], "passed": bool(c[2]), "detail": c[3]} for c in checks],
  "rows_per_year": {str(k): v for k, v in sorted(year_counts.items())},
}
write_json(f"{QUALITY_DIR}/gate.json", verdict)
for c in checks:
    print(f"{'PASS' if c[2] else c[1]:5} {c[0]:22} {c[3]}")

if failed:
    raise RuntimeError(f"QUALITY GATE FAILED: {', '.join(failed)}. Nothing published.")
print("QUALITY GATE PASSED")
