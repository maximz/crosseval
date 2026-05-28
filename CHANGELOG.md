# Changelog

## 0.0.7

* Add `column_format="keyname"` to `get_model_comparison_stats` (and the underlying `_get_stats`/`_full_report_scores`/`aggregated_per_fold_scores`/`global_scores`) so callers can get stable scorer-keyname columns directly instead of reverse-engineering the friendly-to-keyname map. Default `"friendly"` keeps existing column names unchanged.

## 0.0.6

* Add `formatted=False` outputs for programmatic consumers of summary methods.
* Add README and uv-based development workflow.
* Harden score aggregation, incomplete fold/model detection, abstention masking, mixed label handling, and sklearn compatibility.
* Avoid importing genetools plotting/stat helpers during top-level `crosseval` import.

## 0.0.1

* First release on PyPI.
