"""Scoring of source-bound scalar extraction records (source_bound_scalar_v1).

No database, model, source-text adjudication or network access.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Callable


RECORD_SCHEMA = "titmc.extraction.records.v1"
SPLIT_SCHEMA = "titmc.extraction.split.v1"
RULE_SCHEMA = "titmc.extraction.rules.v1"
RESULT_SCHEMA = "titmc.extraction.evaluation.v1"
FIELDS = (
    "material_state", "sample", "process", "test_condition", "property",
    "quantity_role", "value", "unit", "statistic", "provenance",
)
RECORD_KEYS = {"record_id", "document_family", *FIELDS}
DEFAULT_RULES = {"schema": RULE_SCHEMA, "numeric_rules": []}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_json(value: Any, path: str = "$") -> None:
    """Reject non-finite numbers even in unused metadata or nested conditions."""
    if isinstance(value, float):
        require(math.isfinite(value), f"{path}: non-finite number")
    elif isinstance(value, dict):
        for key, child in value.items():
            require(isinstance(key, str), f"{path}: keys must be strings")
            validate_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_json(child, f"{path}[{index}]")
    else:
        require(value is None or isinstance(value, (str, int, bool)),
                f"{path}: not a JSON value")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_float(token: str) -> float:
    value = float(token)
    require(math.isfinite(value), "non-finite JSON number")
    require(Fraction(token) == Fraction(str(value)),
            "JSON fractional number would lose decimal precision or underflow; use a lossless supported value")
    return value


def decode_json(raw: bytes) -> Any:
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_float=_parse_float)
    validate_json(data)
    return data


def load_json(path: Path) -> Any:
    return decode_json(path.read_bytes())


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False) + "\n"


def object_hash(value: Any) -> str:
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def _text(value: Any, path: str) -> None:
    require(isinstance(value, str) and bool(value.strip()), f"{path}: nonempty string required")


def _number(value: Any, path: str) -> Fraction:
    require(type(value) in (int, float), f"{path}: JSON number required, not bool/string")
    if isinstance(value, float):
        require(math.isfinite(value), f"{path}: finite number required")
    return Fraction(str(value))


def _freeze(value: Any) -> Any:
    """Exact JSON semantics: number 25 equals 25.0; bool does not equal 1."""
    if type(value) in (int, float):
        return ("number", Fraction(str(value)))
    if isinstance(value, dict):
        return ("object", tuple((k, _freeze(v)) for k, v in sorted(value.items())))
    if isinstance(value, list):
        return ("array", tuple(_freeze(v) for v in value))
    return (type(value).__name__, value)


def validate_provenance(value: Any, path: str) -> None:
    require(isinstance(value, dict) and set(value) == {"asset_sha256", "locator"},
            f"{path}: asset_sha256 and locator required")
    require(isinstance(value["asset_sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", value["asset_sha256"]) is not None,
            f"{path}: lowercase 64-character SHA-256 required")
    loc = value["locator"]
    require(isinstance(loc, dict), f"{path}.locator: object required")
    kind = loc.get("kind")
    if kind == "text_span":
        require(set(loc) == {"kind", "start", "end"}, f"{path}: text_span keys")
        require(type(loc["start"]) is int and type(loc["end"]) is int
                and 0 <= loc["start"] < loc["end"], f"{path}: invalid text span")
    elif kind == "xml_element":
        require(set(loc) == {"kind", "id"}, f"{path}: xml_element keys")
        _text(loc["id"], f"{path}.locator.id")
    elif kind == "table_cell":
        require(set(loc) == {"kind", "table_id", "row", "column", "header_path", "footnotes"},
                f"{path}: table_cell requires full header/footnote keys")
        for name in ("table_id", "row", "column"):
            _text(loc[name], f"{path}.locator.{name}")
        for name in ("header_path", "footnotes"):
            require(isinstance(loc[name], list), f"{path}.locator.{name}: list required")
            for item in loc[name]:
                _text(item, f"{path}.locator.{name}")
        require(bool(loc["header_path"]), f"{path}: empty table header path")
    else:
        raise ValueError(f"{path}: supported locator kinds are text_span, xml_element, table_cell")


def validate_dataset(data: Any, *, gold: bool) -> None:
    validate_json(data)
    require(isinstance(data, dict) and data.get("schema") == RECORD_SCHEMA, "record schema mismatch")
    for key in ("benchmark_id", "evaluation_split"):
        _text(data.get(key), key)
    require(data.get("data_kind") in ("CODE_TEST_ONLY", "EXTERNAL_HUMAN_ADJUDICATED"),
            "data_kind must explicitly distinguish synthetic code tests from external gold")
    if gold and data["data_kind"] == "EXTERNAL_HUMAN_ADJUDICATED":
        adjudication = data.get("adjudication")
        require(isinstance(adjudication, dict)
                and adjudication.get("status") == "external_human_adjudicated",
                "external gold requires adjudication status declaration")
        _text(adjudication.get("reference"), "adjudication.reference")
    families = data.get("evaluation_families")
    require(isinstance(families, list), "evaluation_families must include the complete frozen denominator")
    for family in families:
        _text(family, "evaluation_families")
    require(len(set(families)) == len(families), "duplicate evaluation family")
    records = data.get("records")
    require(isinstance(records, list), "records must be a list")
    seen: set[str] = set()
    for i, record in enumerate(records):
        prefix = f"records[{i}]"
        require(isinstance(record, dict) and set(record) == RECORD_KEYS,
                f"{prefix}: keys must be exactly {sorted(RECORD_KEYS)}")
        for key in ("record_id", "document_family", "property", "quantity_role"):
            _text(record[key], f"{prefix}.{key}")
        require(record["record_id"] not in seen, f"{prefix}: duplicate record_id")
        seen.add(record["record_id"])
        require(record["document_family"] in families, f"{prefix}: family outside frozen evaluation universe")
        for key in ("material_state", "sample", "unit"):
            if record[key] is not None:
                _text(record[key], f"{prefix}.{key}")
        require(record["process"] is None or isinstance(record["process"], (dict, list)),
                f"{prefix}.process: object, ordered list or null required")
        require(record["statistic"] is None or isinstance(record["statistic"], dict),
                f"{prefix}.statistic: object or null required")
        statistic = record["statistic"]
        if statistic is not None:
            _text(statistic.get("kind"), f"{prefix}.statistic.kind")
            if statistic.get("uncertainty") is not None:
                require(_number(statistic["uncertainty"], f"{prefix}.statistic.uncertainty") >= 0,
                        f"{prefix}: uncertainty must be nonnegative")
                _text(statistic.get("uncertainty_kind"), f"{prefix}.statistic.uncertainty_kind")
                _text(statistic.get("uncertainty_unit"), f"{prefix}.statistic.uncertainty_unit")
            if statistic.get("n") is not None:
                require(type(statistic["n"]) is int and statistic["n"] > 0,
                        f"{prefix}.statistic.n: positive integer or null required")
        condition = record["test_condition"]
        require(isinstance(condition, dict) and {"mode", "temperature", "direction"} <= set(condition),
                f"{prefix}.test_condition: explicit mode, temperature and direction keys required")
        for key in ("mode", "direction"):
            if condition[key] is not None:
                _text(condition[key], f"{prefix}.test_condition.{key}")
        temperature = condition["temperature"]
        require(isinstance(temperature, dict) and set(temperature) == {"status", "value", "unit"},
                f"{prefix}: temperature requires status, value and unit")
        require(temperature["status"] in {"reported_numeric", "room_temperature_unspecified",
                                           "not_reported", "unresolved", "conflict"},
                f"{prefix}: invalid temperature status")
        if temperature["status"] == "reported_numeric":
            _number(temperature["value"], f"{prefix}.temperature.value")
            _text(temperature["unit"], f"{prefix}.temperature.unit")
        else:
            require(temperature["value"] is None and temperature["unit"] is None,
                    f"{prefix}: nonnumeric temperature status cannot imply a numeric value/unit")
        if record["value"] is None:
            require(not gold, f"{prefix}: gold records must be positive scalar numeric observations")
        else:
            _number(record["value"], f"{prefix}.value")
            _text(record["unit"], f"{prefix}.unit")
        validate_provenance(record["provenance"], f"{prefix}.provenance")


def validate_split(manifest: Any, families: list[str], split: str) -> dict[str, Any]:
    validate_json(manifest)
    require(isinstance(manifest, dict) and manifest.get("schema") == SPLIT_SCHEMA, "split schema mismatch")
    partitions = manifest.get("partitions")
    require(isinstance(partitions, dict) and split in partitions, "evaluation split absent from manifest")
    seen: dict[str, str] = {}
    for name, members in sorted(partitions.items()):
        _text(name, "partition name")
        require(isinstance(members, list), "partition members must be lists")
        for family in members:
            _text(family, "partition family")
            require(family not in seen, f"family overlap/duplicate: {family} in {seen.get(family)} and {name}")
            seen[family] = name
    require(set(partitions[split]) == set(families),
            "evaluation_families must exactly cover selected partition, including empty families")
    assets = manifest.get("family_assets", {})
    require(isinstance(assets, dict), "family_assets must be an object when supplied")
    if "family_assets" in manifest:
        require(set(assets) == set(seen), "family_assets must cover every declared family when supplied")
    owners: dict[str, str] = {}
    for family, hashes in sorted(assets.items()):
        require(family in seen and isinstance(hashes, list), "family_assets has unknown family or invalid list")
        for digest in hashes:
            require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                    "family_assets requires SHA-256 values")
            require(digest not in owners or seen[owners[digest]] == seen[family],
                    f"asset hash overlaps partitions: {digest}")
            owners[digest] = family
    return {"status": "PASS_DECLARED_IDENTIFIERS", "n_partitions": len(partitions),
            "n_families": len(seen), "n_asset_hashes": len(owners),
            "content_deduplication_verified": False, "training_contamination_verified": False}


def compile_rules(config: Any) -> dict[tuple[str, str], dict[str, Any]]:
    validate_json(config)
    require(isinstance(config, dict) and set(config) == {"schema", "numeric_rules"}
            and config["schema"] == RULE_SCHEMA, "numeric rule schema mismatch")
    require(isinstance(config["numeric_rules"], list), "numeric_rules must be a list")
    compiled = {}
    for rule in config["numeric_rules"]:
        require(isinstance(rule, dict) and set(rule) == {
            "property", "quantity_role", "canonical_unit", "units", "absolute_tolerance", "relative_tolerance"
        }, "numeric rule keys mismatch")
        for key in ("property", "quantity_role", "canonical_unit"):
            _text(rule[key], f"rule.{key}")
        key = (rule["property"], rule["quantity_role"])
        require(key not in compiled, "duplicate property/quantity_role rule")
        units = rule["units"]
        require(isinstance(units, dict) and rule["canonical_unit"] in units, "canonical unit must have identity mapping")
        transformed = {}
        for unit, transform in units.items():
            _text(unit, "rule unit")
            require(isinstance(transform, dict) and set(transform) == {"scale", "offset"}, "unit transform keys")
            scale = _number(transform["scale"], "unit scale")
            offset = _number(transform["offset"], "unit offset")
            require(scale > 0, "unit scale must be positive")
            transformed[unit] = (scale, offset)
        require(transformed[rule["canonical_unit"]] == (1, 0), "canonical mapping must be identity")
        atol = _number(rule["absolute_tolerance"], "absolute tolerance")
        rtol = _number(rule["relative_tolerance"], "relative tolerance")
        require(atol >= 0 and rtol >= 0, "tolerances must be nonnegative")
        compiled[key] = {**rule, "units": transformed, "atol": atol, "rtol": rtol}
    return compiled


def normalized_numeric(record: dict[str, Any], rules: dict) -> tuple[Any, Any, Fraction, Fraction]:
    value = None if record["value"] is None else Fraction(str(record["value"]))
    unit = record["unit"]
    rule = rules.get((record["property"], record["quantity_role"]))
    if rule is not None and unit in rule["units"]:
        scale, offset = rule["units"][unit]
        return (None if value is None else value * scale + offset,
                rule["canonical_unit"], rule["atol"], rule["rtol"])
    return value, unit, Fraction(0), Fraction(0)


def numeric_match(gold: dict, pred: dict, rules: dict) -> bool:
    if (gold["property"], gold["quantity_role"]) != (pred["property"], pred["quantity_role"]):
        return False
    gv, gu, atol, rtol = normalized_numeric(gold, rules)
    pv, pu, _, _ = normalized_numeric(pred, rules)
    if gv is None or pv is None or gu != pu:
        return False
    return abs(pv - gv) <= max(atol, rtol * abs(gv))


def maximum_matching(gold: list[dict], pred: list[dict], compatible: Callable[[dict, dict], bool]) -> list[tuple[int, int]]:
    """Maximum-cardinality bipartite matching; iterative augmenting paths.

    A greedy first-match can lose true positives with overlapping tolerances.
    Stable record_id ordering determines ties, never TP multiplicity.
    """
    adjacency = [[j for j, p in enumerate(pred) if compatible(g, p)] for g in gold]
    left_to_right: dict[int, int] = {}
    right_to_left: dict[int, int] = {}
    for root in range(len(gold)):
        queue = deque([root])
        seen_left = {root}
        parent_right: dict[int, int] = {}
        free = None
        while queue and free is None:
            left = queue.popleft()
            for right in adjacency[left]:
                if right in parent_right:
                    continue
                parent_right[right] = left
                if right not in right_to_left:
                    free = right
                    break
                other = right_to_left[right]
                if other not in seen_left:
                    seen_left.add(other)
                    queue.append(other)
        if free is not None:
            right = free
            while True:
                left = parent_right[right]
                previous = left_to_right.get(left)
                left_to_right[left] = right
                right_to_left[right] = left
                if previous is None:
                    break
                right = previous
    return sorted(left_to_right.items())


def metrics(tp: int, fp: int, fn: int) -> dict[str, Any]:
    n_pred, n_gold = tp + fp, tp + fn
    fden = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "n_gold": n_gold, "n_pred": n_pred,
            "precision": tp / n_pred if n_pred else None,
            "recall": tp / n_gold if n_gold else None,
            "f1": 2 * tp / fden if fden else None,
            "undefined": [name for name, den in (("precision", n_pred), ("recall", n_gold), ("f1", fden)) if not den]}


def _macro(items: list[dict]) -> dict[str, Any]:
    result = {"n_families": len(items)}
    for name in ("precision", "recall", "f1"):
        values = [item[name] for item in items if item[name] is not None]
        result[name] = math.fsum(values) / len(values) if values else None
        result[f"n_defined_{name}"] = len(values)
    return result


def _pooled(items: list[dict]) -> dict[str, Any]:
    return metrics(*(sum(item[key] for item in items) for key in ("tp", "fp", "fn")))


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def family_bootstrap(items: list[dict], repetitions: int, seed: int) -> dict:
    require(type(repetitions) is int and repetitions > 0, "bootstrap repetitions must be positive integer")
    require(type(seed) is int, "bootstrap seed must be an explicit integer")
    require(bool(items), "bootstrap requires at least one declared family")
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {"micro_f1": [], "family_macro_f1": []}
    for _ in range(repetitions):
        # Keep repeated draws as distinct copies. Do not group by original ID.
        draw = [items[rng.randrange(len(items))] for _ in items]
        for name, value in (("micro_f1", _pooled(draw)["f1"]), ("family_macro_f1", _macro(draw)["f1"])):
            if value is not None:
                samples[name].append(value)
    return {"method": "paper_family_cluster_percentile_linear_interpolation",
            "confidence_level": 0.95, "repetitions": repetitions, "seed": seed,
            "resample_unit": "document_family", "n_families": len(items),
            "warnings": (["INSUFFICIENT_INDEPENDENT_FAMILIES_INTERVAL_IS_DEGENERATE"] if len(items) < 2 else [])
                        + ["CONDITIONAL_ON_SAMPLED_FAMILIES_NOT_SOURCE_REPRESENTATIVENESS"],
            "intervals": {name: {"lower": _quantile(values, .025), "upper": _quantile(values, .975),
                                 "defined_replicates": len(values),
                                 "undefined_replicates": repetitions - len(values)}
                          for name, values in samples.items()}}


def evaluate(gold: dict, predictions: dict, split_manifest: dict, *, rules: dict | None = None,
             bootstrap_repetitions: int = 0, seed: int | None = None) -> dict[str, Any]:
    validate_dataset(gold, gold=True)
    validate_dataset(predictions, gold=False)
    for name in ("benchmark_id", "evaluation_split", "data_kind"):
        require(gold[name] == predictions[name], f"gold/prediction {name} mismatch")
    require(set(gold["evaluation_families"]) == set(predictions["evaluation_families"]),
            "gold/prediction evaluation universe mismatch")
    split_report = validate_split(split_manifest, gold["evaluation_families"], gold["evaluation_split"])
    if "family_assets" in split_manifest:
        for dataset in (gold, predictions):
            for record in dataset["records"]:
                require(record["provenance"]["asset_sha256"] in split_manifest["family_assets"][record["document_family"]],
                        "record asset is not registered to its declared family")
    config = DEFAULT_RULES if rules is None else rules
    compiled = compile_rules(config)
    require(type(bootstrap_repetitions) is int and bootstrap_repetitions >= 0, "invalid bootstrap repetitions")
    if bootstrap_repetitions:
        require(type(seed) is int, "bootstrap requires --seed")
    elif seed is not None:
        require(type(seed) is int, "seed must be integer")

    def strict(g: dict, p: dict) -> bool:
        return (all(_freeze(g[k]) == _freeze(p[k]) for k in FIELDS if k not in ("value", "unit"))
                and g["document_family"] == p["document_family"] and numeric_match(g, p, compiled))

    def field_match(g: dict, p: dict, field: str) -> bool:
        if g["document_family"] != p["document_family"]:
            return False
        if field == "provenance":
            return _freeze(g[field]) == _freeze(p[field])
        if _freeze(g["provenance"]) != _freeze(p["provenance"]):
            return False
        if field == "value":
            return numeric_match(g, p, compiled)
        if field == "unit":
            return (g["property"] == p["property"] and g["quantity_role"] == p["quantity_role"]
                    and normalized_numeric(g, compiled)[1] == normalized_numeric(p, compiled)[1])
        return _freeze(g[field]) == _freeze(p[field])

    grouped_gold: dict[str, list] = defaultdict(list)
    grouped_pred: dict[str, list] = defaultdict(list)
    for record in gold["records"]:
        grouped_gold[record["document_family"]].append(record)
    for record in predictions["records"]:
        grouped_pred[record["document_family"]].append(record)
    per_family, matches = [], []
    field_counts: dict[str, list] = {field: [] for field in FIELDS}
    for family in sorted(gold["evaluation_families"]):
        gs = sorted(grouped_gold[family], key=lambda r: r["record_id"])
        ps = sorted(grouped_pred[family], key=lambda r: r["record_id"])
        pairs = maximum_matching(gs, ps, strict)
        per_family.append({"document_family": family, **metrics(len(pairs), len(ps) - len(pairs), len(gs) - len(pairs))})
        matches.extend({"document_family": family, "gold_record_id": gs[i]["record_id"],
                        "prediction_record_id": ps[j]["record_id"]} for i, j in pairs)
        for field in FIELDS:
            # Null is not a positive field claim. There is no true-negative reward.
            fg, fp = [r for r in gs if r[field] is not None], [r for r in ps if r[field] is not None]
            n = len(maximum_matching(fg, fp, lambda g, p: field_match(g, p, field)))
            field_counts[field].append(metrics(n, len(fp) - n, len(fg) - n))
    output = {
        "schema": RESULT_SCHEMA, "scoring_mode": "source_bound_scalar_v1",
        "benchmark_id": gold["benchmark_id"], "data_kind": gold["data_kind"],
        "claim_scope": "CODE_TEST_ONLY_NOT_PROJECT_ACCURACY" if gold["data_kind"] == "CODE_TEST_ONLY"
                       else "SCORES_SUPPLIED_EXTERNAL_GOLD_ONLY_NOT_INDEPENDENT_SOURCE_VALIDATION",
        "source_truth_verified": False, "adjudication_declaration": gold.get("adjudication"),
        "scoring_contract": {"endpoint": "strict_record_fidelity_to_supplied_gold",
                             "allows_gold_declared_null_state": True,
                             "is_state_complete_precision": False,
                             "coverage_is_separate_from_fidelity": True,
                             "field_nulls_count_as_true_negatives": False},
        "input_hashes": {"basis": "sorted_indented_utf8_json_plus_newline",
                         "gold_sha256": object_hash(gold), "predictions_sha256": object_hash(predictions),
                         "split_manifest_sha256": object_hash(split_manifest), "rules_sha256": object_hash(config)},
        "rules": config, "split_check": split_report,
        "counts": {"n_families": len(per_family), "n_gold": len(gold["records"]),
                   "n_pred": len(predictions["records"]),
                   "n_null_value_predictions": sum(r["value"] is None for r in predictions["records"]),
                   "n_duplicate_gold_records": len(gold["records"]) - len({_freeze({k: v for k, v in r.items() if k != "record_id"}) for r in gold["records"]}),
                   "n_duplicate_prediction_records": len(predictions["records"]) - len({_freeze({k: v for k, v in r.items() if k != "record_id"}) for r in predictions["records"]}),
                   "gold_null_fields": {field: sum(r[field] is None for r in gold["records"]) for field in FIELDS},
                   "prediction_null_fields": {field: sum(r[field] is None for r in predictions["records"]) for field in FIELDS},
                   "gold_nonnumeric_test_temperature": sum(r["test_condition"]["temperature"]["value"] is None for r in gold["records"]),
                   "prediction_nonnumeric_test_temperature": sum(r["test_condition"]["temperature"]["value"] is None for r in predictions["records"]),
                   "n_empty_gold_families": sum(item["n_gold"] == 0 for item in per_family)},
        "strict_record_micro": _pooled(per_family), "strict_record_family_macro": _macro(per_family),
        "field_diagnostics": {"definition": "independent_one_to_one_source_anchored_projections_nonnull_only",
                              "not_joint_record_f1": True,
                              "metrics": {field: _pooled(counts) for field, counts in field_counts.items()}},
        "per_family": per_family, "strict_matches": matches,
        "bootstrap": family_bootstrap(per_family, bootstrap_repetitions, seed) if bootstrap_repetitions else None,
    }
    return output


def _bounded_text(value: Any, *, unit: bool = False, label: bool = False) -> str:
    """Presentation normalization only; units retain case and dimensional meaning."""
    import unicodedata
    _text(value, "bounded label")
    value = "".join(unicodedata.normalize("NFKC", value).split())
    if label:
        require(not unit, "Unit/native quotation normalization must preserve punctuation")
        value = re.sub(r"[_\-‐‑‒–—()\[\]{}]", "", value)
    return value if unit else value.casefold()


def _bounded_aliases(rules: dict, field: str, *, unit: bool = False) -> dict[str, str]:
    declared = rules[field]
    require(isinstance(declared, dict), field + " must be a canonical-to-alias mapping")
    result = {}
    for canonical, aliases in declared.items():
        _text(canonical, field)
        require(isinstance(aliases, list), field + " aliases must be lists")
        for alias in [canonical, *aliases]:
            key = _bounded_text(alias, unit=unit, label=not unit)
            require(key not in result or result[key] == canonical,
                    field + ": ambiguous declared alias " + alias)
            result[key] = canonical
    return result


def _bounded_endpoint_diagnostic(record: Any, maps: dict, endpoint_terms: dict) -> dict | None:
    """Report vocabulary support independently of state resolution; never gate a match.

    A generic endpoint name may have qualifying source evidence. This diagnostic
    checks only the current reference's declared vocabulary, not that evidence or
    physical truth, and never selects a replacement property or reference record.
    """
    if not isinstance(record, dict):
        return None
    property_label, endpoint_label = record.get("property"), record.get("reported_endpoint")
    if not all(isinstance(value, str) and value.strip() for value in (property_label, endpoint_label)):
        return None  # Malformed fields remain the existing record validator's responsibility.
    prop = maps["property_aliases"].get(_bounded_text(property_label, label=True))
    endpoint = _bounded_text(endpoint_label, label=True)
    declared_properties = sorted(name for name, terms in endpoint_terms.items() if endpoint in terms)
    if prop is not None and prop in declared_properties:
        return None
    status = ("UNDECLARED_PROPERTY_LABEL" if prop is None else
              "UNSUPPORTED_PROPERTY_ENDPOINT_PAIR_IN_CURRENT_REFERENCE")
    return {"status": status, "submitted_property": property_label,
            "submitted_reported_endpoint": endpoint_label, "canonical_submitted_property": prop,
            "declared_endpoint_properties": declared_properties,
            "source_evidence_adjudicated": False}


def _bounded_core(record: dict, state: str, condition: dict, maps: dict,
                  endpoint_terms: dict, *, gold: bool) -> dict:
    """Resolve only the submitted record/state fields; never consult a gold candidate."""
    def alias(value, name, nullable=False):
        if value is None and nullable:
            return None
        key = _bounded_text(value, unit=name == "unit_aliases", label=name != "unit_aliases")
        require(key in maps[name], "Undeclared " + name + ": " + str(value))
        return maps[name][key]

    def numeric(value, name, *, integer=False):
        if value is None:
            return None
        require(type(value) is int if integer else type(value) in (int, float),
                name + ": native JSON number required")
        require(math.isfinite(value), name + ": finite number required")
        return value

    require(isinstance(record, dict) and isinstance(condition, dict), "Record/condition must be objects")
    prop = alias(record["property"], "property_aliases")
    endpoint = _bounded_text(record["reported_endpoint"], label=True)
    require(endpoint in endpoint_terms[prop], "Property and reported endpoint are incompatible")
    require(record["quantity_role"] == "author_reported", "Only author-reported endpoints are in scope")
    require(record["sample"] is None or isinstance(record["sample"], str), "Invalid individual specimen identity")
    _text(record["native_value_text"], "native_value_text")
    approximate = record["qualifications"]["approximate"] if gold else record["approximate"]
    require(type(approximate) is bool, "approximate must be an explicit boolean")
    temperature, rate, statistic = condition["temperature"], condition["rate"], record["statistic"]
    require(all(isinstance(x, dict) for x in (temperature, rate, statistic)), "Structured condition/statistic required")
    status = temperature["status"]
    require(status in {"reported_numeric", "room_temperature_unspecified", "not_reported", "unresolved", "conflict"},
            "Invalid temperature status")
    require(status == "reported_numeric" or (temperature["value"] is None and temperature["unit"] is None),
            "Unknown/room-temperature status cannot carry an invented numeric temperature")
    require(status != "reported_numeric" or temperature["value"] is not None, "Reported temperature needs a number")
    values = [numeric(record["value"], "value"), numeric(statistic["uncertainty"], "uncertainty"),
              numeric(statistic["n"], "statistic.n", integer=True),
              numeric(temperature["value"], "temperature"), numeric(rate["value"], "rate"),
              numeric(condition["reported_repeat_count"], "reported_repeat_count", integer=True)]
    require(values[0] is not None, "Endpoint value cannot be null")
    require(values[1] is None or values[1] >= 0, "Negative uncertainty")
    require(all(values[i] is None or values[i] > 0 for i in (2, 5)), "Repeat counts must be positive")
    semantic = (state, prop, record["quantity_role"], alias(record["unit"], "unit_aliases"),
                record["sample"], approximate,
                alias(condition["mode"], "mode_aliases", True), status,
                alias(temperature["unit"], "unit_aliases", True),
                alias(condition["direction"], "direction_aliases", True),
                None if rate["reported_label"] is None else _bounded_text(rate["reported_label"], label=True),
                alias(rate["unit"], "unit_aliases", True),
                alias(statistic["kind"], "statistic_kind_aliases"),
                alias(statistic["uncertainty_kind"], "uncertainty_kind_aliases"),
                alias(statistic["uncertainty_unit"], "unit_aliases", True),
                tuple(value is None for value in values))
    return {"semantic": semantic, "numbers": [0 if value is None else value for value in values],
            "state": state, "property": prop}


def _bounded_locator(locator: Any, native: dict) -> dict:
    """Check a submitted quote in submitted native nodes, without resolving a state."""
    require(isinstance(locator, dict), "Locator must be an object")
    required = {"xml_node_ids", "table_id", "data_row_1based", "column_1based", "source_quote"}
    require(required <= locator.keys(), "Incomplete native locator")
    ids = locator["xml_node_ids"]
    require(isinstance(ids, list) and all(isinstance(v, str) and v for v in ids), "Invalid native node IDs")
    table = locator["table_id"]
    require(table is None or isinstance(table, str), "Invalid table ID")
    candidates = set(ids) | ({table} if table else set())
    require(bool(candidates) and candidates <= native.keys(), "Locator is outside the declared XML scope")
    quote = _bounded_text(locator["source_quote"], unit=True)
    require(any(quote in native[node] for node in candidates), "Quote is not present in the submitted native location")
    for key in ("data_row_1based", "column_1based"):
        value = locator[key]
        require(value is None or (type(value) is int and value > 0), "Invalid table coordinate")
    return {"ids": candidates, "table": table, "row": locator["data_row_1based"],
            "column": locator["column_1based"], "quote": quote}


def _bounded_cuda_metrics(torch, rows: list[list[int]]) -> dict:
    """All reported count reductions and rates run on CUDA, including missing sources."""
    counts = torch.tensor(rows, dtype=torch.int64, device="cuda")

    def rates(values):
        values = values.to(torch.float64)
        tp, fp, fn = values.unbind(-1)
        denominator = torch.stack((tp + fp, tp + fn, 2 * tp + fp + fn), dim=-1)
        numerator = torch.stack((tp, tp, 2 * tp), dim=-1)
        return torch.where(denominator > 0, numerator / denominator.clamp_min(1), 0.0)

    totals = counts.sum(0)
    names = ("precision", "recall", "f1")
    return {"micro": {**dict(zip(("tp", "fp", "fn"), totals.cpu().tolist())),
                      **dict(zip(names, rates(totals).cpu().tolist()))},
            "source_macro": {**dict(zip(names, rates(counts).mean(0).cpu().tolist())),
                             "n_sources": len(rows), "zero_division": 0},
            "per_source": [{**dict(zip(("tp", "fp", "fn"), count)), **dict(zip(names, value))}
                           for count, value in zip(counts.cpu().tolist(), rates(counts).cpu().tolist())]}


def evaluate_bounded_state_records(tasks: list[dict], references: list[dict],
                                   responses: list[dict], *, run_id: str,
                                   method_scope: dict | None = None) -> dict:
    """Score actual bounded-record responses; native semantics are declared before inference.

    Only the input task's source is inherited. State, property, conditions and
    relation endpoints are resolved from model declarations, never gold values
    or locators. This is a three-source development-exposed replay, with no CI.
    """
    from collections import Counter
    from itertools import combinations
    from xml.etree import ElementTree as ET
    _text(run_id, "run_id")
    require(isinstance(tasks, list) and bool(tasks), "Declared task execution protocol is required")
    execution_protocol = tasks[0].get("execution_protocol")
    require(isinstance(execution_protocol, dict), "Task execution_protocol is missing")
    permitted_runs = execution_protocol.get("run_ids")
    require(isinstance(permitted_runs, list) and bool(permitted_runs)
            and all(isinstance(value, str) and value for value in permitted_runs), "Explicit execution run IDs are required")
    if method_scope is None:
        require(run_id in permitted_runs, "Run ID is not permitted by these tasks; a retired pilot cannot be rescored under a revised protocol")
    require(all(task.get("execution_protocol") == execution_protocol for task in tasks),
            "Sources do not share one declared execution protocol")
    require(isinstance(references, list) and bool(references)
            and all(ref.get("execution_run_ids") == permitted_runs for ref in references),
            "Task/reference execution identities differ; use the original archived protocol for historical results")
    import torch
    require(torch.cuda.is_available(), "GPU_REQUIRED: bounded record numerical comparison and statistics")
    require(isinstance(responses, list), "Actual response envelopes are required")
    selected = [(i, row) for i, row in enumerate(responses)
                if isinstance(row, dict) and row.get("run_id") == run_id]
    require(bool(selected), "No actual response for this run; synthetic empty predictions are forbidden")
    task_map = {row["task_id"]: row for row in tasks}
    reference_map = {row["task_id"]: row for row in references}
    require(len(tasks) == len(task_map) == len(references) == len(reference_map) == 3
            and task_map.keys() == reference_map.keys(), "The complete three-source denominator is required")
    require(len({task["source"]["doi"] for task in tasks}) == 3,
            "Different task IDs cannot turn one source into independent source units")
    fields = ("state_aliases", "property_aliases", "unit_aliases", "mode_aliases", "direction_aliases",
              "statistic_kind_aliases", "uncertainty_kind_aliases", "unknown_field_aliases")
    prepared = {}
    for tid, task in task_map.items():
        ref = reference_map[tid]
        require(ref["source"] == task["source"] and ref["reference_type"] == "source_replayed"
                and ref["exposure"] == "development_exposed", "Reference identity/exposure differs")
        rules = ref["scoring_rules"]
        require(rules["numeric_comparison"] == {"native_units_only": True,
                "absolute_tolerance": 0, "relative_tolerance": 0}, "Only declared exact native units are supported")
        maps = {name: _bounded_aliases(rules, name, unit=name == "unit_aliases") for name in fields}
        endpoints = {prop: {_bounded_text(value, label=True) for value in values}
                     for prop, values in rules["endpoint_terms"].items()}
        root = ET.fromstring(task["source_xml"])
        nodes = {node.get("id"): node for node in root.iter() if node.get("id")}
        require(all(node in nodes for node in task["scope_xml_ids"]), "Declared native XML scope is missing")
        native = {node.get("id"): _bounded_text("".join(node.itertext()), unit=True)
                  for scope in task["scope_xml_ids"] for node in nodes[scope].iter()
                  if node.get("id") and "".join(node.itertext()).strip()}
        gold_records, gold_states = ref["records"], {row["material_state"]: row for row in ref["states"]}
        require(len(gold_states) == len(ref["states"]), "Duplicate reference state")
        gold = []
        for index, record in enumerate(gold_records):
            state = maps["state_aliases"][_bounded_text(record["material_state"], label=True)]
            require(state in gold_states and record["provenance"]["asset_sha256"] == task["source"]["sha256"],
                    "Reference record source/state binding differs")
            gold.append({**_bounded_core(record, state, record["test_condition"], maps, endpoints, gold=True),
                         "record_id": record["record_id"], "index": index, "raw": record})
        require(len({row["record_id"] for row in gold}) == len(gold), "Duplicate reference record ID")
        groups = [tuple(sorted(relation["record_ids"])) for relation in ref["relations"]]
        require(all(relation["same_specimen"] is None and relation["relation"] == "same_reported_state_test_group"
                    for relation in ref["relations"]), "Reference relation semantics differ")
        require(all(len(group) >= 2 and len(set(group)) == len(group) for group in groups)
                and sorted(record_id for group in groups for record_id in group) == sorted(row["record_id"] for row in gold),
                "Reference groups must partition the complete endpoint population once")
        gold_by_id = {row["record_id"]: row for row in gold}
        require(all({gold_by_id[record_id]["state"] for record_id in relation["record_ids"]} == {relation["material_state"]}
                    for relation in ref["relations"]), "Reference relation crosses reported states")
        edges = [tuple(pair) for group in groups for pair in combinations(group, 2)]
        unknown = [(state, name, status) for state, row in gold_states.items()
                   for name, status in row["unreported_fields"].items()]
        reviewed = {(state, field) for state, field, _ in unknown}
        reported = {(state, field) for state, names in rules["reported_fields_by_state"].items() for field in names}
        require(not reviewed.intersection(reported), "Reference marks the same field reported and unreported")
        prepared[tid] = {"maps": maps, "endpoints": endpoints, "native": native, "gold": gold,
                         "groups": groups, "edges": edges, "unknown": unknown, "reviewed": reviewed | reported}
    require(sum(len(p["gold"]) for p in prepared.values()) == 39
            and sum(len(p["groups"]) for p in prepared.values()) == 10
            and sum(len(p["edges"]) for p in prepared.values()) == 64,
            "The declared 39-fact/10-group/64-edge population differs")
    scope_metadata = None
    if method_scope is not None:
        # Validate the complete unchanged benchmark before selecting a view.
        require(isinstance(method_scope, dict)
                and method_scope.get("protocol_name") == "source-method-adaptation"
                and method_scope.get("base_execution_protocol") == execution_protocol,
                "Method scope must inherit the actual common base execution protocol")
        applicable = method_scope.get("applicable_sources")
        require(isinstance(applicable, list) and bool(applicable)
                and all(isinstance(doi, str) and doi for doi in applicable)
                and len(set(applicable)) == len(applicable)
                and set(applicable) <= {t["source"]["doi"] for t in tasks},
                "Method scope must declare a unique subset of the original sources")
        witnesses = [(i, row) for i, row in enumerate(responses) if isinstance(row, dict)
                     and row.get("protocol") == "source-method-adaptation"
                     and row.get("execution_protocol") == method_scope]
        require(bool(witnesses), "No actual method-adaptation envelope declares this scope")
        witness_run = witnesses[0][1].get("run_id")
        method_runs, repetition = method_scope.get("run_ids"), method_scope.get("repetition_index")
        require(isinstance(method_runs, list) and len(method_runs) == 3
                and all(isinstance(value, str) and value for value in method_runs)
                and len(set(method_runs)) == 3 and type(repetition) is int and repetition in (1, 2, 3)
                and witness_run == method_runs[repetition - 1], "Method repetition identity differs")
        profile_name = witnesses[0][1].get("arm")
        require(method_scope.get("arms") == [profile_name]
                and method_scope.get("model_call_limits") == [3]
                and method_scope.get("exposure") == "development_exposed"
                and method_scope.get("cumulative_observed_output_token_budget") == 24000
                and method_scope.get("task_wall_budget_seconds") == 900
                and method_scope.get("source_tool_limit") == 4
                and isinstance(method_scope.get("stage_prompts"), list)
                and len(method_scope["stage_prompts"]) == 3
                and all(isinstance(prompt, str) and prompt.strip() for prompt in method_scope["stage_prompts"]),
                "Method arm, exposure, budget or declared prompts differ")
        witnessed_tasks = []
        for index, row in enumerate(responses):
            if not isinstance(row, dict) or row.get("run_id") != witness_run:
                continue
            declaration = row.get("method_adaptation") or {}
            task = task_map.get(row.get("task_id"))
            require(task is not None and task["source"]["doi"] in applicable
                    and row.get("source_doi") == task["source"]["doi"]
                    and row.get("source_sha256") == task["source"]["sha256"],
                    "Adaptation envelope has an untrusted or inapplicable source")
            require(row.get("protocol") == "source-method-adaptation"
                    and row.get("execution_protocol") == method_scope and row.get("arm") == profile_name
                    and declaration.get("profile") == profile_name
                    and declaration.get("applicable_sources") == applicable
                    and declaration.get("declared_run_ids") == method_runs
                    and declaration.get("repetition_index") == repetition
                    and declaration.get("stage_prompts") == method_scope["stage_prompts"]
                    and row.get("model") == execution_protocol["model"]
                    and row.get("reasoning_effort") == execution_protocol["reasoning_effort"],
                    "Actual adaptation envelope does not match the supplied method declaration")
            actual_budget = row.get("budget") or {}
            require(actual_budget.get("output_tokens") == 24000 and actual_budget.get("wall_seconds") == 900
                    and actual_budget.get("model_calls") == 3 and actual_budget.get("source_tool_calls") == 4,
                    "Actual adaptation budget differs")
            witnessed_tasks.append(row["task_id"])
        require(len(witnessed_tasks) == len(set(witnessed_tasks)), "Duplicate adaptation run/source envelopes")
        base_run = run_id in permitted_runs
        require(base_run or run_id == witness_run, "Run is neither a legal base run nor the witnessed adaptation repetition")
        # Base A/B/C/D runs may include the other sources. Validate them before
        # applying the declared subset; model correctness never selects sources.
        for _, row in selected:
            task = task_map.get(row.get("task_id"))
            require(task is not None and row.get("source_doi") == task["source"]["doi"]
                    and row.get("source_sha256") == task["source"]["sha256"], "Selected response source differs")
            require((base_run and row.get("protocol") == "bounded-record-api"
                     and row.get("execution_protocol") == execution_protocol
                     and row.get("arm") in execution_protocol["arms"])
                    or (not base_run and row.get("execution_protocol") == method_scope),
                    "Selected response does not have its legal base/adaptation execution identity")
        included = [tid for tid, task in task_map.items() if task["source"]["doi"] in applicable]
        scope_metadata = {"declaration": method_scope, "witness_run_id": witness_run,
                          "witness_response_indices": [i for i, _ in witnesses],
                          "included_task_ids": included,
                          "not_applicable_task_ids": [tid for tid in task_map if tid not in included],
                          "partial_adaptation_run": set(witnessed_tasks) != set(included),
                          "evaluated_arms": list(dict.fromkeys(row["arm"] for _, row in selected)),
                          "selection_basis": "Predeclared source applicability, independent of status, values or matching outcomes",
                          "comparison_limit": "Same source subset and unchanged scoring; capture/transport implementation differences remain separate from method effects"}
        prepared = {tid: prepared[tid] for tid in included}
        task_map = {tid: task_map[tid] for tid in included}
        selected = [(i, row) for i, row in selected if row["task_id"] in task_map]
    by_arm = defaultdict(list)
    if scope_metadata is not None:
        # An observed base-run arm with no returned applicable source must not
        # disappear from the comparison: the ordinary missing-source FN apply.
        for arm in scope_metadata["evaluated_arms"]:
            by_arm[arm] = []
    for index, row in selected:
        require(row.get("task_id") in task_map, "Untrusted outer source task")
        source = task_map[row["task_id"]]["source"]
        require(row.get("source_doi") == source["doi"] and row.get("source_sha256") == source["sha256"],
                "Outer response source identity differs")
        _text(row.get("arm"), "arm")
        by_arm[row["arm"]].append((index, row))
    categories = ("core_fact", "source_bound_core_fact", "relation_edges", "complete_groups", "unknown_reviewed_fields")
    runs = []
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        for arm, envelopes in by_arm.items():
            per_source, diagnostics, matches = [], [], []
            for tid in task_map:
                context = prepared[tid]
                maps, native, gold = context["maps"], context["native"], context["gold"]
                predicted, relations, unknown_predictions, unscored_unknown = [], [], [], []
                invalid, raw_envelopes = [], []
                endpoint_diagnostics = []

                def rejected(kind, index, payload, reason):
                    invalid.append({"kind": kind, "response_index": index, "payload": payload, "reason": reason})

                def resolve_state(state):
                    label = _bounded_text(state["material_state"], label=True)
                    require(label in maps["state_aliases"], "Undeclared output material state")
                    return maps["state_aliases"][label]

                for envelope_index, envelope in envelopes:
                    if envelope["task_id"] != tid:
                        continue
                    raw_envelopes.append({"response_index": envelope_index, "response": {
                        key: envelope.get(key) for key in ("run_id", "arm", "task_id", "status", "protocol",
                            "source_doi", "source_sha256", "model", "reasoning_effort", "budget", "cost",
                            "elapsed_seconds", "error")}})
                    answer = envelope.get("parsed_answer")
                    if not isinstance(answer, dict):
                        no_final = envelope.get("_protocol_failure_no_final") or envelope.get("status") in {"FAILED", "BUDGET_EXHAUSTED"}
                        if envelope.get("raw_answer") and not no_final:
                            predicted.append({"valid": False, "raw": envelope["raw_answer"]})
                            rejected("invalid_final_response", envelope_index, envelope["raw_answer"], "No usable final JSON object")
                        continue
                    conflict = answer.get("task_id") != tid
                    states = answer.get("states")
                    if not isinstance(states, list):
                        rejected("states_container", envelope_index, states, "states must be an array")
                        states = []
                    state_ids = Counter(s.get("state_id") for s in states if isinstance(s, dict) and isinstance(s.get("state_id"), str))
                    state_map = {s["state_id"]: s for s in states if isinstance(s, dict)
                                 and isinstance(s.get("state_id"), str) and state_ids[s["state_id"]] == 1}
                    records = answer.get("records")
                    if not isinstance(records, list):
                        rejected("records_container", envelope_index, records, "records must be an array; no scalar-to-array repair")
                        records = [None]
                    record_ids = Counter(r.get("record_id") for r in records if isinstance(r, dict) and isinstance(r.get("record_id"), str))
                    local_record_map = {}
                    for record in records:
                        position = len(predicted)
                        item = {"valid": False, "raw": record, "response_index": envelope_index}
                        if not conflict:
                            endpoint_diagnostic = _bounded_endpoint_diagnostic(record, maps, context["endpoints"])
                            if endpoint_diagnostic is not None:
                                endpoint_diagnostics.append({"response_index": envelope_index,
                                    "prediction_index": position, "record_id": record.get("record_id"),
                                    "state_ref": record.get("state_ref"), **endpoint_diagnostic})
                        try:
                            require(not conflict, "Model task ID conflicts with its trusted input source")
                            require(isinstance(record, dict), "Record is not an object")
                            _text(record.get("record_id"), "record_id")
                            state_ref = record["state_ref"]
                            _text(state_ref, "state_ref")
                            require(isinstance(state_ref, str) and state_ref in state_map, "Missing or ambiguous state_ref")
                            state_object = state_map[state_ref]
                            state = resolve_state(state_object)
                            item.update(_bounded_core(record, state, state_object["test_condition"], maps,
                                                      context["endpoints"], gold=False), valid=True)
                            try:
                                item["locator"] = _bounded_locator(record.get("source_locator"), native)
                                item["condition_locator"] = _bounded_locator(state_object["test_condition"].get("source_locator"), native)
                                item["state_locator"] = _bounded_locator(state_object.get("source_locator"), native)
                            except (ValueError, KeyError, TypeError) as exc:
                                item["locator_error"] = str(exc)
                        except (ValueError, KeyError, TypeError, OverflowError) as exc:
                            rejected("record", envelope_index, record, str(exc))
                        predicted.append(item)
                        if isinstance(record, dict) and isinstance(record.get("record_id"), str) and record_ids[record["record_id"]] == 1:
                            local_record_map[record["record_id"]] = position
                    submitted_relations = answer.get("relations")
                    if not isinstance(submitted_relations, list):
                        rejected("relations_container", envelope_index, submitted_relations, "relations must be an array")
                        submitted_relations = [None]
                    for relation in submitted_relations:
                        relations.append({"raw": relation, "records": local_record_map, "conflict": conflict,
                                          "response_index": envelope_index})
                    submitted_unknown = answer.get("unreported_fields")
                    if not isinstance(submitted_unknown, list):
                        rejected("unreported_fields_container", envelope_index, submitted_unknown, "unreported_fields must be an array")
                        submitted_unknown = [None]
                    for declaration in submitted_unknown:
                        try:
                            require(not conflict and isinstance(declaration, dict), "Invalid unreported-field declaration")
                            state_ref = declaration["state_ref"]
                            _text(state_ref, "state_ref")
                            require(isinstance(state_ref, str) and state_ref in state_map, "Unresolved unreported-field state_ref")
                            state = resolve_state(state_map[state_ref])
                            field = maps["unknown_field_aliases"].get(_bounded_text(declaration["field"], label=True))
                            if field is None or (state, field) not in context["reviewed"]:
                                unscored_unknown.append({"response_index": envelope_index, "declaration": declaration,
                                                         "status": "OUTSIDE_REVIEWED_FIELD_SCOPE"})
                                continue
                            require(declaration["status"] in {"not_reported_in_declared_scope", "unresolved", "not_applicable"},
                                    "Invalid unreported-field status")
                            unknown_predictions.append((state, field, declaration["status"]))
                        except (ValueError, KeyError, TypeError) as exc:
                            unknown_predictions.append(None)
                            rejected("unreported_field", envelope_index, declaration, str(exc))
                valid_positions = [i for i, item in enumerate(predicted) if item["valid"]]
                numeric = {}
                if valid_positions:
                    left = torch.tensor([row["numbers"] for row in gold], dtype=torch.float64, device="cuda")
                    right = torch.tensor([predicted[i]["numbers"] for i in valid_positions], dtype=torch.float64, device="cuda")
                    equal = (left[:, None, :] == right[None, :, :]).all(-1).cpu().tolist()
                    numeric = {(i, position): equal[i][j] for i in range(len(gold)) for j, position in enumerate(valid_positions)}
                candidates = [{**item, "index": i} for i, item in enumerate(predicted)]

                def compatible(g, p):
                    return p["valid"] and g["semantic"] == p["semantic"] and numeric.get((g["index"], p["index"]), False)

                def located(g, p):
                    if not compatible(g, p) or "locator_error" in p:
                        return False
                    loc, expected = p["locator"], g["raw"]["provenance"]["locator"]
                    condition_id = g["raw"]["test_condition"]["source_xml_id"]
                    condition_loc = p["condition_locator"]
                    if condition_id not in condition_loc["ids"] or condition_loc["quote"] not in native[condition_id]:
                        return False
                    # Identity was already matched independently. A table quote
                    # must retain its whole native cell, including uncertainty.
                    if expected["kind"] == "table_cell":
                        return ((loc["table"], loc["row"], loc["column"]) == (expected["table_id"], expected["row"], expected["column"])
                                and _bounded_text(g["raw"]["native_value_text"], unit=True) in loc["quote"]
                                and loc["quote"] in native[expected["table_id"]])
                    # Prose may quote a shorter native clause. Do not demand the
                    # arbitrary length of the reference builder's regex slice.
                    clause = _bounded_text(expected["source_quote"], unit=True)
                    number = str(g["raw"]["value"]).removesuffix(".0")
                    native_number = re.search(r"(?<![0-9.+\-−])" + re.escape(number) + r"(?![0-9.])", loc["quote"])
                    native_unit = g["raw"]["unit"] == "dimensionless strain" or _bounded_text(g["raw"]["unit"], unit=True) in loc["quote"]
                    return (expected["id"] in loc["ids"] and loc["quote"] in native[expected["id"]]
                            and (clause in loc["quote"] or loc["quote"] in clause) and bool(native_number) and native_unit
                            and loc["table"] is None and loc["row"] is None and loc["column"] is None)

                core_pairs = maximum_matching(gold, candidates, compatible)
                bound_pairs = maximum_matching(gold, candidates, located)
                mapping = {pi: gold[gi]["record_id"] for gi, pi in bound_pairs}
                matches.extend({"task_id": tid, "gold_record_id": gold[gi]["record_id"],
                                "prediction_index": pi, "response_index": predicted[pi]["response_index"],
                                "prediction_record_id": predicted[pi]["raw"]["record_id"]}
                               for gi, pi in bound_pairs)
                predicted_edges, predicted_groups = [], []
                for item in relations:
                    relation, record_map = item["raw"], item["records"]
                    ids = relation.get("record_ids") if isinstance(relation, dict) else None
                    valid = False
                    try:
                        require(not item["conflict"] and isinstance(relation, dict), "Invalid relation object/task")
                        require(relation["relation"] == "same_reported_state_test_group" and relation["same_specimen"] is None,
                                "Only same reported group with unknown individual specimen identity is supported")
                        require(isinstance(ids, list) and len(ids) >= 2 and all(isinstance(v, str) and v for v in ids),
                                "Relation needs at least two declared record IDs")
                        relation_locator = _bounded_locator(relation.get("source_locator"), native)
                        result_ids = {row["provenance"]["result_xml_id"] for row in reference_map[tid]["relations"]}
                        require(any(node in relation_locator["ids"] and relation_locator["quote"] in native[node]
                                    for node in result_ids), "Relation quote is not in a declared result-group location")
                        valid = True
                    except (ValueError, KeyError, TypeError) as exc:
                        rejected("relation", item["response_index"], relation, str(exc))
                    pairs = list(combinations(ids, 2)) if isinstance(ids, list) and len(ids) >= 2 else [(None, None)]
                    for a, b in pairs:
                        endpoints = [mapping.get(record_map.get(v)) if isinstance(v, str) else None for v in (a, b)]
                        predicted_edges.append(tuple(sorted(endpoints)) if valid and a != b and all(v is not None for v in endpoints) else None)
                    group = [mapping.get(record_map.get(v)) if isinstance(v, str) else None for v in ids] if isinstance(ids, list) else []
                    predicted_groups.append(tuple(sorted(group)) if valid and all(v is not None for v in group)
                                            and len(set(group)) == len(group) else None)

                def identity_count(expected, submitted):
                    return len(maximum_matching(expected, submitted, lambda g, p: p is not None and g == p))

                sizes = {"core_fact": (len(core_pairs), len(predicted), len(gold)),
                         "source_bound_core_fact": (len(bound_pairs), len(predicted), len(gold)),
                         "relation_edges": (identity_count(context["edges"], predicted_edges), len(predicted_edges), len(context["edges"])),
                         "complete_groups": (identity_count(context["groups"], predicted_groups), len(predicted_groups), len(context["groups"])),
                         "unknown_reviewed_fields": (identity_count(context["unknown"], unknown_predictions), len(unknown_predictions), len(context["unknown"]))}
                per_source.append({"task_id": tid, "counts": {name: [tp, n_pred - tp, n_gold - tp]
                                                             for name, (tp, n_pred, n_gold) in sizes.items()}})
                diagnostics.append({"task_id": tid, "submitted_envelopes": raw_envelopes, "invalid_outputs": invalid,
                                    "endpoint_definition_diagnostics": endpoint_diagnostics,
                                    "unscored_unknown_declarations": unscored_unknown,
                                    "unmatched_core_prediction_indices": [i for i in range(len(predicted)) if i not in {p for _, p in core_pairs}],
                                    "locator_failures": [{"prediction_index": i, "reason": p["locator_error"]}
                                                         for i, p in enumerate(predicted) if "locator_error" in p],
                                    "source_missing": not raw_envelopes})
            results = {}
            for category in categories:
                results[category] = _bounded_cuda_metrics(torch, [source["counts"][category] for source in per_source])
                for row, source in zip(results[category]["per_source"], per_source):
                    row["task_id"] = source["task_id"]
            runs.append({"arm": arm, "metrics": results, "source_bound_matches": matches, "diagnostics": diagnostics})
        torch.cuda.synchronize()
    kernels = sorted({event.name for event in profile.events() if "CUDA" in str(event.device_type)
                      and not event.name.startswith("Memcpy")})
    require(bool(kernels), "GPU_REQUIRED: no observed CUDA numerical kernels")
    result = {"scoring_mode": "bounded_state_record_source_replay", "run_id": run_id,
            "execution_protocol": execution_protocol,
            "reference_corrections": {ref["task_id"]: ref.get("reference_corrections", []) for ref in references},
            "reference_type": "source_replayed", "development_exposed_sources": list(task_map),
            "denominator": {"sources": 3, "core_facts": 39, "state_groups": 10, "within_group_edges": 64},
            "runs": runs, "bootstrap": None, "ci95": None,
            "claim_scope": "Core fact and declared co-membership reconstruction in three previously exposed XML scopes; not blind or source-sufficient validation",
            "not_scored": {"process": "NOT_SCORED_FREE_TEXT", "geometry": "NOT_SCORED_FREE_TEXT",
                           "qualifications": "NOT_SCORED_FREE_TEXT_EXCEPT_EXPLICIT_APPROXIMATE_FLAG",
                           "reported_material_and_composition": "NOT_SCORED_FREE_TEXT",
                           "unknown_explanation": "NOT_SCORED_FREE_TEXT"},
            "matching": "Existing maximum_matching for small identity graphs; CUDA native numeric equality and count/rate reductions; no gold-derived prediction labels",
            "endpoint_diagnostic_scope": "Non-gating declared-vocabulary checks before state resolution; no source-evidence or physical-error adjudication, replacement labels or metric changes. Relation matching still depends on source-bound core matches; an inherited endpoint failure does not establish an independent relation error.",
            "locator_scope": "Table quotations retain the complete native value/uncertainty cell. Prose quotations may be shorter compatible native clauses containing the matched number and native unit; dimensionless strain has no printed unit. Both must lie in the actual result node; condition quotations lie in the native test paragraph. Free process/state explanation completeness is not scored.",
            "gpu": {"backend": "torch CUDA", "device": torch.cuda.get_device_name(),
                    "tensor_device": "cuda:" + str(torch.cuda.current_device()),
                    "observed_cuda_kernels": kernels}, "scorer_model_calls": 0, "database_access": False}
    if scope_metadata is not None:
        result["method_scope"] = scope_metadata
        result["execution_protocol"] = execution_protocol if run_id in permitted_runs else method_scope
        # Structural lengths derive from the original reference view; numerical
        # metrics and their source reductions remain in the same CUDA owner.
        result["denominator"] = {"sources": len(task_map),
            "core_facts": sum(len(p["gold"]) for p in prepared.values()),
            "state_groups": sum(len(p["groups"]) for p in prepared.values()),
            "within_group_edges": sum(len(p["edges"]) for p in prepared.values())}
        result["claim_scope"] = "Unchanged core-fact/co-membership scoring on the predeclared applicable-source subset; development-exposed, not official-method replication or independent validation"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--bootstrap", type=int, default=0, metavar="REPETITIONS")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, help="New receipt; default stdout. Cannot overwrite an input.")
    args = parser.parse_args(argv)
    inputs = {"gold": args.gold, "predictions": args.predictions, "split_manifest": args.split_manifest}
    if args.rules is not None:
        inputs["rules"] = args.rules
    try:
        require(args.output is None or args.output.resolve() not in {p.resolve() for p in inputs.values()},
                "output must not overwrite an input")
        require(args.output is None or not args.output.exists(), "output must be a new file; refusing to overwrite")
        raw_inputs = {name: path.read_bytes() for name, path in inputs.items()}
        loaded = {name: decode_json(raw) for name, raw in raw_inputs.items()}
        result = evaluate(loaded["gold"], loaded["predictions"], loaded["split_manifest"],
                          rules=loaded.get("rules"), bootstrap_repetitions=args.bootstrap, seed=args.seed)
        result["input_file_hashes"] = {name: hashlib.sha256(raw).hexdigest()
                                      for name, raw in raw_inputs.items()}
        result["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if args.output is None:
            print(json_text(result), end="")
        else:
            # Exclusive creation also protects hardlink aliases and a creation race.
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(json_text(result))
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"validation error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
