#!/usr/bin/env python3
"""Check release checksums, cohort membership, and the identity of saved results."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def rows(root, name):
    with (root / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root: Path):
    manifest = json.loads((root / "EXPORT_MANIFEST.json").read_text("utf-8"))
    inputs = json.loads((root / "DATA_INPUTS.json").read_text("utf-8"))
    qualification = json.loads((root / "SOURCE_QUALIFICATION.json").read_text("utf-8"))
    expected = inputs["expected"]
    known = set()
    total_bytes = 0
    for entry in manifest["files"]:
        name = entry["path"]
        path = root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(root), "Unsafe path: " + name)
        payload = path.read_bytes()
        require(len(payload) == entry["bytes"], "Byte count differs: " + name)
        require(hashlib.sha256(payload).hexdigest() == entry["sha256"], "Checksum differs: " + name)
        require(name not in known, "Duplicate manifest path: " + name)
        known.add(name)
        total_bytes += len(payload)
    for entry in inputs["files"]:
        payload = (root / entry["path"]).read_bytes()
        require(entry["path"] in known and hashlib.sha256(payload).hexdigest() == entry["sha256"],
                "Data index differs from the export: " + entry["path"])

    decision_rows = rows(root, "paper01/source_data/TITMC_PAPER01_source_heldout_decisions.csv")
    frozen = [r for r in decision_rows if r["admitted"] == "True" and r["method"] == "ridge"]
    by_id = {r["observation_id_uts"]: r for r in frozen}
    require(len(by_id) == len(frozen) == expected["frozen_candidates"], "Frozen candidate identities differ.")
    require(len({r["paper_family_id"] for r in frozen}) == expected["frozen_sources"], "Frozen source count differs.")
    removed = qualification["removed_pairs"]
    require(len(removed) == expected["excluded_pairs"], "Exclusion count differs.")
    excluded_ids = set()
    for row in removed:
        candidate = by_id[row["observation_id_uts"]]
        require(candidate["observation_id_el"] == row["observation_id_el"] and
                candidate["paper_family_id"] == row["paper_family_id"], "An excluded pair has a different identity.")
        excluded_ids.add(row["observation_id_uts"])
    retained = [r for r in frozen if r["observation_id_uts"] not in excluded_ids]
    require(len(retained) == expected["retained_candidates"], "Retained candidate count differs.")
    source_rows = defaultdict(list)
    for row in retained:
        source_rows[row["paper_family_id"]].append(row)
    primary = {source: group for source, group in source_rows.items() if len(group) >= 2}
    primary_ids = {r["observation_id_uts"] for group in primary.values() for r in group}
    require(len(primary) == expected["primary_sources"], "Primary source count differs.")
    require(len(primary_ids) == expected["primary_candidates"], "Primary candidate count differs.")

    inventory = rows(root, "paper01/analysis/predictive_utility/source_inventory.csv")
    marked = {r["paper_family_id"]: r for r in inventory if r["primary_cohort"] == "True"}
    require(set(marked) == set(primary), "Saved source membership differs from endpoint qualification.")
    strata = Counter()
    for source, group in primary.items():
        hits = sum(float(r["uts_mpa"]) >= 1000 and float(r["elong_pct"]) >= 10 for r in group)
        require(int(marked[source]["n_candidates"]) == len(group), "Source candidate count differs: " + source)
        require(int(marked[source]["observed_hit_count"]) == hits, "Source hit count differs: " + source)
        strata["no_hit_sources" if hits == 0 else "all_hit_sources" if hits == len(group) else "mixed_hit_sources"] += 1
    for name, value in strata.items():
        require(value == expected[name], "Hit-status count differs: " + name)

    blocks = rows(root, "paper01/analysis/source_aware_ranking/representation_blocks_label_informed.csv")
    require(len(blocks) == expected["input_equivalence_blocks"], "Saved block count differs.")
    members = []
    for block in blocks:
        ids = block["observation_ids_uts"].split(";")
        require(len(ids) == int(block["block_size"]), "Block size differs.")
        require(all(by_id[i]["paper_family_id"] == block["paper_family_id"] for i in ids), "Block crosses sources.")
        members.extend(ids)
    require(len(set(members)) == len(members) and set(members) == primary_ids,
            "Saved representation blocks do not partition the primary cohort.")

    candidates = rows(root, "paper01/analysis/source_aware_ranking/candidate_inputs.csv")
    require({r["observation_id_uts"] for r in candidates} == set(by_id), "Feature rows differ from the frozen predictions.")
    for candidate in candidates:
        reference = by_id[candidate["observation_id_uts"]]
        for field in ("observation_id_el", "paper_family_id"):
            require(candidate[field] == reference[field], "Feature identity differs: " + field)
        candidate_fold, reference_fold = float(candidate["fold"]), float(reference["fold"])
        require(candidate_fold == reference_fold and candidate_fold in range(5), "Feature fold differs.")
    try:
        tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"], check=True,
                                 capture_output=True, text=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        tracked = []
    extras = [name for name in tracked if (root / name).is_file() and name not in known | {"EXPORT_MANIFEST.json"}]
    require(not extras, "Files outside the release manifest: " + ", ".join(extras))
    return {"status": "passed", "files": len(known), "bytes": total_bytes,
            "frozen_candidates": len(frozen), "retained_candidates": len(retained),
            "primary_candidates": len(primary_ids), "primary_sources": len(primary),
            "saved_representation_blocks": len(blocks), **dict(strata),
            "numerical_reproduction": "Run tools/reproduce_analysis.py to recompute results from the features and saved predictions."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
