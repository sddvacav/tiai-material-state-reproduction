#!/usr/bin/env python3
"""Replay fixed and sequential selection from archived predictions and inputs."""
from __future__ import annotations

import argparse
from math import comb
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("knn_density", "ridge", "tiai_xgb_surrogate")
POLICIES = ("fixed_ridge", "kernel_residual_greedy",
            "common_ridge_seed_uniform_completion", "fully_uniform")
ATOL, RTOL = 1e-10, 1e-9


def table(root, name):
    return pd.read_csv(root / name, float_precision="round_trip")


def pool(root):
    decisions = table(root, "paper01/source_data/TITMC_PAPER01_source_heldout_decisions.csv")
    qualification = json.loads((root / "SOURCE_QUALIFICATION.json").read_text("utf-8"))
    expected = qualification["reference_values"]
    if expected != json.loads((root / "DATA_INPUTS.json").read_text("utf-8"))["expected"]:
        raise ValueError("Qualification and data-index reference values differ.")
    excluded = {r["observation_id_uts"] for r in qualification["removed_pairs"]}
    admitted = decisions.loc[decisions.admitted.eq(True) & decisions.method.isin(METHODS)].copy()
    retained = admitted.loc[~admitted.observation_id_uts.isin(excluded)].copy()
    reference = retained.loc[retained.method.eq("ridge")].copy()
    sizes = reference.groupby("paper_family_id").size()
    primary_sources = set(sizes.loc[sizes.ge(2)].index)
    retained = retained.loc[retained.paper_family_id.isin(primary_sources)]
    reference = reference.loc[reference.paper_family_id.isin(primary_sources)]
    if len(reference) != expected["primary_candidates"] or len(primary_sources) != expected["primary_sources"]:
        raise ValueError("The supplied endpoint qualification does not define the current cohort.")
    return retained, reference


def probability(n, hits, budget):
    return 1.0 - (comb(n-hits, budget) / comb(n, budget) if n-hits >= budget else 0.0)


def expected_max(values, budget):
    ordered = np.sort(np.asarray(values, dtype=float))
    denominator = comb(len(ordered), budget)
    return float(sum(ordered[i-1] * comb(i-1, budget-1) / denominator
                     for i in range(budget, len(ordered)+1)))


def hypervolume(points):
    x = np.unique(np.r_[0.0, points[:, 0]])
    return float(sum((right-left) * points[points[:, 0] >= right, 1].max()
                     for left, right in zip(x[:-1], x[1:])))


def expected_hypervolume(points, budget):
    x = np.unique(np.r_[0.0, points[:, 0]])
    y = np.unique(np.r_[0.0, points[:, 1]])
    area = np.multiply.outer(np.diff(x), np.diff(y))
    count = ((points[:, 0, None, None] >= x[None, 1:, None]) &
             (points[:, 1, None, None] >= y[None, None, 1:])).sum(axis=0)
    likelihood = np.array([probability(len(points), h, budget) for h in range(len(points)+1)])
    return float(np.sum(area * likelihood[count]))


def compare(actual, reference, keys, numbers, strings=()):
    a = actual.set_index(keys).sort_index()
    b = reference.set_index(keys).sort_index()
    if not a.index.is_unique or not b.index.is_unique or not a.index.equals(b.index):
        raise ValueError("Result identities differ: " + ", ".join(keys))
    maximum = 0.0
    for column in numbers:
        av, bv = a[column].to_numpy(float), b[column].to_numpy(float)
        np.testing.assert_allclose(av, bv, rtol=RTOL, atol=ATOL, equal_nan=True,
                                   err_msg="Different " + column)
        finite = np.isfinite(av) & np.isfinite(bv)
        if finite.any():
            maximum = max(maximum, float(np.max(np.abs(av[finite]-bv[finite]))))
    for column in strings:
        if not a[column].fillna("").astype(str).equals(b[column].fillna("").astype(str)):
            raise ValueError("Result identities differ in " + column)
    return {"rows": len(a), "numeric_columns": len(numbers),
            "maximum_absolute_difference": maximum}


def save(frame, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, float_format="%.17g")


def static_selection(root, output, retained):
    result = []
    references = {s: g for s, g in retained.loc[retained.method.eq("ridge")].groupby("paper_family_id")}
    for (method, source), candidates in retained.groupby(["method", "paper_family_id"], sort=True):
        ordered = candidates.sort_values(["decision_score", "observation_id_uts", "observation_id_el"],
                                         ascending=[False, True, True], kind="stable")
        truth = references[source]
        points = truth[["uts_mpa", "elong_pct"]].to_numpy(float) / [1000.0, 10.0]
        q, n = points.min(axis=1), len(points)
        hits = int((q >= 1).sum())
        volume = hypervolume(points)
        for k in (1, 2, 3, 5, 10):
            budget = min(k, n)
            selected = ordered.iloc[:budget]
            xy = selected[["uts_mpa", "elong_pct"]].to_numpy(float) / [1000.0, 10.0]
            selected_hits = int((xy >= 1).all(axis=1).sum())
            best = float(xy.min(axis=1).max())
            uniform_best = expected_max(q, budget)
            selected_volume = hypervolume(xy)
            uniform_volume = expected_hypervolume(points, budget)
            result.append({
                "paper_family_id": source, "method": method, "requested_k": k,
                "n_candidates": n, "actual_budget": budget,
                "observed_hit_count": hits, "selected_observed_hits": selected_hits,
                "at_least_one_hit": int(selected_hits > 0),
                "uniform_at_least_one_hit": probability(n, hits, budget),
                "selected_position_precision": selected_hits / budget,
                "uniform_position_precision": hits / n,
                "observed_selected_best_q": best, "uniform_expected_best_q": uniform_best,
                "observed_best_regret": float(q.max()-best),
                "uniform_expected_regret": float(q.max()-uniform_best),
                "selected_hypervolume": selected_volume, "uniform_expected_hypervolume": uniform_volume,
                "hypervolume_coverage": selected_volume/volume,
                "uniform_hypervolume_coverage": uniform_volume/volume,
                "selected_observation_ids_uts": ";".join(selected.observation_id_uts),
                "selected_observation_ids_el": ";".join(selected.observation_id_el),
            })
    frame = pd.DataFrame(result)
    saved = table(root, "paper01/analysis/predictive_utility/source_budget_method_utility.csv")
    saved = saved.loc[saved.primary_cohort.eq(True)]
    keys = ["paper_family_id", "method", "requested_k"]
    strings = ["selected_observation_ids_uts", "selected_observation_ids_el"]
    numbers = [c for c in frame if c not in keys + strings]
    check = compare(frame, saved, keys, numbers, strings)
    summary = table(root, "paper01/analysis/predictive_utility/budget_utility_summary.csv")
    rows = []
    metric_columns = {
        "at_least_one_attainment": ("at_least_one_hit", "uniform_at_least_one_hit"),
        "equal_source_precision": ("selected_position_precision", "uniform_position_precision"),
        "observed_best_regret": ("observed_best_regret", "uniform_expected_regret"),
        "hypervolume_coverage": ("hypervolume_coverage", "uniform_hypervolume_coverage"),
    }
    for row in summary.itertuples():
        part = frame.loc[frame.method.eq(row.method) & frame.requested_k.eq(row.requested_k)]
        if row.cohort == "secondary_exact_k":
            part = part.loc[part.n_candidates.ge(max(2, row.requested_k))]
        elif row.cohort == "diagnostic_n_gt_k":
            part = part.loc[part.n_candidates.gt(row.requested_k)]
        elif row.cohort != "primary_source_qualified_min_k_n":
            raise ValueError("Unknown fixed-budget cohort: " + row.cohort)
        if row.metric == "pooled_position_precision":
            actual = float(part.selected_observed_hits.sum()/part.actual_budget.sum())
            uniform = float((part.uniform_position_precision*part.actual_budget).sum()/part.actual_budget.sum())
        else:
            ac, uc = metric_columns[row.metric]
            actual, uniform = float(part[ac].mean()), float(part[uc].mean())
        gain = uniform-actual if row.metric == "observed_best_regret" else actual-uniform
        rows.append(dict(cohort=row.cohort, stratum=row.stratum, method=row.method,
                         requested_k=row.requested_k, metric=row.metric, n_sources=len(part),
                         available_candidates=int(part.n_candidates.sum()),
                         actual_selection_positions=int(part.actual_budget.sum()),
                         observed=actual, uniform_exact_expectation=uniform, paired_gain=gain))
    aggregate = pd.DataFrame(rows)
    aggregate_check = compare(aggregate, summary,
        ["cohort", "stratum", "method", "requested_k", "metric"],
        ["n_sources", "available_candidates", "actual_selection_positions",
         "observed", "uniform_exact_expectation", "paired_gain"])
    save(frame, output/"fixed_budget_by_source.csv")
    save(aggregate, output/"fixed_budget_summary.csv")
    return {"per_source": check, "aggregate": aggregate_check}


def source_aware_selection(root, output, reference):
    predictions = table(root, "paper01/analysis/source_aware_ranking/out_of_source_rankings.csv")
    predictions = predictions.loc[predictions.observation_id_uts.isin(reference.observation_id_uts)]
    truth = reference.set_index("observation_id_uts")
    ordinary, ties = [], []
    for (method, source), group in predictions.groupby(["method", "paper_family_id"], sort=True):
        group = group.sort_values(["ranking_score", "observation_id_uts", "observation_id_el"],
                                  ascending=[False, True, True], kind="stable")
        labels = truth.loc[group.observation_id_uts]
        q = labels[["uts_mpa", "elong_pct"]].to_numpy(float) / [1000.0, 10.0]
        q = q.min(axis=1)
        hits = q >= 1
        tied = group.ranking_score.to_numpy() == group.ranking_score.iloc[0]
        best, selected, uniform = float(q.max()), float(q[0]), float(q.mean())
        attainment, uniform_hit = float(hits[0]), float(hits.mean())
        tie_hit, tie_q = float(hits[tied].mean()), float(q[tied].mean())
        ordinary.append(dict(paper_family_id=source, method=method, n_candidates=len(group),
            observed_joint_hits=int(hits.sum()), attainment=attainment, uniform_attainment=uniform_hit,
            observed_best_q=best, selected_q=selected, uniform_expected_selected_q=uniform,
            observed_best_regret=best-selected, uniform_expected_regret=best-uniform,
            attainment_gain=attainment-uniform_hit, regret_gain=selected-uniform,
            selected_observation_id_uts=group.observation_id_uts.iloc[0]))
        ties.append(dict(paper_family_id=source, method=method, n_candidates=len(group),
            maximum_score_tie_size=int(tied.sum()), exact_max_score_tie=bool(tied.sum() > 1),
            stable_ID_attainment=attainment, tie_average_attainment=tie_hit, uniform_attainment=uniform_hit,
            stable_ID_regret=best-selected, tie_average_regret=best-tie_q, uniform_expected_regret=best-uniform,
            tie_average_attainment_gain=tie_hit-uniform_hit, tie_average_regret_gain=tie_q-uniform,
            stable_ID_minus_tie_attainment=attainment-tie_hit, stable_ID_minus_tie_selected_q=selected-tie_q,
            maximum_score_observation_ids_uts=";".join(group.loc[tied, "observation_id_uts"])))
    checks = {}
    for name, entries, strings in [
        ("source_top1_utility", ordinary, ["selected_observation_id_uts"]),
        ("exact_tie_source_utility", ties, ["maximum_score_observation_ids_uts"])]:
        frame = pd.DataFrame(entries)
        keys = ["paper_family_id", "method"]
        numbers = [c for c in frame if c not in keys + strings]
        checks[name] = compare(frame, table(root, "paper01/analysis/source_aware_ranking/"+name+".csv"),
                               keys, numbers, strings)
        save(frame, output/(name+".csv"))
    return checks


def sequential_selection(root, output, reference):
    scales = table(root, "paper01/analysis/sequential_design/outer_training_kernel_scales.csv").set_index("fold")
    matrices = {}
    for fold in range(5):
        path = root/f"paper01/analysis/source_aware_ranking/encoded_inputs_fold{fold}.npz"
        with np.load(path, allow_pickle=False) as archive:
            matrices[fold] = (archive["X"].copy(),
                             {str(oid): i for i, oid in enumerate(archive["observation_id_uts"])})
    result = []
    for source, group in reference.groupby("paper_family_id", sort=True):
        group = group.sort_values(["observation_id_uts", "observation_id_el"], kind="stable")
        fold, n = int(group.fold.iloc[0]), len(group)
        matrix, positions = matrices[fold]
        ids, el_ids = group.observation_id_uts.tolist(), group.observation_id_el.tolist()
        X = matrix[[positions[oid] for oid in ids]]
        scale = float(scales.loc[fold, "lengthscale_median_euclidean"])
        delta = X[:, None, :] - X[None, :, :]
        kernel = np.exp(-np.square(delta).sum(axis=2)/(2*scale**2))
        np.fill_diagonal(kernel, 1.0)
        base = group.decision_score.to_numpy(float)
        q = (group[["uts_mpa", "elong_pct"]].to_numpy(float)/[1000.0, 10.0]).min(axis=1)
        fixed = sorted(range(n), key=lambda i: (-float(base[i]), ids[i], el_ids[i]))
        seed = fixed[0]
        adaptive = [seed]
        while len(adaptive) < min(5, n):
            # Only outcomes of the candidates already selected enter the update.
            coefficients = np.linalg.solve(kernel[np.ix_(adaptive, adaptive)]+np.eye(len(adaptive)),
                                           q[adaptive]-base[adaptive])
            # A rowwise reduction gives identical input rows identical updates.
            score = base + np.sum(kernel[:, adaptive] * coefficients[None, :], axis=1)
            chosen = min((i for i in range(n) if i not in adaptive),
                         key=lambda i: (-float(score[i]), ids[i], el_ids[i]))
            adaptive.append(chosen)
        sequences = {"fixed_ridge": fixed, "kernel_residual_greedy": adaptive}
        for k in (1, 2, 3, 5):
            budget = min(k, n)
            common = dict(paper_family_id=source, fold=fold, requested_k=k, actual_budget=budget,
                          n_candidates=n, choice_remaining=n > k,
                          observed_hit_count_diagnostic_only=int((q >= 1).sum()),
                          recorded_best_q=float(q.max()), seed_observation_id_uts=ids[seed],
                          seed_q_evaluation_only=float(q[seed]))
            for policy in POLICIES:
                if policy in sequences:
                    selected = sequences[policy][:budget]
                    values = q[selected]
                    best, hits = float(values.max()), float((values >= 1).sum())
                    attainment, precision = float((values >= 1).any()), hits/budget
                    selected_ids = ";".join(ids[i] for i in selected)
                elif policy == "common_ridge_seed_uniform_completion":
                    rest = np.delete(q, seed)
                    best = float(q[seed]) if budget == 1 else expected_max(np.maximum(q[seed], rest), budget-1)
                    hits = float(q[seed] >= 1)+(budget-1)*float((rest >= 1).mean())
                    attainment = (float(q[seed] >= 1) if budget == 1 or q[seed] >= 1
                                  else probability(n-1, int((rest >= 1).sum()), budget-1))
                    precision = hits/budget
                    selected_ids = "EXACT_EXPECTATION; common_seed="+ids[seed]
                else:
                    best = expected_max(q, budget)
                    attainment = probability(n, int((q >= 1).sum()), budget)
                    precision = float((q >= 1).mean())
                    hits = precision*budget
                    selected_ids = "EXACT_EXPECTATION; no_selected_subset_simulated"
                result.append(dict(common, strategy=policy, selected_observation_ids_uts=selected_ids,
                    attainment=attainment, precision=precision, regret=float(q.max()-best),
                    selected_hits=hits, selected_best_q=best))
    frame = pd.DataFrame(result)
    saved = table(root, "paper01/analysis/sequential_design/source_budget_outcomes.csv")
    keys = ["paper_family_id", "requested_k", "strategy"]
    strings = ["selected_observation_ids_uts", "seed_observation_id_uts"]
    numbers = [c for c in frame if c not in keys + strings]
    save(frame, output/"sequential_by_source.csv")
    check = compare(frame, saved, keys, numbers, strings)
    summary = table(root, "paper01/analysis/sequential_design/budget_outcome_summary.csv")
    aggregates = []
    for row in summary.itertuples():
        part = frame.loc[frame.requested_k.eq(row.requested_k) & frame.strategy.eq(row.strategy)]
        if row.cohort == "sensitivity_n_gt_k":
            part = part.loc[part.n_candidates.gt(row.requested_k)]
        elif row.cohort != "primary_source_qualified":
            raise ValueError("Unknown sequential cohort: "+row.cohort)
        aggregates.append(dict(cohort=row.cohort, requested_k=row.requested_k, strategy=row.strategy,
            metric=row.metric, n_sources=len(part), actual_selection_positions=int(part.actual_budget.sum()),
            estimate=float(part[row.metric].mean())))
    aggregate = pd.DataFrame(aggregates)
    aggregate_check = compare(aggregate, summary, ["cohort", "requested_k", "strategy", "metric"],
                              ["n_sources", "actual_selection_positions", "estimate"])
    paired_reference = table(root, "paper01/analysis/sequential_design/paired_policy_gains.csv")
    pairs = []
    for row in paired_reference.itertuples():
        part = frame.loc[frame.requested_k.eq(row.requested_k)]
        if row.cohort == "sensitivity_n_gt_k":
            part = part.loc[part.n_candidates.gt(row.requested_k)]
        elif row.cohort != "primary_source_qualified":
            raise ValueError("Unknown paired cohort: "+row.cohort)
        adaptive = part.loc[part.strategy.eq("kernel_residual_greedy")].set_index("paper_family_id").sort_index()
        comparison = part.loc[part.strategy.eq(row.comparator)].set_index("paper_family_id").sort_index()
        if not adaptive.index.equals(comparison.index):
            raise ValueError("The policy comparison crosses sources.")
        values = adaptive[row.metric]-comparison[row.metric]
        if row.metric == "regret":
            values = -values
        pairs.append(dict(cohort=row.cohort, requested_k=row.requested_k, comparator=row.comparator,
            metric=row.metric, n_sources=len(values), actual_selection_positions=int(adaptive.actual_budget.sum()),
            paired_gain=float(values.mean())))
    paired = pd.DataFrame(pairs)
    paired_check = compare(paired, paired_reference, ["cohort", "requested_k", "comparator", "metric"],
                           ["n_sources", "actual_selection_positions", "paired_gain"])
    save(frame, output/"sequential_by_source.csv")
    save(aggregate, output/"sequential_summary.csv")
    save(paired, output/"paired_policy_gains.csv")
    return {"per_source": check, "aggregate": aggregate_check, "paired": paired_check}


def run(root: Path, output: Path, analysis: str = "all") -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or any(output.is_relative_to(root/p) for p in ("paper01", "tools", ".git")):
        raise ValueError("Choose a separate results directory.")
    output.mkdir(parents=True, exist_ok=True)
    retained, reference = pool(root)
    report = {"backend": "CPU; NumPy float64", "model_fits": 0, "primary_candidates": len(reference),
              "primary_sources": reference.paper_family_id.nunique(),
              "tolerance": {"absolute": ATOL, "relative": RTOL},
              "bootstrap_intervals_recomputed": False}
    if analysis in ("utility", "all"):
        report["fixed_budget"] = static_selection(root, output/"utility", retained)
        report["source_aware_ranking"] = source_aware_selection(root, output/"utility", reference)
    if analysis in ("sequential", "all"):
        report["sequential"] = sequential_selection(root, output/"sequential", reference)
    report["status"] = "passed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--analysis", choices=("utility", "sequential", "all"), default="all")
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output, args.analysis), indent=2))
