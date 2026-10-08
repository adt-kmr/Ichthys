"""End-to-end test of the job runner against the real evaluate.py CLI.

evaluate.py only needs numpy + motmetrics, so this exercises viz.jobs without
requiring a torch install.

Run:  python scripts/check_jobs.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from viz import jobs

ROOT = REPO_ROOT / "demo_workspace"
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
	status = "ok  " if condition else "FAIL"
	print(f"[{status}] {label}" + (f"  ({detail})" if detail else ""))
	if not condition:
		failures.append(label)


def wait_for(job, timeout: float = 180.0):
	deadline = time.time() + timeout
	while job.is_running and time.time() < deadline:
		time.sleep(0.3)
	return job.returncode


def main() -> int:
	if not ROOT.exists():
		print(f"demo workspace missing at {ROOT}; run scripts/make_demo_workspace.py first")
		return 1

	# --- argv construction ----------------------------------------------------
	argv = jobs.build_argv(
		"evaluate",
		{"pred": "outputs/test1.txt", "gt": "gt.csv", "extra_args": ["--match_pred_range"]},
	)
	check("evaluate argv shape",
		argv[1].endswith("evaluate.py") and "outputs/test1.txt" in argv and "--match_pred_range" in argv,
		" ".join(Path(a).name if a.endswith(".py") else a for a in argv))

	for name, spec in jobs.JOB_COMMANDS.items():
		try:
			jobs.build_argv(name, {})
			check(f"{name} rejects missing required args", False, "no ValueError")
		except ValueError:
			check(f"{name} rejects missing required args", True)

	try:
		jobs.build_argv("nonsense", {})
		check("unknown job rejected", False, "no ValueError")
	except ValueError:
		check("unknown job rejected", True)

	# --- run the real evaluate job -------------------------------------------
	pred = ROOT / "outputs" / "test1.txt"
	gt = ROOT / "synfish" / "test" / "gt-traj" / "test1-gt.csv"
	job = jobs.start_job(
		"evaluate",
		{"pred": str(pred), "gt": str(gt), "extra_args": ["--match_pred_range"]},
		ROOT,
	)
	check("job started", job.error is None and job.process is not None,
		job.error or f"pid {job.process.pid if job.process else '?'}")
	check("job log created", job.log_path.is_file(), job.log_path.name)

	code = wait_for(job)
	check("evaluate job exited cleanly", code == 0, f"exit {code}")

	output = "\n".join(job.tail(200))
	lowered = output.lower()
	check("job printed the MOT summary", "mota" in lowered and "idf1" in lowered,
		f"{len(output.splitlines())} lines")
	check("job summary has a scene row", "scene" in lowered)

	# The dashboard scores in-process; it must agree with this CLI output.
	from viz.metrics import compute_mot_report, format_summary

	report = compute_mot_report(pred, gt, match_pred_range=True)
	display = dict(zip(format_summary(report)["metric"], format_summary(report)["display"]))
	for metric in ("MOTA", "IDF1", "Recall", "Precision", "MOTP (mean dist)", "IDSW", "FP", "FN"):
		column = {
			"MOTA": "mota", "IDF1": "idf1", "Recall": "recall", "Precision": "precision",
			"MOTP (mean dist)": "motp", "IDSW": "num_switches",
			"FP": "num_false_positives", "FN": "num_misses",
		}[metric]
		check(f"dashboard reports {metric}", metric in display, display.get(metric, "missing"))
		# The CLI table prints mota/idf1/recall/precision as percentages, and the
		# dashboard must render the identical string.
		if metric in {"MOTA", "IDF1", "Recall", "Precision"}:
			expected = f"{float(report.summary.iloc[0][column]) * 100:.1f}%"
			check(f"{metric} matches CLI precision format", display[metric] == expected,
				f"dashboard {display[metric]} vs {expected}")
		else:
			expected = f"{float(report.summary.iloc[0][column]):.3f}" if metric == "MOTP (mean dist)" else f"{float(report.summary.iloc[0][column]):.0f}"
			check(f"{metric} matches CLI value", display[metric] == expected,
				f"dashboard {display[metric]} vs {expected}")
	print("\n--- job log ---")
	print("\n".join(output.splitlines()[:14]))
	print("--- end job log ---\n")

	check("job status reports finished", job.status == "finished", job.status)
	check("job duration is sane", 0 < job.duration < 180, f"{job.duration:.1f}s")

	# --- registry -------------------------------------------------------------
	listed = jobs.list_jobs()
	check("job is listed", any(j.log_path == job.log_path for j in listed), f"{len(listed)} job(s)")

	# --- failing job surfaces its exit code -----------------------------------
	bad = jobs.start_job("evaluate", {"pred": str(pred), "gt": str(ROOT / "missing.csv")}, ROOT)
	bad_code = wait_for(bad)
	check("bad job exits non-zero", bad_code not in (0, None), f"exit {bad_code}")
	check("bad job status reports failure", bad.status.startswith("exit"), bad.status)

	# --- stop a long-running job ---------------------------------------------
	slow = jobs.start_job(
		"evaluate",
		{"pred": str(pred), "gt": str(gt), "extra_args": ["--match_pred_range"]},
		ROOT,
	)
	jobs.stop_job(slow.log_path.stem)
	stopped_code = wait_for(slow, timeout=30)
	check("stopped job terminates", not slow.is_running, f"exit {stopped_code}")

	print()
	if failures:
		print(f"{len(failures)} check(s) FAILED:")
		for name in failures:
			print(f"  - {name}")
		return 1
	print("all job-runner checks passed")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
