"""Run app.py through Streamlit's AppTest and assert every tab renders.

This executes the real Streamlit script (not just the helpers), so it catches
widget-key collisions, exceptions inside render functions and bad layout calls.

Run:  python scripts/check_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from streamlit.testing.v1 import AppTest

DEMO_ROOT = REPO_ROOT / "demo_workspace"
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
	status = "ok  " if condition else "FAIL"
	print(f"[{status}] {label}" + (f"  ({detail})" if detail else ""))
	if not condition:
		failures.append(label)


def report_exceptions(at: AppTest, stage: str) -> None:
	for item in at.exception:
		check(f"no exception during {stage}", False, str(item.value))


def main() -> int:
	if not DEMO_ROOT.exists():
		print(f"demo workspace missing at {DEMO_ROOT}; run scripts/make_demo_workspace.py first")
		return 1

	at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=180)
	at.run()

	report_exceptions(at, "initial run")
	check("initial run has no exceptions", not at.exception, f"{len(at.exception)} exception(s)")
	if at.exception:
		for item in at.exception:
			print(item.value)
		return 1

	check("title rendered", any("Ichthys" in (title.value or "") for title in at.title))
	check("seven tabs", len(at.tabs) == 7, f"{len(at.tabs)} tabs")
	tab_labels = [tab.label for tab in at.tabs]
	expected = ["Overview", "3D trajectories", "Tracks", "Metrics", "Scene data", "Training", "Jobs"]
	check("tab labels", tab_labels == expected, str(tab_labels))

	# --- sidebar wiring -------------------------------------------------------
	selectboxes = {sb.label: sb for sb in at.selectbox}
	check("prediction selector present", "Prediction file" in selectboxes, str(sorted(selectboxes)))
	check("gt selector present", "GT file" in selectboxes, str(sorted(selectboxes)))
	check("scene overlay selector present", "Scene JSON" in selectboxes, str(sorted(selectboxes)))

	check("metrics rendered", len(at.metric) >= 8, f"{len(at.metric)} metric tiles")
	labels_seen = [m.label for m in at.metric]
	for wanted in ("Predictions", "Ground truth", "Scene JSONs", "Checkpoints", "Tracks", "MOTA", "IDF1", "IDSW"):
		check(f"metric tile {wanted!r}", wanted in labels_seen, str(labels_seen))

	check("dataframes rendered", len(at.dataframe) >= 4, f"{len(at.dataframe)} dataframes")
	check("plotly charts rendered", len(at.get("plotly_chart")) >= 10, f"{len(at.get('plotly_chart'))} charts")

	# --- error messages -------------------------------------------------------
	errors = [e.value for e in at.error]
	check("no Streamlit errors", not errors, "; ".join(errors)[:300])
	warnings = [w.value for w in at.warning]
	check("no Streamlit warnings", not warnings, "; ".join(warnings)[:300])

	# --- interaction: change the selected prediction file ---------------------
	if len(workspace_predictions := [sb for sb in at.selectbox if sb.label == "Prediction file"]):
		sb = workspace_predictions[0]
		if len(sb.options) > 1:
			sb.select(sb.options[-1]).run()
			report_exceptions(at, "prediction switch")
			check("prediction switch keeps working", not at.exception)
			sb.select(sb.options[0]).run()

	# --- interaction: enable client-side animation ----------------------------
	animate = next((cb for cb in at.checkbox if "Animate" in (cb.label or "")), None)
	check("animate toggle present", animate is not None, "found" if animate else "missing")
	if animate is not None:
		animate.check().run()
		report_exceptions(at, "animation enabled")
		check("animation renders without error", not at.exception, f"{len(at.exception)} exception(s)")
		check("no errors while animating", not at.error, "; ".join(e.value for e in at.error)[:200])
		animate.uncheck().run()
		report_exceptions(at, "animation disabled")
		check("animation toggles off cleanly", not at.exception)

	# --- interaction: exercise the job form ------------------------------------
	check("job form widgets present", any(sb.label == "Job" for sb in at.selectbox),
		"job selector found")
	check("extra CLI flags input present",
		any(t.label == "Extra CLI flags" for t in at.text_input),
		"extra flags field found")

	# --- interaction: switching workspace root to a bad path shows an error ----
	root_input = next((t for t in at.text_input if t.label == "Workspace root"), None)
	check("workspace root input present", root_input is not None)
	if root_input is not None:
		root_input.set_value(str(REPO_ROOT / "definitely-missing")).run()
		check("bad root shows an error", bool(at.error) or bool(at.exception), f"{len(at.error)} error(s)")

	# --- rerun stability ------------------------------------------------------
	fresh = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=180)
	fresh.run()
	check("second cold run is clean", not fresh.exception, f"{len(fresh.exception)} exception(s)")

	print()
	if failures:
		print(f"{len(failures)} check(s) FAILED:")
		for name in failures:
			print(f"  - {name}")
		return 1
	print("all app checks passed")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
