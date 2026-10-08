"""Background job runner so the dashboard can show a run as it happens.

Jobs are plain subprocesses running this repository's own CLIs. They are tracked
in a module-level registry (Streamlit reruns the script on every interaction, so
process handles cannot live in ``st.session_state``) and their output is streamed
to a log file under ``<root>/.ichthys_jobs/`` that you can also tail with
``tail -f``.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# name -> spec. "needs" lists required arguments in order; entries of "flags" are
# passed as `--name value` instead of positionally, because evaluate.py and
# sort3d.py take named options while infer.py and train.py take positionals.
JOB_COMMANDS: Dict[str, Dict[str, object]] = {
	"infer": {
		"script": "infer.py",
		"summary": "Run 3D tracking inference on one scene",
		"needs": ["ckpt", "scene_json", "images_root"],
		"flags": {},
	},
	"evaluate": {
		"script": "evaluate.py",
		"summary": "Score a prediction file against ground truth",
		"needs": ["pred", "gt"],
		"flags": {"pred": "--pred", "gt": "--gt"},
	},
	"train": {
		"script": "train.py",
		"summary": "Train the association model",
		"needs": ["train_folder"],
		"flags": {},
	},
	"sort3d": {
		"script": "sort3d.py",
		"summary": "Run the 3D-SORT baseline over a detection stream",
		"needs": ["input", "output"],
		"flags": {"input": "--input", "output": "--output"},
	},
}


@dataclass
class Job:
	"""One launched subprocess plus its log."""

	name: str
	argv: List[str]
	log_path: Path
	process: Optional[subprocess.Popen] = None
	started_at: float = field(default_factory=time.time)
	finished_at: Optional[float] = None
	returncode: Optional[int] = None
	error: Optional[str] = None

	@property
	def is_running(self) -> bool:
		return self.process is not None and self.process.poll() is None

	@property
	def duration(self) -> float:
		return (self.finished_at or time.time()) - self.started_at

	@property
	def status(self) -> str:
		if self.error:
			return "failed to start"
		if self.is_running:
			return "running"
		if self.returncode == 0:
			return "finished"
		return f"exit {self.returncode}"

	def tail(self, max_lines: int = 400) -> List[str]:
		"""Last ``max_lines`` lines of the job log."""
		if not self.log_path.is_file():
			return []
		with self.log_path.open("r", encoding="utf-8", errors="replace") as handle:
			lines = handle.readlines()
		return [line.rstrip("\n") for line in lines[-max_lines:]]


# Registry survives Streamlit reruns within the same process.
_JOBS: Dict[str, Job] = {}


def jobs_dir(root: Path | str) -> Path:
	"""Directory holding job logs for a given workspace root."""
	directory = Path(root).expanduser() / ".ichthys_jobs"
	directory.mkdir(parents=True, exist_ok=True)
	return directory


def list_jobs() -> List[Job]:
	"""All known jobs, newest first."""
	return sorted(_JOBS.values(), key=lambda job: job.started_at, reverse=True)


def get_job(name: str) -> Optional[Job]:
	return _JOBS.get(name)


def build_argv(name: str, arguments: Dict[str, object]) -> List[str]:
	"""Assemble the CLI argv for a job from the form values."""
	if name not in JOB_COMMANDS:
		raise ValueError(f"Unknown job {name!r}; expected one of {sorted(JOB_COMMANDS)}")
	spec = JOB_COMMANDS[name]
	flags: Dict[str, str] = spec["flags"]  # type: ignore[assignment]
	argv: List[str] = [sys.executable, str(_REPOSITORY_ROOT / str(spec["script"]))]

	for key in spec["needs"]:  # type: ignore[union-attr]
		value = arguments.get(key)
		if value in (None, ""):
			raise ValueError(f"{key.replace('_', ' ')} is required")
		if key in flags:
			argv.extend([flags[key], str(value)])
		else:
			argv.append(str(value))

	argv.extend(arguments.get("extra_args", []) or [])
	return argv


def start_job(name: str, arguments: Dict[str, object], root: Path | str) -> Job:
	"""Launch a job and begin streaming its output to a log file."""
	if name not in JOB_COMMANDS:
		raise ValueError(f"Unknown job {name!r}; expected one of {sorted(JOB_COMMANDS)}")

	argv = build_argv(name, arguments)
	stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	log_path = jobs_dir(root) / f"{name}-{stamp}.log"

	job = Job(name=name, argv=argv, log_path=log_path)
	try:
		handle = log_path.open("w", encoding="utf-8", buffering=1)
	except OSError as exc:
		job.error = str(exc)
		_JOBS[job.log_path.stem] = job
		return job

	handle.write(f"$ {' '.join(shlex.quote(part) for part in argv)}\n\n")
	try:
		# New process group so we can kill the whole tree (torch spawns workers).
		process = subprocess.Popen(
			argv,
			cwd=str(_REPOSITORY_ROOT),
			stdout=handle,
			stderr=subprocess.STDOUT,
			text=True,
			env={**os.environ, "PYTHONUNBUFFERED": "1"},
			start_new_session=(os.name != "nt"),
			creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
		)
	except Exception as exc:
		job.error = str(exc)
		handle.close()
		_JOBS[job.log_path.stem] = job
		return job

	job.process = process
	_JOBS[job.log_path.stem] = job

	def _watch() -> None:
		try:
			job.returncode = process.wait()
		finally:
			job.finished_at = time.time()
			try:
				handle.close()
			except Exception:
				pass

	threading.Thread(target=_watch, daemon=True).start()
	return job


def stop_job(name: str) -> None:
	"""Terminate a running job and its children."""
	job = _JOBS.get(name)
	if job is None or not job.is_running or job.process is None:
		return
	try:
		if os.name == "nt":
			subprocess.run(
				["taskkill", "/F", "/T", "/PID", str(job.process.pid)],
				check=False, capture_output=True,
			)
		else:
			os.killpg(os.getpgid(job.process.pid), signal.SIGTERM)
	except Exception:
		job.process.terminate()


def clear_finished() -> None:
	"""Drop finished jobs from the registry (their logs stay on disk)."""
	for name in [name for name, job in _JOBS.items() if not job.is_running]:
		del _JOBS[name]
