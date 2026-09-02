"""Seeding, device resolution, logging and checkpoint helpers for MAPPO."""

from __future__ import annotations

import csv
import os
import random
import subprocess
import sys
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def resolve_device(pref: str) -> torch.device:
    """cpu | cuda | mps | auto. ``auto`` picks cuda if present, else cpu.

    MPS is never auto-selected: for the tiny MLPs here the host<->device transfer
    overhead makes it slower than CPU. Pass ``--device mps`` explicitly to try it.
    """
    pref = pref.lower()
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def sumo_version() -> str:
    try:
        import sumolib

        return subprocess.check_output([sumolib.checkBinary("sumo"), "--version"], stderr=subprocess.DEVNULL).decode().splitlines()[0].strip()
    except Exception:
        return "unknown"


def lib_versions() -> dict:
    def _v(name):
        try:
            return __import__(name).__version__
        except Exception:
            return "n/a"

    return {
        "python": sys.version.split()[0],
        "torch": _v("torch"),
        "numpy": _v("numpy"),
        "gymnasium": _v("gymnasium"),
        "pettingzoo": _v("pettingzoo"),
        "sumo_rl": _v("sumo_rl"),
        "sumo": sumo_version(),
    }


class CsvLogger:
    """Append-only CSV logger. Column set is fixed from the first row."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.header: Optional[list[str]] = None

    def log(self, row: dict) -> None:
        first = not os.path.exists(self.path)
        if self.header is None:
            self.header = list(row.keys())
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.header, extrasaction="ignore")
            if first:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in self.header})


class TBWriter:
    """Thin optional wrapper around tensorboard's SummaryWriter (no-op if unavailable)."""

    def __init__(self, logdir: str):
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(logdir)
        except Exception:
            pass

    def add(self, tag: str, value, step: int) -> None:
        if self.writer is None or value is None:
            return
        try:
            self.writer.add_scalar(tag, float(value), step)
        except Exception:
            pass

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def save_checkpoint(path: str, mappo, iteration: int, global_step: int, cfg, g_rms=None, extra: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "mappo": mappo.state_dict(),
        "iteration": iteration,
        "global_step": global_step,
        "config": cfg.to_dict(),
        "g_rms": g_rms.state_dict() if g_rms is not None else None,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
