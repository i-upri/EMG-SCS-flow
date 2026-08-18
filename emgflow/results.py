"""Read what the pipeline actually wrote.

The GUI never recomputes detections — it renders the pipeline's own outputs, so what you
see is what the scripted run produces. Folder names here mirror io_utils.build_output_dirs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src.neurosoft import CONDITION, SCENARIOS

SIR_DIR = "Stimulation-induced responses"
SS_DIR = "StartStop analysis"
COND_DIR = "Condition test"


def mode_dir(root: Path, folder: str) -> Path:
    """Where *folder*-mode results live in this output root.

    A run that produced only one analysis writes straight into ``results/``; the
    ``<mode>/`` level is kept only when a root holds more than one. Prefer the
    nested path when it exists so both layouts read the same way.
    """
    nested = Path(root) / "results" / folder
    return nested if nested.exists() else Path(root) / "results"

METRIC_COLS = [
    "Configuration", "Stim. amplitude", "Epoch", "Channel",
    "Onset latency", "Peak1 latency", "Peak2 latency",
    "Peak1 value", "Peak2 value", "PTP amplitude",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=METRIC_COLS)
    # Runs made before mid-2026 carry a "Time series" column: an attempt at storing each
    # epoch's waveform inline, which numpy's repr silently truncated to seven numbers and an
    # ellipsis, so it was never usable data — only bulk, up to 120 MB a file. It is no longer
    # written, and skipped at PARSE time (not after) for the runs that still have it.
    # Waveforms live in the epoch .fif files.
    try:
        return pd.read_csv(path, usecols=lambda c: c != "Time series")
    except ValueError:
        return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Finding already-processed recordings on disk
# --------------------------------------------------------------------------- #
@dataclass
class ProcessedRun:
    """One output root that already holds results.

    Deliberately cheap to build: no metrics CSV is read here. Older ones run to 50-120 MB
    (the dead 'Time series' column), and reading one per run would make a scan take minutes.
    """
    root: Path
    mode: str            # "sir" | "startstop" | "condition"
    name: str
    n_items: int         # crops (SIR) or saved conditions (StartStop)
    has_spontaneous: bool
    mtime: float
    size_mb: float

    @property
    def mode_label(self) -> str:
        return {"sir": "Stimulation-induced", "condition": "Condition test"}.get(
            self.mode, "StartStop")

    @property
    def items_label(self) -> str:
        return {"sir": "crops", "condition": "ISI"}.get(self.mode, "conditions")


#: Which mode a recorded Neurosoft scenario belongs to. Built off ``SCENARIOS``
#: so a scenario added there cannot be forgotten here; only the Condition test
#: writes its own output tree, the rest are SIR deliverables.
_MODE_OF_SCENARIO = {s: ("condition" if s == CONDITION else "sir") for s in SCENARIOS}


def _recorded_mode(root: Path) -> str | None:
    """The mode the LAST run recorded in ``review/run.json``, or None.

    None covers a run made before the manifest existed, and equally a run that
    carries no scenario at all — a plain SIR or StartStop recording, which the
    manifest cannot tell apart and the content probes can.
    """
    p = Path(root) / "review" / "run.json"
    if not p.exists():
        return None
    try:
        scen = json.loads(p.read_text(encoding="utf-8")).get("scenario")
    except Exception:
        return None
    return _MODE_OF_SCENARIO.get(scen)


def detect_mode(root: Path) -> str | None:
    """Which mode produced this output root — judged by CONTENT, not by folder names.

    Folders are created lazily now (a SIR run no longer leaves an empty
    'StartStop analysis/' behind), but content remains the safer test: older
    output roots on disk still carry both empty trees.

    Content alone cannot say which of two analyses is CURRENT, though, and a
    re-run under a corrected scenario leaves both on disk: the cleanup at the end
    of a run removes the other scenarios' deliverables, not another mode's files,
    and the Condition test is not a SIR scenario. So a file first misclassified
    as a Condition test kept its ``condition_summary.csv`` sitting in a flat
    ``results/`` while the corrected SIR run nested itself under
    ``Stimulation-induced responses/`` beside it — and since Condition was probed
    first, every later open of that folder reopened the wrong, superseded tab.
    The run manifest is rewritten on every run (see ``write_run_manifest``), so
    it settles the order; the probes still decide, and a run that recorded a mode
    but wrote nothing still reads as empty rather than as that mode.
    """
    root = Path(root)

    def has_condition() -> bool:
        cond = mode_dir(root, COND_DIR)
        cond_wf = cond / "Waterfall"
        return (cond / "Excel" / "condition_summary.csv").exists() or (
            cond_wf.exists() and any(cond_wf.glob("*.png")))

    def has_sir() -> bool:
        sir_epochs = mode_dir(root, SIR_DIR) / "Stimulus-centered epochs"
        return sir_epochs.exists() and any(sir_epochs.glob("*-epo.fif"))

    def has_startstop() -> bool:
        ss_raw = mode_dir(root, SS_DIR) / "Detections raw"
        return ss_raw.exists() and any(ss_raw.glob("*.fif"))

    probes = {"condition": has_condition, "sir": has_sir, "startstop": has_startstop}
    # Condition test is a distinct output tree (no SIR epochs, no StartStop fif),
    # so probe it first by its own summary CSV / plots — unless the manifest says
    # this root was last produced by something else.
    order = ["condition", "sir", "startstop"]
    recorded = _recorded_mode(root)
    if recorded is not None:
        order.remove(recorded)
        order.insert(0, recorded)
    for mode in order:
        if probes[mode]():
            return mode

    # Metrics CSV alone: both modes write the same file name, so it can only
    # disambiguate when the mode folder is present.
    if (root / "results" / SIR_DIR / "Excel" / "Large_dataset_emg_response_metrics.csv").exists():
        return "sir"
    if (root / "results" / SS_DIR / "Excel" / "Large_dataset_emg_response_metrics.csv").exists():
        return "startstop"
    return None


# Never descend into these: they are the pipeline's own output internals and can hold
# thousands of PNGs. On a network drive (Google Drive) walking them costs minutes.
_PRUNE = {
    "data", "results", "review", "templates", "Excel", "Boxplots", "Envelopes",
    "Plots", "Raw epochs", "Templates", "Detections raw", "Spontaneous EMG",
    "Stimulus-centered epochs", "Template overlays", "annot_crops_fif", "__pycache__",
    "Condition test", "Waterfall", "Amplitude vs condition", "Recruitment", "arrays",
    "Jendrassik", "Paired stimulation", "Curves per condition", "H-reflex",
    "Plots with grid and markers", "Plots without grid and markers",
    "Plots grouped by amplitude", "Template overlays per amplitude",
}


def scan_processed(base: Path, max_depth: int = 4) -> list[ProcessedRun]:
    """Walk a directory tree and list every output root that already holds results.

    Uses os.scandir (whose is_dir() comes from the directory entry, not a separate stat per
    file) and prunes the pipeline's own output folders — otherwise a run's thousands of
    plot PNGs turn a scan of a Drive folder into minutes of stat calls.
    """
    import os

    base = Path(base)
    found: list[ProcessedRun] = []

    def walk(d: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = list(os.scandir(d))
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            return

        subdirs = [
            e for e in entries
            if e.is_dir(follow_symlinks=False) and not e.name.startswith(".")
        ]
        if any(e.name == "results" for e in subdirs):
            mode = detect_mode(d)
            if mode:
                found.append(_describe(d, mode))
                return  # an output root never contains another one

        for e in subdirs:
            if e.name not in _PRUNE:
                walk(Path(e.path), depth + 1)

    walk(base, 0)
    found.sort(key=lambda r: r.mtime, reverse=True)
    return found


def _describe(root: Path, mode: str) -> ProcessedRun:
    """Summarise a run from directory metadata alone — stat and listings, never a CSV read."""
    if mode == "sir":
        base = mode_dir(root, SIR_DIR)
        items = base / "Stimulus-centered epochs"
        pattern = "*-epo.fif"
        csv = base / "Excel" / "Large_dataset_emg_response_metrics.csv"
    elif mode == "condition":
        # A Condition run has no crops and no detection fif; what it produced is
        # a set of inter-stimulus intervals, counted from its own summary.
        base = mode_dir(root, COND_DIR)
        items = base / "Waterfall"
        pattern = "*_by_artifact.png"
        csv = base / "Excel" / "condition_summary.csv"
    else:
        base = mode_dir(root, SS_DIR)
        items = base / "Detections raw"
        pattern = "*.fif"
        csv = base / "Excel" / "Large_dataset_emg_response_metrics.csv"

    n_items = len(list(items.glob(pattern))) if items.exists() else 0
    if mode == "condition" and csv.exists():
        try:
            n_items = int(pd.read_csv(csv)["Condition (ISI ms)"].nunique())
        except Exception:
            pass

    size_mb, mtime = 0.0, 0.0
    if csv.exists():
        st = csv.stat()
        size_mb, mtime = st.st_size / 1e6, st.st_mtime
    if not mtime:
        mtime = root.stat().st_mtime

    spont = mode_dir(root, SS_DIR) / "Spontaneous EMG"
    return ProcessedRun(
        root=root,
        mode=mode,
        name=root.name,
        n_items=n_items,
        has_spontaneous=spont.is_dir() and next(spont.iterdir(), None) is not None,
        mtime=mtime,
        size_mb=size_mb,
    )


# --------------------------------------------------------------------------- #
# SIR
# --------------------------------------------------------------------------- #
@dataclass
class Crop:
    """One (config, amplitude) block — the unit of work in SIR mode."""
    config: str
    amp: str
    epochs_path: Path

    @property
    def label(self) -> str:
        return f"{self.config} @ {self.amp}"


def _parse_crop_stem(stem: str) -> tuple[str, str] | None:
    """'1+2-_9-epo' -> ('1+2', '9').  Config = everything before the first '-',
    matching the pipeline's own `filename.split('-')[0]` convention. Amplitude labels are
    opaque strings ('2' != '2,0' != '02'), so they are never normalised."""
    name = stem[:-4] if stem.endswith("-epo") else stem
    name = re.sub(r"\(\d+\)$", "", name)  # duplicate-crop suffix, e.g. '..._9(1)'
    if "_" not in name:
        return None
    head, amp = name.rsplit("_", 1)
    config = head.split("-")[0]
    return config, amp


class SIRResults:
    def __init__(self, output_root: Path) -> None:
        self.root = Path(output_root)
        self.results_dir = mode_dir(self.root, SIR_DIR)
        self.metrics = _read_csv(self.results_dir / "Excel" / "Large_dataset_emg_response_metrics.csv")
        self.crops: list[Crop] = []
        epochs_dir = self.results_dir / "Stimulus-centered epochs"
        for f in sorted(epochs_dir.glob("*-epo.fif")):
            parsed = _parse_crop_stem(f.stem)
            if parsed:
                self.crops.append(Crop(parsed[0], parsed[1], f))

    @property
    def ok(self) -> bool:
        return bool(self.crops)

    def raw_path(self) -> Path | None:
        """The preprocessed recording the run was built from — what the raw browser shows."""
        d = self.root / "data" / SIR_DIR
        if not d.exists():
            d = self.root / "data"
        for pattern in ("*_preprocessed_raw.fif", "*_original_raw.fif"):
            hits = sorted(d.glob(pattern))
            if hits:
                return hits[0]
        return None

    def channels(self) -> list[str]:
        if not self.metrics.empty:
            return sorted(self.metrics["Channel"].dropna().unique().tolist())
        return []

    def load_epochs(self, crop: Crop) -> mne.Epochs:
        return mne.read_epochs(crop.epochs_path, preload=True, verbose="ERROR")

    def markers(self, crop: Crop, channel: str) -> pd.DataFrame:
        """Per-epoch onset/P1/P2 for one channel of one crop."""
        if self.metrics.empty:
            return pd.DataFrame(columns=METRIC_COLS)
        m = self.metrics
        sel = (
            (m["Configuration"].astype(str) == crop.config)
            & (m["Stim. amplitude"].astype(str) == crop.amp)
            & (m["Channel"].astype(str) == channel)
        )
        return m[sel]

    def detection_count(self, crop: Crop, session=None) -> int:
        """Detections in a crop. With a session, rejected ones stop counting immediately —
        no re-run needed for the number to reflect what you just marked."""
        if self.metrics.empty:
            return 0
        m = self.metrics
        sel = (
            (m["Configuration"].astype(str) == crop.config)
            & (m["Stim. amplitude"].astype(str) == crop.amp)
        )
        rows = m[sel]
        if session is not None and not rows.empty:
            # A plain list of bools would be read as column labels by pandas 2.x, so the mask
            # has to be an ndarray.
            keep = np.array([
                not self._is_rejected(session, crop.config, crop.amp, str(ch))
                for ch in rows["Channel"]
            ], dtype=bool)
            rows = rows[keep]
        # Either component counts on an H-reflex run — see channel_summary.
        if {"M peak1 latency", "H peak1 latency"} <= set(rows.columns):
            return int(rows[["M peak1 latency", "H peak1 latency"]]
                       .notna().any(axis=1).sum())
        if "Peak1 latency" not in rows.columns:
            return 0
        return int(rows["Peak1 latency"].notna().sum())

    @staticmethod
    def _is_rejected(session, config: str, amp: str, channel: str) -> bool:
        return (
            (config, amp, channel) in session.suppress_keys
            or channel in session.exclude_channels
            or config in session.exclude_configs
        )

    def channel_summary(
        self, crop: Crop, session=None, channels: list[str] | None = None
    ) -> pd.DataFrame:
        """Per-channel view of one crop, for the live table under the plot.

        `channels` (the crop's actual EMG channels) makes silent channels appear too: the
        pipeline only writes metric rows for channels it detected something on, so without
        this the table would simply omit the muscles that stayed quiet.
        """
        m = self.metrics
        sel = (
            (m["Configuration"].astype(str) == crop.config)
            & (m["Stim. amplitude"].astype(str) == crop.amp)
        ) if not m.empty else None
        groups = dict(list(m[sel].groupby("Channel"))) if sel is not None else {}

        names = channels if channels is not None else sorted(groups)
        rows = []
        for ch in names:
            grp = groups.get(ch)
            rejected = session is not None and self._is_rejected(session, crop.config, crop.amp, str(ch))
            if grp is None:
                rows.append({
                    "Channel": str(ch), "Detections": 0, "Epochs": 0,
                    "P1 (ms)": np.nan, "PTP (µV)": np.nan,
                    "Status": "rejected" if rejected else "no response",
                })
                continue
            # An H-reflex run has two response columns, and the single-response
            # one holds the M-wave. Counting only that would report "24
            # detections" on a channel carrying 24 M-waves and 41 reflexes, and
            # call a reflex-only channel silent — the exact confusion the
            # scenario exists to remove. Detections there means either component.
            det_cols = ["Peak1 latency"]
            if "H peak1 latency" in grp.columns and "M peak1 latency" in grp.columns:
                det_cols = ["M peak1 latency", "H peak1 latency"]
            det = 0 if rejected else int(
                grp[det_cols].notna().any(axis=1).sum())
            p1 = grp["Peak1 latency"].to_numpy(dtype=float)
            # Amplitudes are stored in VOLTS; the column header promises µV.
            ptp = grp["PTP amplitude"].to_numpy(dtype=float) * 1e6
            status = "rejected" if rejected else ("detected" if det else "no response")
            if not rejected and session is not None and (crop.config, crop.amp, str(ch)) in session.force_keys:
                status = "whitelisted"
            rows.append({
                "Channel": str(ch),
                "Detections": det,
                "Epochs": int(len(grp)),
                "P1 (ms)": np.nan if rejected or not np.isfinite(p1).any() else 1000 * np.nanmean(p1),
                "PTP (µV)": np.nan if rejected or not np.isfinite(ptp).any() else np.nanmean(ptp),
                "Status": status,
            })
        return pd.DataFrame(rows)

    def recruitment_table(self, session=None) -> pd.DataFrame:
        """Detections per (config, amplitude, channel) — the 'which crops are 0' view.

        With a session, manual rejections are applied here too, so the table and the plots
        always tell the same story.
        """
        if self.metrics.empty:
            return pd.DataFrame()
        m = self.metrics.copy()
        m["detected"] = m["Peak1 latency"].notna()
        if session is not None:
            rejected = np.array([
                self._is_rejected(session, str(c), str(a), str(ch))
                for c, a, ch in zip(m["Configuration"], m["Stim. amplitude"], m["Channel"])
            ], dtype=bool)
            m.loc[rejected, "detected"] = False
            m.loc[rejected, ["Peak1 latency", "PTP amplitude"]] = np.nan
        g = (
            m.groupby(["Configuration", "Stim. amplitude", "Channel"], dropna=False)
            .agg(
                detections=("detected", "sum"),
                epochs=("detected", "size"),
                p1_lat_ms=("Peak1 latency", lambda s: 1000 * np.nanmean(s) if s.notna().any() else np.nan),
                # stored in volts
                ptp_uv=("PTP amplitude", lambda s: 1e6 * np.nanmean(s) if s.notna().any() else np.nan),
            )
            .reset_index()
        )
        return g

    def recruitment_curves(self, config: str, session=None) -> pd.DataFrame:
        """Amplitude -> response size per channel, with a 95 % CI across epochs.

        The x value is the amplitude label parsed as a number (`9,5` -> 9.5); labels that
        are not numeric are dropped from the curve but keep their row in the table. The
        label itself is never rewritten — `2` and `2,0` remain different crops.
        """
        if self.metrics.empty:
            return pd.DataFrame()
        m = self.metrics
        sub = m[m["Configuration"].astype(str) == str(config)].copy()
        if sub.empty:
            return pd.DataFrame()

        if session is not None:
            rejected = np.array([
                self._is_rejected(session, str(config), str(a), str(ch))
                for a, ch in zip(sub["Stim. amplitude"], sub["Channel"])
            ], dtype=bool)
            sub.loc[rejected, ["Peak1 latency", "PTP amplitude"]] = np.nan

        def _amp(label) -> float:
            try:
                return float(str(label).replace(",", "."))
            except ValueError:
                return np.nan

        sub["amp_value"] = [_amp(a) for a in sub["Stim. amplitude"]]
        sub["ptp_uv"] = sub["PTP amplitude"].astype(float) * 1e6  # stored in volts

        rows = []
        for (amp_label, ch), grp in sub.groupby(["Stim. amplitude", "Channel"], dropna=False):
            vals = grp["ptp_uv"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            n = int(vals.size)
            mean = float(np.mean(vals)) if n else 0.0
            # No CI from a single trial; a zero-width band is the honest rendering.
            sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            rows.append({
                "Channel": str(ch),
                "amp_label": str(amp_label),
                "amp_value": float(grp["amp_value"].iloc[0]),
                "n": n,
                "mean_ptp_uv": mean,
                "ci95": 1.96 * sem,
                "values": vals,
            })
        out = pd.DataFrame(rows)
        return out.sort_values(["Channel", "amp_value"], na_position="last")

    def mean_wave(self, crop: Crop, channel: str) -> tuple[np.ndarray, np.ndarray]:
        """Times (s) and the mean epoch waveform (µV) — the curve a template is built from."""
        epochs = self.load_epochs(crop)
        data = epochs.get_data(picks=[channel])[:, 0, :] * 1e6
        return epochs.times, data.mean(axis=0)

    # ------------------------------------------------------------------ #
    # Neurosoft curve exports
    # ------------------------------------------------------------------ #
    @property
    def scenario(self) -> str | None:
        """Which Neurosoft protocol this run produced, or None for a normal SIR run.

        Read off the deliverable folder the pipeline wrote — the scenarios are
        mutually exclusive, so the first hit is the answer.
        """
        for folder in ("H-reflex", "Recruitment", "Jendrassik", "Paired stimulation"):
            if (self.results_dir / folder).is_dir():
                return folder
        return None

    def hreflex_by_curve(self, config: str, session=None) -> pd.DataFrame:
        """Per-curve M and H metrics side by side, for the H-reflex scenario.

        Same row set as ``recruitment_by_curve`` — one per curve, undetected
        ones included — but with a column pair per component, because on these
        files a curve routinely carries one response and not the other and the
        two have to be plotted, marked and corrected apart.
        """
        from src.hreflex import COMPONENTS, columns_for

        if self.metrics.empty:
            return pd.DataFrame()
        sub = self.metrics[self.metrics["Configuration"].astype(str) == str(config)].copy()
        if sub.empty or "Epoch" not in sub.columns:
            return pd.DataFrame()
        if not all(columns_for(c)["p1"] in sub.columns for c in COMPONENTS):
            return pd.DataFrame()

        out = pd.DataFrame({
            "Channel": sub["Channel"].astype(str),
            "curve": sub["Epoch"].astype(int) + 1,
        })
        for comp in COMPONENTS:
            cols = columns_for(comp)
            pre = comp.lower()
            rejected = None
            if session is not None:
                rejected = np.array([
                    self._is_rejected(session, str(config), str(a), str(ch))
                    for a, ch in zip(sub["Stim. amplitude"], sub["Channel"])
                ], dtype=bool)
            ptp = pd.to_numeric(sub[cols["ptp"]], errors="coerce") * 1e6
            p1v = pd.to_numeric(sub[cols["pv1"]], errors="coerce") * 1e6
            if rejected is not None:
                ptp[rejected] = np.nan
                p1v[rejected] = np.nan
            out[f"{pre}_amp_uv"] = ptp
            out[f"{pre}_p1_uv"] = p1v
            for key, name, scale in (("onset", "onset_ms", 1e3), ("p1", "p1_ms", 1e3),
                                     ("p2", "p2_ms", 1e3)):
                v = pd.to_numeric(sub[cols[key]], errors="coerce") * scale
                if rejected is not None:
                    v[rejected] = np.nan
                out[f"{pre}_{name}"] = v
        return out.sort_values(["Channel", "curve"])

    def recruitment_by_curve(self, config: str, session=None) -> pd.DataFrame:
        """Response size per CURVE, for exports with no amplitude axis.

        A Neurosoft file is one crop whose curves ARE the stimulation ramp: the
        amplitude label is the synthetic ``all`` and carries no number, so the
        amplitude-keyed curve cannot be drawn. The curve index is the ramp, and
        plotting against it is the recruitment curve the clinician asks for.

        Amplitude = peak-to-peak when the response is biphasic, |P1| when it is
        monophasic (P2 goes undetected on most of these channels).

        EVERY curve gets a row, including the ones with no detection, whose
        amplitude is NaN. Dropping them would start the curve at whatever
        stimulus first produced a response and hide the sub-threshold run before
        it — but that run is part of the recruitment curve, and where the
        response comes and goes the gaps are the finding.
        """
        if self.metrics.empty:
            return pd.DataFrame()
        sub = self.metrics[self.metrics["Configuration"].astype(str) == str(config)].copy()
        if sub.empty or "Epoch" not in sub.columns:
            return pd.DataFrame()

        if session is not None:
            rejected = np.array([
                self._is_rejected(session, str(config), str(a), str(ch))
                for a, ch in zip(sub["Stim. amplitude"], sub["Channel"])
            ], dtype=bool)
            sub.loc[rejected, ["Peak1 latency", "Peak1 value", "PTP amplitude"]] = np.nan

        ptp = pd.to_numeric(sub["PTP amplitude"], errors="coerce") * 1e6
        p1 = pd.to_numeric(sub["Peak1 value"], errors="coerce") * 1e6
        out = pd.DataFrame({
            "Channel": sub["Channel"].astype(str),
            "curve": sub["Epoch"].astype(int) + 1,      # curve numbers are 1-based
            "amp_uv": ptp.where(ptp.notna(), p1.abs()),
            "p1_uv": p1,
            "p1_ms": pd.to_numeric(sub["Peak1 latency"], errors="coerce") * 1e3,
            "onset_ms": pd.to_numeric(sub.get("Onset latency"), errors="coerce") * 1e3,
            "p2_ms": pd.to_numeric(sub.get("Peak2 latency"), errors="coerce") * 1e3,
        })
        return out.sort_values(["Channel", "curve"])


# --------------------------------------------------------------------------- #
# StartStop
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """One detected response: an annotation on the saved '<cond>_detections_raw.fif'."""
    index: int
    condition: str
    time: float          # seconds from the START of the saved segment (see `detections`)
    abs_time: float      # onset as stored, i.e. in the ORIGINAL recording's time base
    channels: list[str]  # marker description, e.g. 'ECR L+TR R'

    @property
    def label(self) -> str:
        return f"#{self.index + 1}  {self.time:7.3f} s  —  {'+'.join(self.channels)}"


class StartStopResults:
    def __init__(self, output_root: Path) -> None:
        self.root = Path(output_root)
        self.results_dir = mode_dir(self.root, SS_DIR)
        self.metrics = _read_csv(self.results_dir / "Excel" / "Large_dataset_emg_response_metrics.csv")
        self.channel_qc = _read_csv(self.results_dir / "Excel" / "STARTSTOP_channel_qc.csv")
        self.discarded = _read_csv(self.results_dir / "Excel" / "STARTSTOP_template_anchor_discarded.csv")

        self.raw_paths: dict[str, Path] = {}
        for f in sorted((self.results_dir / "Detections raw").glob("*_detections_raw.fif")):
            self.raw_paths[f.stem.replace("_detections_raw", "")] = f
        self._raw_cache: dict[str, mne.io.BaseRaw] = {}

    @property
    def ok(self) -> bool:
        return bool(self.raw_paths)

    def conditions(self) -> list[str]:
        return list(self.raw_paths)

    def raw(self, condition: str) -> mne.io.BaseRaw:
        if condition not in self._raw_cache:
            self._raw_cache[condition] = mne.io.read_raw_fif(
                self.raw_paths[condition], preload=True, verbose="ERROR"
            )
        return self._raw_cache[condition]

    def detections(self, condition: str) -> list[Detection]:
        """Detections with times made relative to the saved segment.

        The saved .fif is the CONCATENATED start segment (e.g. 11 s long), but its
        annotation onsets are kept in the original recording's time base (e.g. 1023 s).
        Subtracting `first_time` is what puts a marker back on the data it belongs to —
        without it every detection lands far past the end of the file.
        """
        raw = self.raw(condition)
        offset = float(raw.first_time)
        out: list[Detection] = []
        for i, ann in enumerate(raw.annotations):
            desc = str(ann["description"])
            abs_t = float(ann["onset"])
            out.append(Detection(i, condition, abs_t - offset, abs_t, desc.split("+")))
        return out

    def why_empty(self, condition: str) -> pd.DataFrame:
        """Rejected template anchors with the pipeline's own reason — the 'why did I find
        nothing here' view, straight from STARTSTOP_template_anchor_discarded.csv."""
        if self.discarded.empty or "Reason" not in self.discarded.columns:
            return pd.DataFrame()
        d = self.discarded
        col = "Configuration" if "Configuration" in d.columns else None
        if col:
            d = d[d[col].astype(str) == condition]
        return d.groupby(["Channel", "Reason"]).size().reset_index(name="n")


# --------------------------------------------------------------------------- #
# Spontaneous EMG (lives inside the StartStop tree)
# --------------------------------------------------------------------------- #
class SpontaneousResults:
    def __init__(self, output_root: Path) -> None:
        self.root = Path(output_root)
        self.base = mode_dir(self.root, SS_DIR) / "Spontaneous EMG"

    @property
    def ok(self) -> bool:
        return self.base.exists() and any(self.base.iterdir())

    def conditions(self) -> list[str]:
        if not self.base.exists():
            return []
        return sorted(d.name for d in self.base.iterdir() if d.is_dir())

    def bursts(self, condition: str) -> pd.DataFrame:
        p = self.base / condition / "Excel" / f"Spontaneous_EMG_bursts_{condition}.csv"
        return _read_csv(p) if p.exists() else pd.DataFrame()

    def summary(self, condition: str) -> pd.DataFrame:
        p = self.base / condition / "Excel" / f"Spontaneous_EMG_summary_{condition}.csv"
        return _read_csv(p) if p.exists() else pd.DataFrame()

    def coactivation_episodes(self, condition: str) -> pd.DataFrame:
        """Antagonist-pair burst episodes behind the L-shape plot (one row per episode)."""
        p = self.base / condition / "Excel" / f"Spontaneous_EMG_coactivation_episodes_{condition}.csv"
        return _read_csv(p) if p.exists() else pd.DataFrame()

    def envelope_files(self, condition: str) -> list[Path]:
        d = self.base / condition / "Envelopes"
        return sorted(d.glob("*.txt")) if d.exists() else []

    def envelopes_on_segment(self, condition: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per-channel RMS envelopes placed back on the segment's own timeline.

        Two kinds of export exist and they use different time bases:
          * `<ch>_fullenvelope_rms.txt`  — already in segment time;
          * `<ch>_burst<k>_rms_envelope.txt` — re-centred on the burst midpoint (that is the
            form the stimulator wants), so it has to be shifted back by that midpoint before
            it can be drawn over the recording.
        """
        bursts = self.bursts(condition)
        out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

        for f in self.envelope_files(condition):
            stem = f.stem
            try:
                t, v = self.read_envelope(f)
            except Exception:
                continue

            if "_fullenvelope" in stem:
                ch = stem.split("_fullenvelope")[0]
                out.setdefault(ch, []).append((t, v))
                continue

            if "_burst" in stem:
                ch = stem.split("_burst")[0]
                try:
                    k = int(stem.split("_burst")[1].split("_")[0])
                except (IndexError, ValueError):
                    continue
                if bursts.empty:
                    continue
                row = bursts[(bursts["Channel"].astype(str) == ch) & (bursts["Burst"] == k)]
                if row.empty:
                    continue
                mid = 0.5 * (float(row.iloc[0]["Start_s"]) + float(row.iloc[0]["End_s"]))
                out.setdefault(ch, []).append((t + mid, v))

        merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for ch, parts in out.items():
            t = np.concatenate([p[0] for p in parts])
            v = np.concatenate([p[1] for p in parts])
            order = np.argsort(t)
            merged[ch] = (t[order], v[order])
        return merged

    @staticmethod
    def read_envelope(path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Tab-delimited: (time_from_burst_center_s | time_s), rms_uV."""
        df = pd.read_csv(path, sep="\t")
        return df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)
