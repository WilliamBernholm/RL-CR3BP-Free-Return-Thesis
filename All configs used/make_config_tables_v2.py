#!/usr/bin/env python3
"""
make_config_tables.py

Reads PPO run_config .txt files from:

    All configs used/
        TLI/
        MCC/

and generates LaTeX appendix tables:

    generated_tables/
        tli_run_key_table.tex
        mcc_run_key_table.tex
        ppo_tli_config_table.tex
        ppo_mcc_config_table.tex
        all_config_tables.tex
        parsed_config_summary.csv

The tables are designed to show:
- short run labels: TLI-1, TLI-2, ... and MCC-1, MCC-2, ...
- mapping from short labels to original folder/file/run timestamp
- PPO hyperparameters
- important mission/config values
- initial phase angle / spawn angle for TLI runs
- trajectory/library index for MCC runs
- base reward weights
- per-stage training steps, entropy coefficient, and changing reward weights
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------
# User options
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

TLI_DIR_NAMES = ["TLI", "tli", "PPOA", "PPO-A"]
MCC_DIR_NAMES = ["MCC", "mcc", "PPOB", "PPO-B"]

OUT_DIR = ROOT / "generated_tables"

# Optional manual run notes. Edit these if you want clearer labels.
# The keys are matched if the text appears in the filename or timestamp.
MANUAL_NOTES = {
    # Example:
    # "2026-05-24_10-01-16": "MCC case trained from TLI-generated lunar-impact trajectory",
    # "2026-05-16_07-30-12": "TLI run with different initial phase angle",
}

# If you know exactly which MCC run used the lunar impact case, add it here.
# Example:
# LUNAR_IMPACT_MCC_MATCH = "2026-05-24_10-01-16"
LUNAR_IMPACT_MCC_MATCH = ""

# If you know exactly which TLI run used a different initial phase angle, add it here.
# Example:
# DIFFERENT_PHASE_TLI_MATCH = "2026-05-16_07-30-12"
DIFFERENT_PHASE_TLI_MATCH = ""


# ---------------------------------------------------------------------
# Fields shown in tables
# ---------------------------------------------------------------------

PPO_FIELDS = [
    ("gamma", r"$\gamma$"),
    ("gae_lambda", r"$\lambda_{\mathrm{GAE}}$"),
    ("learning_rate", "Learning rate"),
    ("batch_size", "Batch size"),
    ("n_steps", "Rollout steps"),
    ("n_epochs", "PPO epochs"),
    ("clip_range", "Clip range"),
    ("max_grad_norm", "Max gradient norm"),
    ("ent_coef_default", "Default entropy coefficient"),
    ("n_envs", "Number of environments"),
]

TLI_MISSION_FIELDS = [
    ("spawn_theta_min", "Initial phase angle min"),
    ("spawn_theta_max", "Initial phase angle max"),
    ("tli_dv_max_kms", "Max TLI burn magnitude"),
    ("mcc_dv_max_kms", "Max MCC burn magnitude"),
    ("tli_ballistic_trigger_kms", "TLI commit threshold"),
    ("r_moon_flyby", "Lunar flyby radius"),
    ("rp_min", "Return corridor min"),
    ("rp_max", "Return corridor max"),
    ("t_max", "Maximum mission time"),
]

MCC_MISSION_FIELDS = [
    ("ppo_b_fixed_index", "Trajectory source index"),
    ("ppo_b_library_path", "Trajectory library"),
    ("ppo_b_prob_good", "Scenario prob. good"),
    ("ppo_b_prob_savable", "Scenario prob. savable"),
    ("ppo_b_prob_bad", "Scenario prob. bad"),
    ("tli_dv_max_kms", "Max staged burn magnitude"),
    ("mcc_dv_max_kms", "MCC max burn magnitude"),
    ("r_moon_flyby", "Lunar flyby radius"),
    ("rp_min", "Return corridor min"),
    ("rp_max", "Return corridor max"),
    ("t_max", "Maximum mission time"),
]

BASE_REWARD_FIELDS = [
    ("w_flyby", r"$w_{\mathrm{flyby}}$"),
    ("w_return", r"$w_{\mathrm{return}}$"),
    ("w_dv", r"$w_{\Delta v}$"),
    ("w_budget", r"$w_{\mathrm{budget}}$"),
    ("w_escape", r"$w_{\mathrm{escape}}$"),
    ("w_earth_crash", r"$w_{\mathrm{earth}}$"),
    ("w_moon_crash", r"$w_{\mathrm{moon}}$"),
    ("w_postflyby_earth_crash", r"$w_{\mathrm{postflyby\,earth}}$"),
    ("w_invalid_preflyby_earth_return", r"$w_{\mathrm{invalid}}$"),
]

STAGE_FIELDS_ALWAYS = [
    ("timesteps", "Training steps"),
    ("entropy_coef", "Entropy coefficient"),
]

STAGE_REWARD_FIELDS = [
    ("w_flyby", r"$w_{\mathrm{flyby}}$"),
    ("w_return", r"$w_{\mathrm{return}}$"),
    ("w_dv", r"$w_{\Delta v}$"),
    ("w_budget", r"$w_{\mathrm{budget}}$"),
    ("w_escape", r"$w_{\mathrm{escape}}$"),
    ("w_earth_crash", r"$w_{\mathrm{earth}}$"),
    ("w_moon_crash", r"$w_{\mathrm{moon}}$"),
    ("w_postflyby_earth_crash", r"$w_{\mathrm{postflyby\,earth}}$"),
    ("w_invalid_preflyby_earth_return", r"$w_{\mathrm{invalid}}$"),
    ("dv_noise_sigma_tli", r"$\sigma_{\Delta v,\mathrm{TLI}}$"),
    ("dv_noise_sigma_mcc", r"$\sigma_{\Delta v,\mathrm{MCC}}$"),
]


@dataclass
class ParsedRun:
    path: Path
    kind: str
    label: str = ""
    timestamp: str = ""
    filename: str = ""
    sections: Dict[str, Dict[str, str]] = field(default_factory=dict)
    stages: List[Tuple[str, Dict[str, str]]] = field(default_factory=list)
    backend: Dict[str, str] = field(default_factory=dict)
    note: str = ""


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------

def find_existing_dir(names: List[str]) -> Optional[Path]:
    for name in names:
        p = ROOT / name
        if p.exists() and p.is_dir():
            return p
    return None


def parse_key_value(line: str) -> Optional[Tuple[str, str]]:
    if "=" not in line:
        return None
    key, val = line.split("=", 1)
    key = key.strip()
    val = val.strip()
    if not key:
        return None
    return key, val


def parse_config_file(path: Path, kind: str) -> ParsedRun:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    run = ParsedRun(path=path, kind=kind, filename=path.name)

    current_section = None
    current_stage_name = None
    current_stage_data: Optional[Dict[str, str]] = None
    in_backend = False

    for raw in lines:
        line = raw.rstrip("\n")

        if line.startswith("timestamp:"):
            run.timestamp = line.split(":", 1)[1].strip()
            continue

        if line.strip() == "=== actual_rl_backend ===":
            if current_stage_data is not None:
                run.stages.append((current_stage_name or f"Stage {len(run.stages)+1}", current_stage_data))
                current_stage_data = None
                current_stage_name = None
            in_backend = True
            current_section = None
            continue

        m_section = re.match(r"^\[(.+?)\]\s*$", line.strip())
        if m_section:
            if current_stage_data is not None:
                run.stages.append((current_stage_name or f"Stage {len(run.stages)+1}", current_stage_data))
                current_stage_data = None
                current_stage_name = None
            current_section = m_section.group(1).strip()
            run.sections.setdefault(current_section, {})
            in_backend = False
            continue

        m_stage = re.match(r"^Stage\s+(\d+)\s*:\s*(.+?)\s*$", line.strip())
        if m_stage:
            if current_stage_data is not None:
                run.stages.append((current_stage_name or f"Stage {len(run.stages)+1}", current_stage_data))
            current_stage_name = f"Stage {m_stage.group(1)}"
            current_stage_data = {"stage_name_original": m_stage.group(2).strip()}
            current_section = "CURRICULUM"
            in_backend = False
            continue

        kv = parse_key_value(line.strip())
        if kv is None:
            continue

        key, val = kv

        if in_backend:
            run.backend[key] = val
        elif current_stage_data is not None:
            current_stage_data[key] = val
        elif current_section is not None:
            run.sections.setdefault(current_section, {})[key] = val

    if current_stage_data is not None:
        run.stages.append((current_stage_name or f"Stage {len(run.stages)+1}", current_stage_data))

    return run


def get_value(run: ParsedRun, key: str, stage_index: Optional[int] = None) -> str:
    if stage_index is not None:
        if 0 <= stage_index < len(run.stages):
            return run.stages[stage_index][1].get(key, "--")
        return "--"

    # Search priority
    for section in [
        "PPO-LSTM",
        "RUN",
        "BURN CAPS",
        "TLI BALLISTIC TRIGGER",
        "CR3BP CONFIG BASE",
        "REWARD CONFIG DEFAULTS",
        "DRIFT MODEL",
        "PROPAGATION",
    ]:
        if key in run.sections.get(section, {}):
            return run.sections[section][key]

    # If not in base config, use stage 1 value as representative
    if run.stages and key in run.stages[0][1]:
        return run.stages[0][1][key]

    return "--"


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------

def latex_escape(s: str) -> str:
    s = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    return s


def fmt_num(s: str) -> str:
    if s is None:
        return "--"
    s = str(s).strip()
    if s == "" or s == "--":
        return "--"

    # common booleans/strings
    if s in ["True", "False", "auto", "full", "baseline", "scenario_library", "ppo_a", "ppo_b_library"]:
        return latex_escape(s)

    # path: keep only filename when very long
    if "/" in s or "\\" in s:
        return latex_escape(Path(s.replace("\\", "/")).name if "." in Path(s.replace("\\", "/")).name else s)

    try:
        x = float(s)
    except ValueError:
        return latex_escape(s)

    # integers with thin spaces
    if abs(x - round(x)) < 1e-12 and abs(x) >= 1000:
        return f"{int(round(x)):,}".replace(",", r"\,")
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))

    # scientific for tiny nonzero values
    if 0 < abs(x) < 1e-4:
        return f"${x:.3e}$"

    # powers of ten for learning rate
    if abs(x - 1e-4) < 1e-12:
        return r"$10^{-4}$"

    return f"{x:.6g}"


def all_same(values: List[str]) -> bool:
    clean = [v for v in values if v != "--"]
    return len(set(clean)) <= 1


def stage_field_should_show(runs: List[ParsedRun], key: str, stage_idx: int) -> bool:
    values = [get_value(r, key, stage_idx) for r in runs]
    # show if at least one value exists and not all missing
    if all(v == "--" for v in values):
        return False

    # always show training steps + entropy
    if key in {"timesteps", "entropy_coef"}:
        return True

    # show changing reward/noise values only if useful
    across_runs_change = not all_same(values)

    # also show if values change between stages inside any run
    stage_change = False
    for r in runs:
        vals = [get_value(r, key, i) for i in range(len(r.stages))]
        if not all_same(vals):
            stage_change = True
            break

    return across_runs_change or stage_change


def make_table_header(caption: str, label: str, runs: List[ParsedRun]) -> List[str]:
    ncols = len(runs) + 1
    colspec = "@{\\extracolsep{\\fill}}l" + "c" * len(runs)
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\scriptsize",
        rf"\begin{{tabular*}}{{\textwidth}}{{{colspec}}}",
        r"\toprule",
        r"\textbf{Parameter} & " + " & ".join(rf"\textbf{{{r.label}}}" for r in runs) + r"\\",
        r"\midrule",
    ]
    return lines


def section_row(title: str, runs: List[ParsedRun]) -> List[str]:
    return [
        rf"\multicolumn{{{len(runs)+1}}}{{c}}{{\textbf{{{title}}}}}\\",
        r"\midrule",
    ]


def add_rows(lines: List[str], fields: List[Tuple[str, str]], runs: List[ParsedRun], stage_idx: Optional[int] = None):
    for key, display in fields:
        vals = [fmt_num(get_value(r, key, stage_idx)) for r in runs]
        lines.append(display + " & " + " & ".join(vals) + r"\\")


def generate_config_table(kind: str, runs: List[ParsedRun]) -> str:
    is_tli = kind.upper() == "TLI"
    caption = (
        "Summary of the PPO-TLI training configurations used in the thesis. "
        "Only parameters relevant to comparison between runs and curriculum progression are shown."
        if is_tli else
        "Summary of the PPO-MCC training configurations used in the thesis. "
        "Only parameters relevant to comparison between runs and curriculum progression are shown."
    )
    label = "tab:ppoa_all_configs" if is_tli else "tab:ppob_all_configs"

    lines = make_table_header(caption, label, runs)

    lines += section_row("PPO Hyper-Parameters", runs)
    add_rows(lines, PPO_FIELDS, runs)

    lines += [r"\midrule"]
    lines += section_row("Mission / Environment Parameters", runs)
    add_rows(lines, TLI_MISSION_FIELDS if is_tli else MCC_MISSION_FIELDS, runs)

    lines += [r"\midrule"]
    lines += section_row("Base Reward Weights", runs)
    add_rows(lines, BASE_REWARD_FIELDS, runs, stage_idx=0)

    max_stages = max((len(r.stages) for r in runs), default=0)
    lines += [r"\midrule"]
    lines += section_row("Curriculum Stages", runs)

    for stage_idx in range(max_stages):
        lines.append(rf"\multicolumn{{{len(runs)+1}}}{{c}}{{\textit{{Stage {stage_idx+1}}}}}\\")
        lines.append(r"\midrule")

        add_rows(lines, STAGE_FIELDS_ALWAYS, runs, stage_idx=stage_idx)

        for key, display in STAGE_REWARD_FIELDS:
            if stage_field_should_show(runs, key, stage_idx):
                vals = [fmt_num(get_value(r, key, stage_idx)) for r in runs]
                lines.append(display + " & " + " & ".join(vals) + r"\\")

        if stage_idx != max_stages - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def generate_key_table(kind: str, runs: List[ParsedRun]) -> str:
    is_tli = kind.upper() == "TLI"
    caption = (
        "Mapping between PPO-TLI shorthand labels and original run identifiers."
        if is_tli else
        "Mapping between PPO-MCC shorthand labels and original run identifiers."
    )
    label = "tab:tli_run_key" if is_tli else "tab:mcc_run_key"

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\scriptsize",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llll}",
        r"\toprule",
        r"\textbf{Short label} & \textbf{Timestamp} & \textbf{Original file} & \textbf{Note}\\",
        r"\midrule",
    ]

    for r in runs:
        note = r.note or "--"
        lines.append(
            f"{r.label} & {latex_escape(r.timestamp)} & {latex_escape(r.filename)} & {latex_escape(note)}" + r"\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def collect_runs(kind: str) -> List[ParsedRun]:
    dir_path = find_existing_dir(TLI_DIR_NAMES if kind.upper() == "TLI" else MCC_DIR_NAMES)
    if dir_path is None:
        print(f"WARNING: no folder found for {kind}. Tried: {TLI_DIR_NAMES if kind.upper() == 'TLI' else MCC_DIR_NAMES}")
        return []

    files = sorted(dir_path.glob("*.txt"))
    runs = [parse_config_file(p, kind.upper()) for p in files]

    # Sort by timestamp if possible, otherwise filename
    runs.sort(key=lambda r: (r.timestamp or "", r.filename))

    prefix = "TLI" if kind.upper() == "TLI" else "MCC"
    for i, r in enumerate(runs, start=1):
        r.label = f"{prefix}-{i}"

        # Add automatic notes
        notes = []

        if kind.upper() == "TLI":
            spawn_min = get_value(r, "spawn_theta_min", stage_index=0)
            spawn_max = get_value(r, "spawn_theta_max", stage_index=0)
            notes.append(f"phase angle {spawn_min}--{spawn_max}")

            match_text = DIFFERENT_PHASE_TLI_MATCH.strip()
            if match_text and (match_text in r.filename or match_text in r.timestamp):
                notes.append("different initial phase angle")

        if kind.upper() == "MCC":
            idx = get_value(r, "ppo_b_fixed_index", stage_index=0)
            lib = get_value(r, "ppo_b_library_path", stage_index=0)
            notes.append(f"trajectory index {idx}")

            if idx != "--" and idx != "65":
                notes.append("non-default trajectory index")

            match_text = LUNAR_IMPACT_MCC_MATCH.strip()
            if match_text and (match_text in r.filename or match_text in r.timestamp):
                notes.append("TLI-generated lunar-impact trajectory")

            if "impact" in (r.filename + " " + lib).lower():
                notes.append("lunar-impact trajectory")

        for match, note in MANUAL_NOTES.items():
            if match in r.filename or match in r.timestamp:
                notes.append(note)

        # Remove duplicates while preserving order
        seen = set()
        clean_notes = []
        for n in notes:
            if n not in seen:
                seen.add(n)
                clean_notes.append(n)

        r.note = "; ".join(clean_notes)

    return runs


def write_csv(tli_runs: List[ParsedRun], mcc_runs: List[ParsedRun]):
    rows = []
    for r in tli_runs + mcc_runs:
        base = {
            "label": r.label,
            "kind": r.kind,
            "timestamp": r.timestamp,
            "filename": r.filename,
            "note": r.note,
            "gamma": get_value(r, "gamma"),
            "learning_rate": get_value(r, "learning_rate"),
            "n_envs": get_value(r, "n_envs"),
            "r_moon_flyby": get_value(r, "r_moon_flyby"),
            "rp_min": get_value(r, "rp_min"),
            "rp_max": get_value(r, "rp_max"),
        }
        for i in range(3):
            base[f"stage_{i+1}_timesteps"] = get_value(r, "timesteps", i)
            base[f"stage_{i+1}_entropy"] = get_value(r, "entropy_coef", i)
            base[f"stage_{i+1}_w_dv"] = get_value(r, "w_dv", i)
            base[f"stage_{i+1}_w_invalid"] = get_value(r, "w_invalid_preflyby_earth_return", i)
        if r.kind == "TLI":
            base["phase_min"] = get_value(r, "spawn_theta_min", 0)
            base["phase_max"] = get_value(r, "spawn_theta_max", 0)
        else:
            base["trajectory_index"] = get_value(r, "ppo_b_fixed_index", 0)
            base["trajectory_library"] = get_value(r, "ppo_b_library_path", 0)
        rows.append(base)

    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with (OUT_DIR / "parsed_config_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    tli_runs = collect_runs("TLI")
    mcc_runs = collect_runs("MCC")

    if not tli_runs and not mcc_runs:
        raise SystemExit("No config .txt files found. Check folder names: TLI/ and MCC/")

    outputs = []

    if tli_runs:
        tli_key = generate_key_table("TLI", tli_runs)
        tli_table = generate_config_table("TLI", tli_runs)
        (OUT_DIR / "tli_run_key_table.tex").write_text(tli_key, encoding="utf-8")
        (OUT_DIR / "ppo_tli_config_table.tex").write_text(tli_table, encoding="utf-8")
        outputs.extend([tli_key, tli_table])

    if mcc_runs:
        mcc_key = generate_key_table("MCC", mcc_runs)
        mcc_table = generate_config_table("MCC", mcc_runs)
        (OUT_DIR / "mcc_run_key_table.tex").write_text(mcc_key, encoding="utf-8")
        (OUT_DIR / "ppo_mcc_config_table.tex").write_text(mcc_table, encoding="utf-8")
        outputs.extend([mcc_key, mcc_table])

    (OUT_DIR / "all_config_tables.tex").write_text("\n\n".join(outputs), encoding="utf-8")
    write_csv(tli_runs, mcc_runs)

    print("\nGenerated LaTeX tables:")
    for p in sorted(OUT_DIR.glob("*.tex")):
        print("  ", p)
    print("\nGenerated CSV:")
    print("  ", OUT_DIR / "parsed_config_summary.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
