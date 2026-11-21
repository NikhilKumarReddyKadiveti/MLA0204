"""
Synthetic student dataset generator for EduPredict.

Run:  python generate_dataset.py
Optional arguments:
    --records 300     number of synthetic students (minimum 30)
    --seed 42         random seed (same seed = same dataset)
    --output PATH     output CSV path (default: sample_students.csv next to this script)

IMPORTANT: every record produced here is SYNTHETIC (randomly generated).
It does not describe any real student.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "Student_ID",
    "Attendance",
    "Previous_Exam_Marks",
    "Assignment_Marks",
    "Internal_Marks",
    "Study_Hours",
    "Class_Participation",
    "Assignment_Completion",
    "Performance",
]

THEME_CONFIG = """[theme]
base = "light"
primaryColor = "#1d4ed8"
backgroundColor = "#f4f7fc"
secondaryBackgroundColor = "#ffffff"
textColor = "#1f2937"
font = "sans serif"

[browser]
gatherUsageStats = false
"""


def generate_students(n_records: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic student dataset.

    A hidden 'ability' factor drives all features, so they are naturally
    correlated (strong students tend to attend more, score more and study more).
    The Performance label is derived from a weighted composite of the features
    plus random noise, so a model can learn it but cannot reach 100% accuracy.
    """
    n = int(n_records)
    if n < 30:
        raise ValueError("n_records must be at least 30.")
    rng = np.random.default_rng(seed)

    ability = rng.normal(0, 1, n)

    attendance = np.clip(rng.normal(80, 9, n) + 4.0 * ability, 40, 100)
    study_hours = np.clip(rng.normal(3.3, 1.2, n) + 0.6 * ability, 0.3, 9.5)
    previous = np.clip(62 + 11 * ability + rng.normal(0, 8, n), 20, 100)
    assignment = np.clip(66 + 8 * ability + 0.10 * (attendance - 80) + rng.normal(0, 7, n), 20, 100)
    internal = np.clip(14 + 0.40 * previous + 0.30 * assignment + 0.15 * attendance + rng.normal(0, 5, n), 20, 100)
    completion = np.clip(82 + 7 * ability + 0.15 * (attendance - 80) + rng.normal(0, 8, n), 30, 100)

    part_score = 0.9 * ability + rng.normal(0, 0.7, n)
    participation = np.where(part_score < -0.45, "Low", np.where(part_score > 0.55, "High", "Medium"))
    part_points = pd.Series(participation).map({"Low": 40, "Medium": 65, "High": 90}).to_numpy()

    composite = (
        0.24 * previous
        + 0.20 * internal
        + 0.14 * assignment
        + 0.14 * attendance
        + 0.10 * completion
        + 0.10 * np.minimum(study_hours * 10, 100)
        + 0.08 * part_points
        + rng.normal(0, 2.5, n)
    )
    low_cut, high_cut = np.percentile(composite, [25, 70])
    performance = np.where(composite < low_cut, "Low", np.where(composite >= high_cut, "High", "Medium"))

    df = pd.DataFrame(
        {
            "Student_ID": [f"STU{i + 1:04d}" for i in range(n)],
            "Attendance": attendance.round(1),
            "Previous_Exam_Marks": previous.round(1),
            "Assignment_Marks": assignment.round(1),
            "Internal_Marks": internal.round(1),
            "Study_Hours": study_hours.round(1),
            "Class_Participation": participation,
            "Assignment_Completion": completion.round(1),
            "Performance": performance,
        }
    )
    return df[COLUMNS]


def write_theme_config(base_dir: Path) -> None:
    """Create .streamlit/config.toml (blue/white theme) if it does not exist."""
    cfg_dir = base_dir / ".streamlit"
    cfg_file = cfg_dir / "config.toml"
    if not cfg_file.exists():
        cfg_dir.mkdir(exist_ok=True)
        cfg_file.write_text(THEME_CONFIG, encoding="utf-8")
        print(f"Created theme file: {cfg_file}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate a synthetic student performance dataset.")
    parser.add_argument("--records", type=int, default=300, help="number of records (default 300)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--output", type=str, default=str(base_dir / "sample_students.csv"))
    args = parser.parse_args()

    df = generate_students(args.records, args.seed)
    out_path = Path(args.output)
    df.to_csv(out_path, index=False)
    write_theme_config(base_dir)

    print(f"Saved {len(df)} SYNTHETIC student records to: {out_path}")
    print("Class distribution:")
    print(df["Performance"].value_counts().to_string())
    print("\nNote: this data is randomly generated and does not describe real students.")


if __name__ == "__main__":
    main()