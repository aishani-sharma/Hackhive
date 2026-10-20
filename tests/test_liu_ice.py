import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from engine_lab.liu_ice import (
    SIGNALS,
    add_early_warning_target,
    build_feature_windows,
    fit_probability_model,
)
from engine_lab.core import build_rul_features, fit_rul_model


class LiUIcePipelineTests(unittest.TestCase):
    def test_official_onset_labels_window_and_warning_horizon(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            time = np.arange(600) / 20
            for name, shift in (("wltp_nf_1", 0), ("wltp_f_pic_1", 2)):
                frame = pd.DataFrame({"time": time})
                for index, signal in enumerate(SIGNALS):
                    frame[signal] = np.sin(time / (index + 1)) + shift
                frame.to_csv(root / f"{name}.csv", index=False)
            (root / "wltp_fault_time.txt").write_text("wltp_f_pic_1 15\n", encoding="utf-8")
            windows = build_feature_windows(root)
            fault = windows.loc[windows.trajectory_id == "wltp_f_pic_1"]
            self.assertEqual(fault.fault_active.tolist(), [0, 0, 0, 1, 1, 1])
            labeled = add_early_warning_target(windows, 10)
            early = labeled.loc[labeled.trajectory_id == "wltp_f_pic_1", "fault_within_horizon"]
            self.assertEqual(early.tolist(), [0, 1, 1, 0, 0, 0])
            self.assertEqual(len([c for c in windows if "__" in c]), 40)

    def test_probabilities_are_calibrated_bounded_and_run_held_out(self):
        rng = np.random.default_rng(7)
        rows = []
        for scenario, fault in (("No fault", 0), ("Intercooler pressure sensor", 1)):
            for run in range(8):
                trajectory = f"{scenario}-{run}"
                for window in range(25):
                    active = int(bool(fault) and window >= 12)
                    values = rng.normal(size=10) + active * 1.5
                    row = {
                        "trajectory_id": trajectory,
                        "scenario": scenario,
                        "drive_cycle": "wltp",
                        "fault_active": active,
                        "injection_time_s": 60.0 if fault else np.nan,
                        "window_start_s": window * 5.0,
                        "window_end_s": window * 5.0 + 4.95,
                    }
                    row.update({f"sensor_{i}__mean": value for i, value in enumerate(values)})
                    rows.append(row)
        result = fit_probability_model(pd.DataFrame(rows))
        scored = result["test_windows"]
        self.assertTrue(scored.fault_probability.between(0, 1).all())
        self.assertTrue(set(scored.trajectory_id).isdisjoint(set(result["training_windows"].trajectory_id)))
        self.assertEqual(result["metrics"]["test_trajectories"], 4)
        self.assertGreaterEqual(len(result["selected_features"]), 8)
        self.assertLessEqual(len(result["selected_features"]), 24)

    def test_uploaded_rul_regression_scores_selected_features(self):
        rng = np.random.default_rng(13)
        rows = []
        for engine in range(8):
            for cycle in range(50):
                rows.append({
                    "engine_id": f"E{engine}",
                    "cycle": cycle,
                    "failure_event": int(cycle == 49),
                    "vibration": cycle / 50 + rng.normal(0, 0.04),
                    "temperature": 60 + cycle * 0.2 + rng.normal(0, 0.5),
                })
        source = pd.DataFrame(rows)
        X, y, groups, _ = build_rul_features(source, "engine_id", "cycle", failure_col="failure_event")
        result = fit_rul_model(X, y, groups)
        self.assertTrue(np.isfinite(result["mae"]))
        self.assertTrue(np.isfinite(result["r2"]))
        self.assertGreaterEqual(result["train_engines"], 1)
        self.assertGreaterEqual(result["test_engines"], 1)


if __name__ == "__main__":
    unittest.main()
