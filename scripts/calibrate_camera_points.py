#!/usr/bin/env python3
"""Fit camera-to-base extrinsics from measured corresponding 3D landmarks."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", default="outputs/perception/camera_to_base.json")
    args = parser.parse_args()
    import numpy as np

    from isaac_vision_rl.perception.cube import fit_camera_to_base

    data = json.loads(Path(args.pairs).read_text())
    if data.get("units") != "metres" or data.get("camera_frame") != "color_optical" or data.get("source") != "measured":
        parser.error("Use measured metre-valued correspondences in the color optical frame")
    T, residual = fit_camera_to_base(data["camera_points_m"], data["base_points_m"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "source": "measured",
                "units": "metres",
                "camera_frame": "color_optical",
                "base_z_up": True,
                "T_base_camera": T.tolist(),
                "fit_max_error_m": float(max(residual)),
                "fit_rms_error_m": float(np.sqrt(np.mean(residual**2))),
                "note": "In-sample fit only: verify independent landmarks before control",
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
