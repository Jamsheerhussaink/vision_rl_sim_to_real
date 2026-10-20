"""Gate regressions: training completion is never treated as lift mastery."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from isaac_vision_rl.utils import pipeline


class PipelineTests(unittest.TestCase):
    def test_finger_contact_buffers_cover_mesh_manifolds(self):
        from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg

        cfg = PickLiftEnvCfg()
        for sensor in (cfg.scene.fixed_finger_contact, cfg.scene.moving_finger_contact):
            self.assertTrue(sensor.track_contact_points)
            self.assertGreaterEqual(sensor.max_contact_data_count_per_prim, 64)

    def test_hand_table_sensors_filter_the_table_rigid_body(self):
        from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg
        cfg = PickLiftEnvCfg()
        for name in ("fixed_table_contact", "moving_table_contact", "wrist_table_contact"):
            sensor = getattr(cfg.scene, name)
            self.assertEqual(sensor.filter_prim_paths_expr,
                             ["{ENV_REGEX_NS}/Table/Geometry/table_base_link/table_top_link"])

    def test_stale_workcell_is_rejected_by_default(self):
        import json

        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "ROOT", Path(folder)):
            evidence = Path(folder) / "outputs/pipeline"
            evidence.mkdir(parents=True)
            (evidence / "workcell_physics.json").write_text(json.dumps({"passed": True}))
            with patch.object(pipeline, "sha256", return_value="current"):
                with self.assertRaisesRegex(ValueError, "stale"):
                    pipeline.read_physical_report()

    def reports(self):
        return [
            {
                "seed": seed,
                "episodes": 50,
                "success_rate": 0.92,
                "mean_table_slide_cm": 0.2,
                "curriculum": "lift",
                "task": "lift",
            }
            for seed in pipeline.SEEDS
        ]

    def test_each_seed_must_pass_success_and_drag(self):
        reports = self.reports()
        self.assertTrue(pipeline.evaluations_pass(reports))
        reports[1]["success_rate"] = 0.88
        self.assertFalse(pipeline.evaluations_pass(reports))
        reports[1]["success_rate"] = 0.92
        reports[2]["mean_table_slide_cm"] = 1.1
        self.assertFalse(pipeline.evaluations_pass(reports))
        self.assertFalse(pipeline.evaluations_pass(reports[:2]))

    def test_changed_checkpoint_and_failed_gate_cannot_advance(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(pipeline, "ROOT", Path(folder)),
            patch.object(pipeline, "fingerprint", return_value={"config": "v1"}),
        ):
            checkpoint = Path(folder) / "model.pt"
            checkpoint.write_bytes(b"policy1")
            reports = self.reports()
            for r in reports:
                r["checkpoint"] = str(checkpoint)
            pipeline.write_gate("lift", checkpoint, reports)
            self.assertEqual(pipeline.require_gate("lift"), str(checkpoint))
            checkpoint.write_bytes(b"policy2")
            with self.assertRaisesRegex(ValueError, "stale"):
                pipeline.require_gate("lift")
            reports[0]["success_rate"] = 0
            pipeline.write_gate("lift", checkpoint, reports)
            with self.assertRaisesRegex(ValueError, "not passed"):
                pipeline.require_gate("lift")

    def test_wrong_curriculum_cannot_be_used_as_lift_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "model.pt"
            checkpoint.write_bytes(b"policy")
            reports = self.reports()
            for r in reports:
                r["checkpoint"] = str(checkpoint)
                r["curriculum"] = "grasp"
            with self.assertRaisesRegex(ValueError, "curriculum"):
                pipeline.write_gate("lift", checkpoint, reports)

    def test_fixed_stage_disables_domain_randomization(self):
        from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg

        cfg = PickLiftEnvCfg()
        pipeline.configure_curriculum(cfg, "fixed")
        self.assertIsNotNone(cfg.events.reset_cube_pose)
        self.assertEqual(cfg.events.reset_cube_pose.params["pose_range"], {})
        self.assertEqual(cfg.events.reset_cube_pose.params["velocity_range"], {})
        self.assertIsNone(cfg.events.randomize_cube_mass)
        self.assertIsNone(cfg.events.randomize_cube_friction)

    def test_mild_cube_jitter_keeps_reset_and_fixed_physics(self):
        import os

        from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg

        with patch.dict(os.environ, {"SO101_CURRICULUM": "fixed", "SO101_CUBE_JITTER": "0.01"}):
            cfg = PickLiftEnvCfg()
            self.assertEqual(cfg.events.reset_cube_pose.params["pose_range"]["x"], (-0.01, 0.01))
            self.assertEqual(cfg.events.reset_cube_pose.params["pose_range"]["y"], (-0.01, 0.01))
            self.assertIsNone(cfg.events.randomize_cube_mass)
            self.assertIsNone(cfg.events.randomize_cube_friction)

    def test_newton_focus_uses_environment_origin_once(self):
        from types import SimpleNamespace as NS
        from unittest.mock import Mock

        import torch

        from isaac_vision_rl.tasks.pickplace.mdp.events import select_task_camera_view

        visualizer = NS(cfg=NS(visualizer_type="newton_gl"), set_camera_view=Mock())
        env = NS(
            sim=NS(has_gui=False, visualizers=[visualizer]),
            scene=NS(env_origins=torch.tensor([[10.0, 20.0, 0.0]])),
            cfg=NS(sim=NS(default_visualizer_cfg=NS(eye=(0.055, -0.3, 0.98), lookat=(0.055, 0.11, 0.4)))),
        )
        select_task_camera_view(env)
        visualizer.set_camera_view.assert_called_once_with((10.055, 19.7, 0.98), (10.055, 20.11, 0.4))
        select_task_camera_view(env)
        self.assertEqual(visualizer.set_camera_view.call_count, 1)

    def test_base_is_upright_and_anchored(self):
        import torch

        from isaaclab.utils.math import matrix_from_quat

        from isaac_vision_rl.tasks.pickplace.config.so101.env_cfg import PickLiftEnvCfg

        cfg = PickLiftEnvCfg()
        self.assertTrue(cfg.scene.robot.spawn.articulation_props.fix_root_link)
        rotation = matrix_from_quat(torch.tensor([cfg.scene.robot.init_state.rot]))
        torch.testing.assert_close(rotation[0, :, 2], torch.tensor([0.0, 0.0, 1.0]))
        self.assertIsNone(cfg.rewards.reach_cube)


if __name__ == "__main__":
    unittest.main()
