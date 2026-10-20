# Gripper contact inspection

Cleanup note (2026-10-17): historical videos, per-frame traces, and diagnostic
images referenced below were removed. Retained JSON summaries remain evidence;
new runs can create fresh recordings. See [the cleanup guide](project_cleanup.md).

Use eval_pipeline.py --record-video --contact-overlay to render the authored
collision-pad boxes and PhysX mean contact positions on the saved video only.
The overlay is never fed into the actor. It does not change collision shapes,
friction, motor limits or success thresholds. Cyan is static_pad/fixed-side;
magenta is moving_pad/moving-side. Dots are mean body-to-cube contact positions,
not individual contact patches. Force labels describe whole-body contact, not
necessarily contact on the displayed pad. The trace stores world-space pad
corners, mean contact positions and body contact force magnitudes.

Headless recording:

```bash
python scripts/eval_pipeline.py --task lift --curriculum fixed --policy rgb \
  --checkpoint logs/rsl_rl/so101_picklift_rgb_tuned64/2026-10-14_09-57-08/model_1250.pt \
  --headless --episodes 1 --seed 101 --record-video --contact-overlay \
  --output outputs/evaluations/rgb_model1250_collision_overlay.json
```

Newton GUI alternative: replace --headless with --viz newton. The overlay is
in the saved MP4; the streaming camera continues to show unmodified RGB.
Evaluate separately from training to avoid RAM exhaustion. Compare the pad
outlines and contact dots with the visible fingers before changing geometry.
The body sensors include other collision shapes, so bilateral force alone is
not proof of pad-only support or real-robot grasp fidelity.

## Tighter CAD collision fit — 2026-10-15

Active asset: usd/robot_grasp_surface_v3.usda. Historical v2 and trained
checkpoints are preserved. The two actual finger mesh colliders use PhysX
convex decomposition with shrinkWrap=true, errorPercentage=0.1,
maxConvexHulls=128, hullVertexLimit=64, minThickness=0.0001 m and
voxelResolution=2000000. These options are defined in the installed PhysX schema.
The old invisible pad reference boxes remain non-colliding. Robot and cube
contact offsets remain 0.0005 m each; no motor/friction/success thresholds changed.

Offline script check_gripper_surface.py measures PhysX mean contact positions
against visible CAD triangles. These are not exact minimum cube/finger gaps:
mean contact points can lie away from a surface when multiple patches contribute.
The v2 fixed-finger sample had median distance 2.406 mm and maximum 3.124 mm.
The v3 ten-episode sample had median 0.228 mm and maximum 0.819 mm (38 samples).
Moving-finger median was 0.399 mm and maximum 2.215 mm. Do not interpret every
mean-point distance as a physical gap or claim mathematically exact contact.

model_1250.pt, fixed lift, seed 202: 10/10 held lifts, mean peak lift 9.661 cm,
mean table slide 0.948 cm. Video/report:
outputs/evaluations/rgb_model1250_surface_v3_10ep_seed202.*.
This is a small fixed-scene regression check, not the three-seed curriculum gate
or real-robot validation. All 53 retained tests passed.

```bash
python scripts/checks/check_gripper_surface.py \
  --asset usd/robot_grasp_surface_v3.usda \
  --trace outputs/evaluations/rgb_model1250_surface_v3_10ep_seed202.trace.json \
  --output outputs/evaluations/surface_v3_10ep_contact_distance.json
```

Use --headless for evaluation without a window, or --viz newton for the Newton
viewer. Diagnostic outlines are historical references, not active colliders.
