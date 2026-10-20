```text
Goal
Improve the SO-101 monitoring UI so a beginner can judge the visible robot's behavior from measured progress and clear messages, rather than interpreting oscillating reward curves.

Context (carry forward)
Repository: /home/ai/jamshi/isaac_vision_rl. Python, Isaac Lab, RSL-RL PPO, RGB only, Newton viewer and standalone ImGui/ImPlot reward window. Latest configured training version is v24_overlap_descend_close; verify current files and logs before making claims. Reaching and jaw opening improved in recorded evaluations, but grasp, held lift, and placement are unproven. Stage 2 occupancy is not grasp success. Earlier attempts repeatedly changed reward formulas without resolving the full task; this request is monitoring only.

Scope
Inspect reward_panel.py, training_reward_charts.py, reward_window.py, four_stage.py, scripts/staged_task.py and their relevant tests under src/isaac_vision_rl/utils, scripts, and tests. Explain the current data flow and proposed changes briefly before editing. Modify only monitoring/telemetry/UI and focused tests/documentation. Preserve reward functions, stage transitions, physics, actor/critic observations, action controllers, checkpoints, RGB preprocessing and manual controls.

Required behavior
- Default to a beginner-facing behavior dashboard. Keep raw reward plots in a clearly labeled Advanced section.
- Show Environment 0, checkpoint/run version, sample freshness, episode identifier, current stage and the exact next-stage checks. Separate visible-environment readings from batch-level learning statistics.
- Show big cards for target distance in cm, horizontal alignment error in cm, downward wrist alignment, actual jaw opening versus its threshold, valid opposing finger contact, cube rise from reset in cm, held-lift dwell time, table clearance in mm, dragging distance and episode outcome.
- Use units and meaningful threshold/target bands on plots. Plot a rolling recent window by default with an optional full-session view; retain reset boundaries and never connect unrelated episodes as continuous motion.
- Display distinct, evidence-grounded statuses such as Approaching, Overhead pose reached, Opening jaws, Descending, Closing near cube, Contact on one side, Valid two-finger grasp, Stable held lift, Placement success, Stalled, or Unsafe. Report unknown/stale telemetry explicitly.
- Explain the immediate blocker in one plain sentence, for example: 'Above the cube, but jaws are not open enough' or 'Jaws open; the hand is still 7 cm above the cube.' Determine this from actual measurements and configured conditions, not reward direction or an arbitrary overall goodness score.
- Color measured unsafe conditions red, unmet/waiting/stalled conditions amber, and satisfied checks green, with text labels too. Do not label a policy mastered or safe for hardware based on this UI.
- Provide a separate recent episode summary: attempts, reached stage, valid grasps, sustained lifts, placements, failures and timeouts, with clear denominators. Keep these distinct from checkpoint evaluation results.
- The panel must remain read-only. Use existing dependencies and a compact telemetry stream; avoid extra cameras, repeated GPU-to-CPU copies, blocking writes, or unbounded Python object history. Preserve responsive training if the panel closes or slows.

Verification and boundaries
Do not stop/restart running training, move real motors, launch a second simulator during training, send email, delete files, install dependencies, or commit/push. Ask before work outside scope. Use synthetic recorded snapshots to test: hover with closed jaws, open overhead hover, centered descent, one-finger contact, valid two-finger grasp, stable lift, timeout, unsafe motion, reset and stale data. Verify both standalone and Newton UI paths with a bounded smoke run only when resources are available and authorized. Never invent a passing simulator or visual check.

Done
Deliver the implemented dashboard, focused tests, actual verification results, changed-file summary, known limitations, and exact commands for headless graphs and one visible environment in Newton. Stop once this monitoring task is verified; do not retune training.
```

🎯 Target: Codex. 💡 Optimized for measured behavior and explicit stage blockers instead of misleading reward-only judgments.

This prompt is for an agentic tool with real system access. Review the scope locks, forbidden actions, and stop conditions before pasting. Confirm file paths, directories, and permissions match the actual project.
