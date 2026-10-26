# Cumulative RGB curriculum experiment

Launch with `scripts/train_curriculum.py --lesson approach|grasp|lift|place`
and the usual simple-reward launch arguments. Logs have separate experiment
names for each lesson. Existing checkpoints and the original launcher are preserved.

Each lesson starts from home and ends after that lesson's verified stage gate.
Grasp therefore practises approach then grasp; Lift practises approach, grasp,
then lift; Place practises the whole sequence. These are cumulative lessons,
not pre-grasp or pre-held resets. No IK commands or teacher actions are introduced.

The three existing reward terms for the active stage are retained. Ending an
Approach or Grasp lesson adds no success bonus. Lift retains the configured
height threshold, with `--lift-hold-seconds 0` for no extra hold. Existing safety
terminations remain active. Table-force calibration is still outstanding; this
experiment does not relax those checks.

Carry the actor into the next lesson using `--actor-warmstart --checkpoint PATH`.
That resets critic and optimizer state. Promotion is manual after recorded
policy evaluation; completion of an iteration budget is not a mastery check.
Use the same evaluation positions for each candidate and report approach,
supported grasp, cube sliding, table contact and full-task success separately.
The first launch is a bounded Approach smoke run. Longer training should follow
only after its environment, checkpoint loading, logging and save path work.
