#!/usr/bin/env python3
"""Plan or execute a bounded return to the measured physical SO-101 photo home.

This is the everyday recovery command.  It uses the six calibrated HOME_TICKS
and the calibrated workcell geometry; it does not load a policy checkpoint.
By default it is read-only.  Add --execute only after the printed clearance
plan passes.
"""

import sys

from return_policy_home import main


if __name__ == "__main__":
    # The shared implementation retains the clearance, feedback, tracking,
    # and bounded-distance checks.  This entry point fixes its target source.
    sys.argv.insert(1, "--photo-home")
    main()
