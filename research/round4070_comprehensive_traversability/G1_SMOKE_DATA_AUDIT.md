# G1 actual logging smoke audit

Completed probe-valid histories inspected: 2. No model fitting or simulation was started by this audit.

- G_EMPTY_left_empty_control_00_nominal_s74800 / attempt01: label=PASSABLE_FOR_ACTION, runtime=True, probe=True, physics=True; 102 real probe samples, median dt=0.020000 s (50.0 Hz), span=1.995 s; previous-action updates verified=101; causal cutoff=True, actual upper q_cmd=True; acceleration=SIGNAL_PRESENT_NOT_HARDWARE_CALIBRATED.
- G_EMPTY_right_empty_control_00_nominal_s74800 / attempt01: label=PASSABLE_FOR_ACTION, runtime=True, probe=True, physics=True; 102 real probe samples, median dt=0.020000 s (50.0 Hz), span=1.995 s; previous-action updates verified=101; causal cutoff=True, actual upper q_cmd=True; acceleration=SIGNAL_PRESENT_NOT_HARDWARE_CALIBRATED.

The acceleration candidate is retained robot state, not hardware IMU calibration. Unless an explicit independent validation flag is present, it remains unavailable to every model with an explicit missingness mask. A valid probe followed by an execution failure may remain UNKNOWN response history; this does not make that entire runtime valid.
