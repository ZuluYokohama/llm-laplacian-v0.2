# ETA from the Arm 0 smoke and one M=8 pulse

- Plain M=4 seconds/step: 0.354415
- Spectral M=4 seconds/step at the step-135000 weights, 2.026190260000294 probe: 2.026190
- One M=8 culling pulse, calibration 512 sequences: 605.349 seconds
- Plain 300k runs counted at the plain rate: arm0, arm1, arm3
- Spectral 300k runs counted at the spectral rate: arm2, arm4, scale_m2, scale_m4, scale_m16, grid_s0p25_l15, grid_s0p25_l25, grid_s0p25_l135, grid_s0p5_l15, grid_s0p5_l135, grid_s0p75_l15, grid_s0p75_l25, grid_s0p75_l135
- Arm 1 is M=32 and Arm 5 is SmolLM2-135M. Neither rate was measured here, so both are outside this projection.
- Projection of the counted TinyStories runs plus seven pulses on the four full-protocol widths that are not aliases: 96.76 days
- Aliases grid s=0.5 layers {2,5} and scale M=8 are not counted twice.
