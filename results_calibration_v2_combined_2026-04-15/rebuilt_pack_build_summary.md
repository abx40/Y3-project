# Leak-Free Calibration V2 Build

## Calibration Sessions
- calibration fake_only #1: 599.967s, real=0.000s, fake=599.967s, segments=42
- calibration fake_only #2: 600.333s, real=0.000s, fake=600.333s, segments=40
- calibration mixed #1: 602.333s, real=300.127s, fake=302.206s, segments=47
- calibration mixed #2: 605.900s, real=304.401s, fake=301.499s, segments=48
- calibration mixed #3: 603.333s, real=300.229s, fake=303.104s, segments=46
- calibration mixed #4: 600.400s, real=300.132s, fake=300.268s, segments=46
- calibration real_only #1: 600.233s, real=600.233s, fake=0.000s, segments=41
- calibration real_only #2: 603.533s, real=603.533s, fake=0.000s, segments=44
- calibration real_only #3: 600.300s, real=600.300s, fake=0.000s, segments=44

## Leakage Checks
- calibration identity overlap vs validation: 0
- calibration identity overlap vs test: 0
- calibration source overlap vs validation: 0
- calibration source overlap vs test: 0

## Summary
- calibration real segment seconds: 3008.966
- calibration fake segment seconds: 2407.365
- validation clips reused unchanged: real=369, fake=1367
- test clips reused unchanged: real=369, fake=1392

## Notes
- Validation and test sessions were hard-linked from the existing leak-free pack.
- New calibration sessions were drawn only from clips assigned to the calibration split in `manifest_with_splits.csv`.
- No source clip was reused across the new calibration sessions.
- Sessions were normalized to 1280x720 at 30 FPS and written as video-only MP4 files.
