# Data and evaluation

Source: [UTD-MHAD, University of Texas at Dallas](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html).
The authors' [paper](https://jafari.tamu.edu/wp-content/uploads/2019/06/ICIP2015-Chen-Final.pdf)
describes the data as freely available and public domain for research. The site
does not provide a separate standard license file. Raw recordings are downloaded
from the authors and are not included in this repository or its code license.

Cite: C. Chen, R. Jafari, and N. Kehtarnavaz, *UTD-MHAD: A Multimodal Dataset for
Human Action Recognition Utilizing a Depth Camera and a Wearable Inertial Sensor*,
IEEE ICIP, 2015. DOI: 10.1109/ICIP.2015.7350781.

There are 861 trials, 27 actions, and 8 participants. A trial key identifies its
action, participant, and repetition. The loader pairs video and IMU keys exactly;
it rejects duplicate IDs, invalid shapes, non-finite signals, and mismatched sets.
The six `d_iner` columns contain acceleration XYZ followed by angular velocity
XYZ. The nominal sensor rate is 50 Hz. Wrist placement is used for actions 1–21;
thigh placement is used for 22–27.

The source synchronized captures, but the released six-column IMU arrays do not
include per-sample timestamps. Pairing checks confirm matching trials, not exact
frame-level alignment. No unrelated recordings are paired.

The sensor archive is 5.3 MB; RGB is 1.1 GB. Downloads resume after interruptions
and must match the SHA-256 checksums pinned in `fetch_data.py`.

Development: subjects 1, 3, 5. Validation: subject 7. Test: subjects 2, 4, 6, 8.
Select C from 0.1, 1, and 10 on validation macro-F1, then refit on 1, 3, 5, 7.
Report accuracy, macro-F1, confusion counts, per-class and per-subject results.
Do not tune against the test results. CI uses generated test fixtures and does
not download the dataset; fixture scores are not benchmark results.
