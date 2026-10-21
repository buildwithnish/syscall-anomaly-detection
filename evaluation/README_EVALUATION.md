
## Evaluation

The model was evaluated using a normal-only calibration set, a
held-out normal test set, and two independently collected
PID-isolated controlled workloads.

### Configuration

- Model: LSTM Autoencoder
- Embedding dimension: 32
- Hidden dimension: 64
- Sequence length: 20
- Vocabulary: 109 known syscalls + `<UNK>`
- Threshold: `0.304810`
- Threshold method: normal-only 99th percentile calibration

### Results

| Dataset | Sequences | Result |
|---|---:|---:|
| Held-out normal | 99,981 | 2.7315% false-positive rate |
| Isolated workload 1 | 99,981 | 99.9650% detected |
| Independent workload 2 | 1,399 | 8.7205% detected |

### Interpretation

The detector successfully identifies highly out-of-distribution
syscall sequences in the first controlled isolated workload.
The independent second workload shows substantially lower
sequence-level detection, demonstrating that anomaly sensitivity
depends on the syscall distribution of the workload.

The threshold and model were kept frozen during these experiments.
The controlled workloads should not be interpreted as confirmed
malicious attacks.

### Limitations

- Evaluation workloads are controlled experiments rather than
  real-world attack traces.
- Sequence-level detection does not directly represent
  process-level attack detection.
- The held-out normal false-positive rate was higher than the
  1% calibration target.
- Additional datasets containing real benign applications and
  representative attack traces are required for production
  security evaluation.
