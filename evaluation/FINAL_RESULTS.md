# Syscall Anomaly Detection — Final Evaluation

## Final Model

The V2 LSTM Autoencoder is selected as the production baseline.

### Architecture

- Model: LSTM Autoencoder
- Embedding dimension: 32
- Hidden dimension: 64
- Sequence length: 20
- Vocabulary size: 136
- Unknown token: `<UNK>`

## Model Evolution

| Version | Approach | Normal FPR | Workload #1 | Workload #2 |
|---|---|---:|---:|---:|
| V1 | Baseline LSTM | 2.7315% | 99.9650% | 8.7205% |
| V2 | Diverse normal corpus | **1.8874%** | 0% | 8.5776% |
| V3 | PID-aware sequences | 3.1146% | 0% | 8.7205% |
| V4 | LSTM + rarity + transitions | 3.7257% | 100.0000% | 8.7920% |
| V5 | Smoothed behavioral hybrid | 3.3196% | 99.9880% | 8.7920% |

## Final Selection

V2 is selected because it achieved the lowest false-positive rate on the held-out normal dataset among the evaluated versions.

The behavioral hybrid experiments demonstrated that adding syscall rarity and transition novelty can substantially increase sensitivity to the first controlled anomaly workload, but this came with increased false positives on held-out normal data.

## Evaluation Caveat

Workload #2 was collected with a BPF collector warning indicating that samples may have been lost. Therefore, its detection rate should not be interpreted as a definitive measurement of model accuracy.

The reported anomaly metrics represent detection rates on the specific controlled workloads used in this experiment.

## Limitations

1. Evaluation was performed on a single machine.
2. Cross-machine generalization has not been established.
3. Workload #2 experienced syscall sample loss during collection.
4. The anomaly workloads represent controlled test behaviors rather than a comprehensive attack benchmark.
5. The detector uses syscall sequences and therefore may be sensitive to workload distribution.

## Conclusion

The experiments show that syscall-sequence reconstruction with an LSTM autoencoder can distinguish certain abnormal execution patterns from normal syscall behavior.

The V2 model provides the strongest normal-behavior false-positive performance among the evaluated versions and is therefore used as the final baseline.
