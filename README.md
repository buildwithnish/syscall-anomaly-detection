# Linux System Call Anomaly Detection

A Linux-based anomaly detection system that collects system call events and uses an LSTM Autoencoder to identify syscall sequences that differ from learned normal behavior.

## Project Overview

The project combines Linux syscall collection, event preprocessing, sequence-based anomaly scoring, and threshold-based alerting.

### Architecture

1. **Collection:** Capture Linux syscall events using the project's BCC-based collector.
2. **Preprocessing:** Map syscall identifiers and build sequences for each process.
3. **Model:** Use a trained V2 LSTM Autoencoder to calculate reconstruction-based anomaly scores.
4. **Detection:** Compare scores against a configured threshold and report anomalous behavior.
5. **Evaluation:** Compare model versions using held-out normal data and controlled anomaly workloads.

## Technology Stack

- Python
- PyTorch
- Linux system call tracing
- BCC/eBPF-based collection
- LSTM Autoencoder
- JSONL event data

## Model Configuration

- Model: V2 LSTM Autoencoder
- Embedding dimension: 32
- Hidden dimension: 64
- Sequence length: 20
- Vocabulary size: 136
- Unknown token: `<UNK>`
- Score threshold: 0.304810 (configured threshold)

Model artifacts are stored under `models/`.

## Run the Detector

Activate the project's Python virtual environment:

```bash
source .venv/bin/activate
```

Run detection on an existing JSONL event file:

```bash
PYTHONPATH=. python src/detection/detector.py PATH_TO_EVENTS.jsonl
```

Replace `PATH_TO_EVENTS.jsonl` with the path to a compatible event file.

To follow a file as new events are appended:

```bash
PYTHONPATH=. python src/detection/detector.py --follow PATH_TO_EVENTS.jsonl
```

The live detector starts at the current end of the file and processes subsequently appended records.

## Live Collection

Live collection requires Linux tracing support, BCC dependencies, and appropriate privileges.

Example:

```bash
sudo env SYSCALL_OUTPUT_FILE=data/raw/live_syscalls.jsonl PYTHONPATH=src/collector .venv/bin/python src/collector/collect_dataset.py
```

In another terminal, run the detector:

```bash
source .venv/bin/activate
PYTHONPATH=. python src/detection/detector.py --follow data/raw/live_syscalls.jsonl
```

Stop both processes after the test. Live collection can generate large files quickly, so keep initial demonstrations short.

## Evaluation

See [`evaluation/FINAL_RESULTS.md`](evaluation/FINAL_RESULTS.md) for the recorded model comparison and limitations.

V2 was selected because it had the lowest held-out normal false-positive rate among the evaluated versions: 1.8874%.

The controlled anomaly results varied by workload. In particular, V2 did not detect the first controlled anomaly workload in the recorded evaluation. These results do not establish comprehensive attack detection accuracy.

## Limitations

- Evaluated on a single machine.
- Generalization across machines has not been established.
- One controlled workload was affected by possible syscall sample loss.
- Results depend on workload distribution and the selected threshold.
- An anomaly alert indicates unusual behavior, not proof of a malicious attack.
- Live performance and alert quality require further benchmarking.

## Responsible Use

Use this project for authorized monitoring and research on systems you control or have permission to inspect. Review alerts in context before drawing security conclusions.
