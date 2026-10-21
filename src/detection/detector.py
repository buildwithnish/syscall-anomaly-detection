import time
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.lstm_autoencoder import LSTMAutoencoder


# ============================================================
# FILES
# ============================================================

MODEL_FILE = PROJECT_ROOT / "models/v2_lstm_autoencoder.pth"
CONFIG_FILE = PROJECT_ROOT / "models/v2_model_config.json"
THRESHOLD_FILE = PROJECT_ROOT / "models/threshold.json"


# ============================================================
# DETECTOR CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 20

# Score every N events for each PID.
# 1 = maximum sensitivity
# 5 = significantly less computation
SCORE_EVERY = 5

# Number of consecutive anomaly windows required
# before creating an alert.
ALERT_CONFIRMATIONS = 2


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD CONFIG
# ============================================================

with open(CONFIG_FILE, "r") as file:
    config = json.load(file)


VOCAB_SIZE = config["vocab_size"]
EMBEDDING_DIM = config["embedding_dim"]
HIDDEN_DIM = config["hidden_dim"]
UNK_INDEX = config["unk_index"]

UNIQUE_SYSCALLS = config["unique_syscalls"]

SYSCALL_TO_INDEX = {
    int(syscall_id): index
    for index, syscall_id in enumerate(UNIQUE_SYSCALLS)
}


# ============================================================
# LOAD THRESHOLD
# ============================================================

with open(THRESHOLD_FILE, "r") as file:
    threshold_config = json.load(file)

THRESHOLD = float(
    threshold_config["threshold"]
)


# ============================================================
# LOAD MODEL
# ============================================================

model = LSTMAutoencoder(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    reduction="none"
)


# ============================================================
# SCORE ONE SEQUENCE
# ============================================================

def calculate_score(syscall_ids):

    mapped = [
        SYSCALL_TO_INDEX.get(
            int(syscall_id),
            UNK_INDEX
        )
        for syscall_id in syscall_ids
    ]

    sequence = torch.tensor(
        [mapped],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        output = model(sequence)

        loss = criterion(
            output.reshape(-1, VOCAB_SIZE),
            sequence.reshape(-1)
        )

        score = loss.mean().item()

    return float(score)


# ============================================================
# PID STATE
# ============================================================

class PIDState:

    def __init__(self):

        self.sequence = deque(
            maxlen=SEQUENCE_LENGTH
        )

        self.events_since_score = 0

        self.consecutive_anomalies = 0

        self.incident_active = False

        self.max_score = 0.0

        self.windows = 0


# ============================================================
# PROCESS EVENT
# ============================================================

def process_event(event, states):

    pid = int(event["pid"])

    syscall_id = int(
        event["syscall_id"]
    )

    state = states[pid]

    state.sequence.append(
        syscall_id
    )

    state.events_since_score += 1

    # Not enough events yet
    if len(state.sequence) < SEQUENCE_LENGTH:
        return None

    # Reduce inference frequency
    if state.events_since_score < SCORE_EVERY:
        return None

    state.events_since_score = 0

    score = calculate_score(
        state.sequence
    )

    state.windows += 1

    if score > state.max_score:
        state.max_score = score

    if score > THRESHOLD:

        state.consecutive_anomalies += 1

        if (
            state.consecutive_anomalies
            >= ALERT_CONFIRMATIONS
        ):

            if not state.incident_active:

                state.incident_active = True

                return {
                    "type": "ALERT",
                    "pid": pid,
                    "process": event.get(
                        "process",
                        "unknown"
                    ),
                    "score": score,
                    "timestamp": event.get(
                        "timestamp"
                    ),
                    "count": state.consecutive_anomalies
                }

    else:

        state.consecutive_anomalies = 0

        state.incident_active = False

    return None


# ============================================================
# PROCESS FILE
# ============================================================

def process_file(input_file):

    states = defaultdict(
        PIDState
    )

    total_events = 0
    total_windows = 0
    total_alerts = 0
    invalid_events = 0

    print()
    print("=" * 70)
    print("              SYSCALL ANOMALY DETECTOR")
    print("=" * 70)
    print()
    print("Model             : V2 LSTM Autoencoder")
    print(f"Device            : {DEVICE}")
    print(f"Sequence length   : {SEQUENCE_LENGTH}")
    print(f"Vocabulary size   : {VOCAB_SIZE}")
    print(f"Threshold         : {THRESHOLD:.6f}")
    print(f"Score interval    : {SCORE_EVERY} events")
    print(
        f"Alert confirmation: "
        f"{ALERT_CONFIRMATIONS} windows"
    )
    print(f"Input             : {input_file}")
    print()
    print("-" * 70)

    with open(input_file, "r") as file:
    
        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                event = json.loads(line)

                if "pid" not in event:
                    raise ValueError(
                        "missing pid"
                    )

                if "syscall_id" not in event:
                    raise ValueError(
                        "missing syscall_id"
                    )

                alert = process_event(
                    event,
                    states
                )

                total_events += 1

                if alert is not None:

                    total_alerts += 1

                    print()
                    print("!" * 70)
                    print("                 ANOMALY ALERT")
                    print("!" * 70)

                    print(
                        f"PID       : {alert['pid']}"
                    )

                    print(
                        f"Process   : "
                        f"{alert['process']}"
                    )

                    print(
                        f"Score     : "
                        f"{alert['score']:.6f}"
                    )

                    print(
                        f"Threshold : "
                        f"{THRESHOLD:.6f}"
                    )

                    print(
                        f"Timestamp : "
                        f"{alert['timestamp']}"
                    )

                    print(
                        f"Confirmed : "
                        f"{alert['count']} consecutive windows"
                    )

                    print("!" * 70)
                    print()

            except Exception as error:

                invalid_events += 1

                print(
                    f"WARNING: invalid event "
                    f"at line {line_number}: {error}",
                    file=sys.stderr
                )

    # Calculate windows
    total_windows = sum(
        state.windows
        for state in states.values()
    )

    print()
    print("=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)

    print(
        f"Events processed       : "
        f"{total_events}"
    )

    print(
        f"Windows scored         : "
        f"{total_windows}"
    )

    print(
        f"Anomaly alerts         : "
        f"{total_alerts}"
    )

    print(
        f"Invalid events         : "
        f"{invalid_events}"
    )

    print(
        f"Threshold              : "
        f"{THRESHOLD:.6f}"
    )

    print("=" * 70)

def process_live_file(input_file):
    states = defaultdict(PIDState)
    total_events = 0
    total_alerts = 0
    invalid_events = 0

    print()
    print("=" * 70)
    print("           LIVE SYSCALL ANOMALY DETECTOR")
    print("=" * 70)
    print(f"Model     : V2 LSTM Autoencoder")
    print(f"Device    : {DEVICE}")
    print(f"Threshold : {THRESHOLD:.6f}")
    print(f"Watching  : {input_file}")
    print("Press Ctrl+C to stop.")
    print("=" * 70, flush=True)

    try:
        with open(input_file, "r") as file:
            file.seek(0, 2)
            while True:
                line = file.readline()

                if not line:
                    time.sleep(0.1)
                    continue

                if not line.endswith("\n"):
                    # Wait until the writer finishes this record.
                    while not line.endswith("\n"):
                        time.sleep(0.05)
                        more = file.readline()
                        if more:
                            line += more
                    if not line.strip():
                        continue

                try:
                    event = json.loads(line)

                    if "pid" not in event or "syscall_id" not in event:
                        raise ValueError("missing pid or syscall_id")

                    alert = process_event(event, states)
                    total_events += 1

                    if alert:
                        total_alerts += 1
                        print()
                        print("!" * 70)
                        print("                 ANOMALY ALERT")
                        print("!" * 70)
                        print(f"PID       : {alert['pid']}")
                        print(f"Process   : {alert['process']}")
                        print(f"Score     : {alert['score']:.6f}")
                        print(f"Threshold : {THRESHOLD:.6f}")
                        print(f"Timestamp : {alert['timestamp']}")
                        print(f"Confirmed : {alert['count']} consecutive windows")
                        print("!" * 70, flush=True)

                except (ValueError, TypeError, KeyError) as error:
                    invalid_events += 1
                    print(
                        f"WARNING: invalid live event: {error}",
                        file=sys.stderr,
                        flush=True
                    )

    except KeyboardInterrupt:
        print("\nStopping live detector...")

    finally:
        total_windows = sum(
            state.windows for state in states.values()
        )
        print()
        print("=" * 70)
        print("LIVE DETECTION SUMMARY")
        print("=" * 70)
        print(f"Events processed : {total_events}")
        print(f"Windows scored  : {total_windows}")
        print(f"Anomaly alerts  : {total_alerts}")
        print(f"Invalid events  : {invalid_events}")
        print("=" * 70)


# ============================================================
# MAIN
# ============================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Linux syscall anomaly detector"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="JSONL syscall event file"
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Monitor new events appended to the file"
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        print(
            f"ERROR: file not found: {args.input_file}",
            file=sys.stderr
        )
        sys.exit(1)

    if args.follow:
        process_live_file(args.input_file)
    else:
        process_file(args.input_file)


if __name__ == "__main__":
    main()

