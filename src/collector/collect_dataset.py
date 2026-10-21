# ============================================================
# SYSCALL DATASET COLLECTOR
# ============================================================
#
# Collects Linux syscall events using eBPF and saves them to:
#
# data/raw/syscalls.jsonl
#
# Each line is one JSON record.
#
# Example:
#
# {"timestamp": 1234567890.12,
#  "pid": 1234,
#  "process": "bash",
#  "syscall_id": 257,
#  "syscall": "openat"}
#
# ============================================================

from bcc import BPF
import ctypes as ct
import json
import os
import time

from syscall_mapper import syscall_name


# ============================================================
# OUTPUT FILE
# ============================================================

OUTPUT_FILE = os.environ.get(
    "SYSCALL_OUTPUT_FILE",
    "data/raw/syscalls.jsonl"
)
# Create directory if it does not exist
os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# ============================================================
# eBPF PROGRAM
# ============================================================

BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>


// Structure shared between eBPF and Python
struct syscall_event {

    // Process ID
    u32 pid;

    // Process name
    char comm[16];

    // Syscall number
    u32 syscall_id;
};


// Perf event buffer
BPF_PERF_OUTPUT(events);


// Trace syscall entry
TRACEPOINT_PROBE(raw_syscalls, sys_enter)
{

    struct syscall_event event = {};


    // Get process ID
    event.pid = bpf_get_current_pid_tgid() >> 32;


    // Get process name
    bpf_get_current_comm(
        &event.comm,
        sizeof(event.comm)
    );


    // Get syscall ID
    event.syscall_id = args->id;


    // Send event to Python
    events.perf_submit(
        args,
        &event,
        sizeof(event)
    );


    return 0;
}
"""


# ============================================================
# CREATE BPF OBJECT
# ============================================================

print("Loading eBPF program...")

b = BPF(
    text=BPF_PROGRAM
)

print("eBPF program loaded successfully.")
print()


# ============================================================
# PYTHON EVENT STRUCTURE
# ============================================================

class SyscallEvent(ct.Structure):

    _fields_ = [

        ("pid", ct.c_uint),

        ("comm", ct.c_char * 16),

        ("syscall_id", ct.c_uint),
    ]


# ============================================================
# OPEN DATASET FILE
# ============================================================

print(f"Saving syscall data to: {OUTPUT_FILE}")
print()

output_file = open(
    OUTPUT_FILE,
    "a",
    buffering=1
)


# ============================================================
# COUNTER
# ============================================================

event_count = 0


# ============================================================
# EVENT CALLBACK
# ============================================================

def handle_event(cpu, data, size):

    global event_count

    # Convert event data
    event = ct.cast(
        data,
        ct.POINTER(SyscallEvent)
    ).contents


    # Convert process name
    process_name = event.comm.decode(
        "utf-8",
        "replace"
    )


    # Get syscall name
    name = syscall_name(
        event.syscall_id
    )


    # Current timestamp
    timestamp = time.time()


    # Create Python dictionary
    record = {

        "timestamp": timestamp,

        "pid": int(event.pid),

        "process": process_name,

        "syscall_id": int(event.syscall_id),

        "syscall": name
    }


    # Convert dictionary to JSON
    json_record = json.dumps(
        record
    )


    # Write one JSON object per line
    output_file.write(
        json_record + "\n"
    )


    # Increase counter
    event_count += 1


    # Show progress every 100 events
    if event_count % 100 == 0:

        print(
            f"Collected {event_count} syscall events...",
            flush=True
        )


# ============================================================
# CONNECT PERF BUFFER
# ============================================================

b["events"].open_perf_buffer(
    handle_event
)


# ============================================================
# START COLLECTION
# ============================================================

print("==============================================")
print("       SYSCALL DATASET COLLECTOR")
print("==============================================")
print()
print("Collecting Linux syscall events...")
print()
print("Output file:")
print(OUTPUT_FILE)
print()
print("Press Ctrl+C to stop.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        b.perf_buffer_poll()


except KeyboardInterrupt:

    print()
    print("Stopping collector...")


finally:

    output_file.close()

    print()
    print("==============================================")
    print("Collection finished.")
    print(f"Total events collected: {event_count}")
    print(f"Dataset saved to: {OUTPUT_FILE}")
    print("==============================================")
