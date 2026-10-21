from bcc import BPF
import ctypes as ct
import json
import os
import sys
import time


OUTPUT_FILE = "data/raw/anomaly_syscalls_second.jsonl"


if len(sys.argv) != 2:
    print("Usage: sudo .venv/bin/python src/collector/collect_anomaly_dataset.py <PID>")
    sys.exit(1)


TARGET_PID = int(sys.argv[1])


BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>

struct syscall_event {
    u32 pid;
    char comm[16];
    u32 syscall_id;
};

BPF_PERF_OUTPUT(events);

TRACEPOINT_PROBE(raw_syscalls, sys_enter)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;

    if (pid != TARGET_PID)
        return 0;

    struct syscall_event event = {};

    event.pid = pid;

    bpf_get_current_comm(
        &event.comm,
        sizeof(event.comm)
    );

    event.syscall_id = args->id;

    events.perf_submit(
        args,
        &event,
        sizeof(event)
    );

    return 0;
}
"""


class SyscallEvent(ct.Structure):
    _fields_ = [
        ("pid", ct.c_uint32),
        ("comm", ct.c_char * 16),
        ("syscall_id", ct.c_uint32),
    ]


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


b = BPF(
    text=BPF_PROGRAM,
    cflags=[
        f"-DTARGET_PID={TARGET_PID}"
    ]
)


def handle_event(cpu, data, size):
    event = ct.cast(
        data,
        ct.POINTER(SyscallEvent)
    ).contents

    process_name = event.comm.decode(
        "utf-8",
        "replace"
    )

    record = {
        "timestamp": time.time(),
        "pid": int(event.pid),
        "process": process_name,
        "syscall_id": int(event.syscall_id)
    }

    with open(OUTPUT_FILE, "a") as file:
        file.write(
            json.dumps(record) + "\n"
        )


b["events"].open_perf_buffer(handle_event)


print("=" * 60)
print("     ISOLATED ANOMALOUS SYSCALL COLLECTION")
print("=" * 60)
print()
print(f"Target PID : {TARGET_PID}")
print(f"Output     : {OUTPUT_FILE}")
print()
print("Collector is running.")
print("Run the controlled workload for this PID.")
print("Press Ctrl+C here when finished.")
print()


try:
    while True:
        b.perf_buffer_poll()

except KeyboardInterrupt:
    print()
    print("Collection stopped.")
