# ============================================================
# SYSCALL TRACER
# ============================================================
#
# This program:
#
# Linux kernel
#      ↓
#    eBPF
#      ↓
# raw syscall events
#      ↓
#    Python
#      ↓
# terminal output
#
# ============================================================


from bcc import BPF
import ctypes as ct

from syscall_mapper import syscall_name


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

    // Linux syscall number
    u32 syscall_id;
};


// Create a perf event buffer
BPF_PERF_OUTPUT(events);


// Trace syscall entry
TRACEPOINT_PROBE(raw_syscalls, sys_enter)
{

    // Create an empty event
    struct syscall_event event = {};


    // Get process ID
    event.pid = bpf_get_current_pid_tgid() >> 32;


    // Get process name
    bpf_get_current_comm(
        &event.comm,
        sizeof(event.comm)
    );


    // Get syscall number
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

b = BPF(text=BPF_PROGRAM)

print("eBPF program loaded successfully.")
print()


# ============================================================
# PYTHON REPRESENTATION OF syscall_event
# ============================================================

class SyscallEvent(ct.Structure):

    _fields_ = [

        # Process ID
        ("pid", ct.c_uint),

        # Process name
        ("comm", ct.c_char * 16),

        # Syscall number
        ("syscall_id", ct.c_uint),
    ]


# ============================================================
# EVENT CALLBACK
# ============================================================

def handle_event(cpu, data, size):

    # Convert raw event data into Python object
    event = ct.cast(
        data,
        ct.POINTER(SyscallEvent)
    ).contents


    # Convert process name from bytes to string
    process_name = event.comm.decode(
        "utf-8",
        "replace"
    )


    # Convert syscall ID to syscall name
    name = syscall_name(
        event.syscall_id
    )


    # Print event
    print(
        f"PID={event.pid:<8} "
        f"PROCESS={process_name:<16} "
        f"SYSCALL_ID={event.syscall_id:<4} "
        f"SYSCALL={name}",
        flush=True
    )


# ============================================================
# CONNECT PERF BUFFER
# ============================================================

b["events"].open_perf_buffer(
    handle_event
)


# ============================================================
# START TRACER
# ============================================================

print("==============================================")
print("       SYSCALL ANOMALY DETECTOR")
print("==============================================")
print()
print("Starting syscall tracer...")
print("Listening for Linux syscalls...")
print()
print("Press Ctrl+C to stop.")
print()


# ============================================================
# MAIN EVENT LOOP
# ============================================================

try:

    while True:

        b.perf_buffer_poll()


except KeyboardInterrupt:

    print()
    print("==============================================")
    print("Syscall tracer stopped.")
    print("==============================================")