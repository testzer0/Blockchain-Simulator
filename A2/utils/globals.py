"""
Adithya Bhaskar, 2023.
This file contains global variables that are not configuration parameters.
"""

import os

## Directories

LOG_DIR = "logs/"

def get_next_log_number(log_dir=LOG_DIR):
    max_log_number = 0
    for d in os.listdir(LOG_DIR):
        if d.startswith("run"):
            try:
                log_number = int(d[3:])
                max_log_number = max(max_log_number, log_number)
            except:
                pass
    return max_log_number+1

## -- Directories

## General
CODE_ROOT = "./"
GLOBAL_TIME = 0

def increment_global_time():
    global GLOBAL_TIME
    GLOBAL_TIME += 1

def get_global_time():
    global GLOBAL_TIME
    return GLOBAL_TIME

def inc_and_return_global_time():
    global GLOBAL_TIME
    GLOBAL_TIME += 1
    return GLOBAL_TIME

def set_and_return_global_time(time):
    global GLOBAL_TIME
    GLOBAL_TIME = time
    return GLOBAL_TIME

## -- General

## Conversion factors

TICK_DURATION_IN_SECONDS = 0.001

def seconds_to_ticks(seconds):
    return int(seconds/TICK_DURATION_IN_SECONDS)

def ticks_to_seconds(ticks):
    return ticks * TICK_DURATION_IN_SECONDS

MILLISECOND_IN_TICKS = seconds_to_ticks(0.001)

BIT_SIZE = 1
BYTE_SIZE = 8
KBIT_SIZE = 1024
MBIT_SIZE = 1024 * KBIT_SIZE
GBIT_SIZE = 1024 * MBIT_SIZE

KBPS = KBIT_SIZE * TICK_DURATION_IN_SECONDS
MBPS = MBIT_SIZE * TICK_DURATION_IN_SECONDS
GBPS = GBIT_SIZE * TICK_DURATION_IN_SECONDS

## --Conversion factors

if __name__ == '__main__':
    pass