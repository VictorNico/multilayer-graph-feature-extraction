"""
CPU Core Usage Management

Author: VICTOR DJIEMBOU
Created: 15/11/2023
Last modified: 15/11/2023

Description:
    Configures and limits CPU core usage for scientific computing libraries
    (NumPy, SciPy, scikit-learn) to optimise performance and prevent system
    overload during experiments.

    Sets up:
    - OpenMP thread configuration for numerical compute backends
    - System signal handlers (SIGINT, SIGTERM) for graceful shutdown
    - Automatic resource cleanup on exit via atexit

Key features:
    - Dynamic thread count limiting via environment variables (OMP, MKL,
      OpenBLAS, NumExpr) read from the .env ``MAX_CORE`` setting
    - Clean interruption handling (Ctrl+C)
    - OpenMP/MKL/OpenBLAS thread pool release on exit

Usage:
    Imported at the top of pipeline scripts:
    from .cpu_limitation_usage import *

Dependencies:
    - modules.env: Environment configuration (.env loader)
    - os, signal, sys, atexit: System management
"""
# 00_cpu_limitation usage
from modules.env import *  # Env functions
import os
import signal

# Set thread limits BEFORE importing NumPy/SciPy — these libraries initialise
# their thread pools at import time, so env vars must be set first.

# Load MAX_CORE from .env
print(load_env_with_path()['max_core'])

# Set environment variables to cap parallel threads across compute backends:

# OMP_NUM_THREADS: OpenMP (used by many C/C++ libraries)
os.environ['OMP_NUM_THREADS'] = load_env_with_path()['max_core']

# MKL_NUM_THREADS: Intel Math Kernel Library (NumPy backend)
os.environ['MKL_NUM_THREADS'] = load_env_with_path()['max_core']

# OPENBLAS_NUM_THREADS: OpenBLAS (alternative NumPy backend)
os.environ['OPENBLAS_NUM_THREADS'] = load_env_with_path()['max_core']

# NUMEXPR_NUM_THREADS: NumExpr (fast numerical expression evaluation)
os.environ['NUMEXPR_NUM_THREADS'] = load_env_with_path()['max_core']

import sys  # Program exit management

def cleanup():
    """Explicitly release OpenMP and numerical-compute thread resources.

    Called automatically:
    - At normal program exit (via ``atexit``).
    - On receipt of SIGINT or SIGTERM.

    Reduces active thread pools to 1 to force release and prevent orphaned threads.
    """
    print("Cleaning up OpenMP threads...")

    # Attempt to release NumPy/MKL thread pool
    try:
        import mkl
        mkl.set_num_threads(1)  # Reduce to 1 thread to free the others
    except ImportError:
        pass  # MKL not installed or not the active backend

    # Attempt to release OpenBLAS thread pool
    try:
        import openblas
        openblas.set_num_threads(1)  # Reduce to 1 thread to free the others
    except ImportError:
        pass  # OpenBLAS not installed or not the active backend


def signal_handler(sig, frame):
    """Handle OS signals for graceful shutdown.

    Invoked by the OS when the process receives SIGINT (Ctrl+C) or SIGTERM.
    Logs the signal, releases compute resources via :func:`cleanup`, then exits.

    Args:
        sig (int): Signal number received (e.g. ``signal.SIGINT``).
        frame: Current execution frame at the point of interruption.
    """
    print(f'Signal {sig} - {frame} received, shutting down...')
    cleanup()   # Release compute resources
    sys.exit(0) # Clean exit


# Register signal handlers
# SIGINT: keyboard interrupt (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

# SIGTERM: termination signal (default kill command)
signal.signal(signal.SIGTERM, signal_handler)

# Register cleanup to run at normal program exit
import atexit

atexit.register(cleanup)  # cleanup() is called automatically when the program ends
