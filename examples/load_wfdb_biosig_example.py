#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BioSPPy - Load WFDB and BioSig Files Example
---------------------------------------------

This example demonstrates how to load biomedical signal files using
the new WFDB (PhysioNet) and BioSig format loaders.

:copyright: (c) 2015-2024 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

from biosppy import storage
from biosppy.signals import ecg
import numpy as np


def example_load_wfdb():
    """Example: Load WFDB format files from PhysioNet."""

    print("=" * 60)
    print("WFDB (PhysioNet) Format Loading Example")
    print("=" * 60)

    # Example 1: Load all channels from a WFDB record
    # Note: You need to have a WFDB file available
    # Download example from: https://physionet.org/content/mitdb/1.0.0/

    try:
        # Load entire record (all channels)
        signals, mdata = storage.load_wfdb('path/to/record')

        print("\n1. Loading all channels:")
        print(f"   Number of signals: {mdata['num_signals']}")
        print(f"   Number of samples: {mdata['num_samples']}")
        print(f"   Sampling rate: {mdata['sampling_rate']} Hz")
        print(f"   Signal names: {mdata['labels']}")
        print(f"   Units: {mdata['units']}")
        print(f"   Signal shape: {signals.shape}")

    except Exception as e:
        print(f"\n1. Loading all channels failed: {e}")
        print("   (This is expected if you don't have a WFDB file)")


    # Example 2: Load specific channel by index
    try:
        signal, mdata = storage.load_wfdb('path/to/record', channel=0)

        print("\n2. Loading channel 0:")
        print(f"   Signal shape: {signal.shape}")
        print(f"   Channel name: {mdata['labels'][0]}")
        print(f"   Unit: {mdata['units'][0]}")

    except Exception as e:
        print(f"\n2. Loading channel 0 failed: {e}")


    # Example 3: Load specific channel by name
    try:
        signal, mdata = storage.load_wfdb('path/to/record', channel='MLII')

        print("\n3. Loading channel 'MLII':")
        print(f"   Signal shape: {signal.shape}")

    except Exception as e:
        print(f"\n3. Loading channel by name failed: {e}")


    # Example 4: Load multiple channels
    try:
        signals, mdata = storage.load_wfdb('path/to/record', channel=[0, 1])

        print("\n4. Loading channels 0 and 1:")
        print(f"   Signals shape: {signals.shape}")
        print(f"   Channel names: {mdata['labels']}")

    except Exception as e:
        print(f"\n4. Loading multiple channels failed: {e}")


    # Example 5: Process loaded ECG signal
    try:
        # Load ECG signal
        ecg_signal, mdata = storage.load_wfdb('path/to/ecg_record', channel=0)

        # Process with BioSPPy ECG module
        output = ecg.ecg(
            signal=ecg_signal,
            sampling_rate=mdata['sampling_rate'],
            show=False
        )

        print("\n5. Processing ECG signal:")
        print(f"   Number of R-peaks detected: {len(output['rpeaks'])}")
        print(f"   Average heart rate: {np.mean(output['heart_rate']):.2f} bpm")

    except Exception as e:
        print(f"\n5. Processing ECG failed: {e}")


def example_load_biosig():
    """Example: Load various formats with BioSig."""

    print("\n\n" + "=" * 60)
    print("BioSig Multi-Format Loading Example")
    print("=" * 60)

    # BioSig supports many formats: EDF, GDF, BDF, BrainVision, etc.

    # Example 1: Load EDF file
    try:
        signals, mdata = storage.load_biosig('path/to/file.edf')

        print("\n1. Loading EDF file:")
        print(f"   Number of signals: {mdata['num_signals']}")
        print(f"   Number of samples: {mdata['num_samples']}")
        print(f"   Sampling rate: {mdata['sampling_rate']} Hz")
        print(f"   File type: {mdata['file_type']}")
        if 'labels' in mdata:
            print(f"   Signal names: {mdata['labels']}")

    except Exception as e:
        print(f"\n1. Loading EDF file failed: {e}")
        print("   (This is expected if you don't have a BioSig-compatible file)")


    # Example 2: Load GDF file with specific channel
    try:
        signal, mdata = storage.load_biosig('path/to/file.gdf', channel=0)

        print("\n2. Loading GDF file (channel 0):")
        print(f"   Signal shape: {signal.shape}")
        if 'labels' in mdata:
            print(f"   Channel name: {mdata['labels'][0]}")

    except Exception as e:
        print(f"\n2. Loading GDF file failed: {e}")


    # Example 3: Load multiple channels from BDF file
    try:
        signals, mdata = storage.load_biosig('path/to/file.bdf', channel=[0, 2, 3])

        print("\n3. Loading BDF file (channels 0, 2, 3):")
        print(f"   Signals shape: {signals.shape}")
        if 'labels' in mdata:
            print(f"   Channel names: {mdata['labels']}")

    except Exception as e:
        print(f"\n3. Loading BDF file failed: {e}")


    # Example 4: Compare with native EDF loader
    try:
        # Load with BioSig
        signals_biosig, mdata_biosig = storage.load_biosig('path/to/file.edf')

        # Load with native EDF loader
        signals_edf, mdata_edf = storage.load_edf('path/to/file.edf')

        print("\n4. Comparing BioSig vs native EDF loader:")
        print(f"   BioSig shape: {signals_biosig.shape}")
        print(f"   Native EDF shape: {signals_edf.shape}")
        print(f"   Arrays equal: {np.allclose(signals_biosig, signals_edf)}")

    except Exception as e:
        print(f"\n4. Comparison failed: {e}")


def installation_info():
    """Print installation information."""

    print("\n\n" + "=" * 60)
    print("Installation Information")
    print("=" * 60)

    print("\nTo use these features, install the optional dependencies:")
    print("\n  For WFDB support:")
    print("    pip install wfdb")
    print("    # or")
    print("    pip install biosppy[wfdb]")

    print("\n  For BioSig support:")
    print("    pip install pyBioSig")
    print("    # or")
    print("    pip install biosppy[biosig]")

    print("\n  For all optional dependencies:")
    print("    pip install biosppy[all]")
    print("    # or")
    print("    pip install -r requirements-optional.txt")

    print("\n" + "=" * 60)
    print("\nSupported formats:")
    print("  WFDB: .dat, .hea (PhysioNet format)")
    print("  BioSig: EDF, EDF+, GDF, BDF, BDF+, BrainVision,")
    print("          BioSemi, and many others")
    print("=" * 60)


if __name__ == '__main__':
    # Print installation information
    installation_info()

    # Run examples
    example_load_wfdb()
    example_load_biosig()

    print("\n\nNote: Most examples will fail unless you have actual signal files.")
    print("Download example WFDB files from: https://physionet.org/")
