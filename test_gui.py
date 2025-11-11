#!/usr/bin/env python
"""
Test BioSPPy GUI Installation
==============================

Tests GUI modules without requiring tkinter/display.
"""

import sys
import os

# Add project to path if running from source
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that core GUI modules can be imported (excluding tkinter-dependent ones)."""
    try:
        from biosppy.gui import signal_manager
        from biosppy.gui import plugins
        print("✓ Core GUI modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_signal_manager():
    """Test signal manager functionality."""
    try:
        from biosppy.gui.signal_manager import SignalManager
        import numpy as np

        manager = SignalManager()

        # Add a test signal
        signal = np.random.randn(1000)
        manager.add_signal('test', signal, 'ECG', 1000)

        # Retrieve signal
        data = manager.get_signal('test')
        assert data is not None, "Signal not retrieved"
        assert len(data['signal']) == 1000, "Signal length mismatch"
        assert data['type'] == 'ECG', "Signal type mismatch"
        assert data['sampling_rate'] == 1000, "Sampling rate mismatch"

        # Test list
        signals = manager.get_all_signals()
        assert 'test' in signals, "Signal not in list"

        # Add another signal
        manager.add_signal('test2', signal * 2, 'EDA', 500)
        assert len(manager.get_all_signals()) == 2, "Should have 2 signals"

        # Test remove and undo
        manager.remove_signal('test2')
        assert len(manager.get_all_signals()) == 1, "Should have 1 signal after removal"
        assert manager.undo(), "Undo should succeed"
        assert len(manager.get_all_signals()) == 2, "Should have 2 signals after undo"

        # Test redo
        assert manager.redo(), "Redo should succeed"
        assert len(manager.get_all_signals()) == 1, "Should have 1 signal after redo"

        # Test update signal
        new_signal = signal * 3
        manager.update_signal('test', new_signal, 'Filtered')
        updated_data = manager.get_signal('test')
        assert updated_data['processed'] == True, "Signal should be marked as processed"
        assert 'Filtered' in updated_data['processing_history'], "Processing step not recorded"

        print("✓ Signal manager tests passed")
        return True
    except Exception as e:
        print(f"✗ Signal manager error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plugin_manager():
    """Test plugin manager functionality."""
    try:
        from biosppy.gui.plugins import PluginManager
        import os

        # Create a mock main window
        class MockWindow:
            pass

        main_window = MockWindow()
        manager = PluginManager(main_window)

        # Test plugin directory management (create dir first)
        test_plugin_dir = '/tmp/test_plugins'
        os.makedirs(test_plugin_dir, exist_ok=True)
        manager.add_plugin_dir(test_plugin_dir)
        assert test_plugin_dir in manager.plugin_dirs, "Plugin dir not added"

        # Test discovery (won't find any, but should not crash)
        plugins = manager.discover_plugins()
        assert isinstance(plugins, list), "Should return a list"

        # Test list loaded plugins
        loaded = manager.list_loaded_plugins()
        assert isinstance(loaded, list), "Should return a list"

        print("✓ Plugin manager tests passed")
        return True
    except Exception as e:
        print(f"✗ Plugin manager error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all GUI files exist."""
    try:
        gui_dir = os.path.join(os.path.dirname(__file__), 'biosppy', 'gui')
        required_files = [
            '__init__.py',
            'main_window.py',
            'signal_manager.py',
            'plot_widget.py',
            'menubar.py',
            'toolbar.py',
            'status_bar.py',
            'context_menu.py',
            'dialogs.py',
            'plugins.py',
            'README.md',
            'INSTALLATION.md'
        ]

        missing = []
        for filename in required_files:
            filepath = os.path.join(gui_dir, filename)
            if not os.path.exists(filepath):
                missing.append(filename)

        if missing:
            print(f"✗ Missing files: {', '.join(missing)}")
            return False

        print("✓ All required GUI files exist")
        return True
    except Exception as e:
        print(f"✗ File structure error: {e}")
        return False

def test_python_syntax():
    """Test that all Python files compile."""
    try:
        import py_compile
        gui_dir = os.path.join(os.path.dirname(__file__), 'biosppy', 'gui')

        py_files = [f for f in os.listdir(gui_dir) if f.endswith('.py')]

        for filename in py_files:
            filepath = os.path.join(gui_dir, filename)
            try:
                py_compile.compile(filepath, doraise=True)
            except py_compile.PyCompileError as e:
                print(f"✗ Syntax error in {filename}: {e}")
                return False

        print(f"✓ All {len(py_files)} Python files have valid syntax")
        return True
    except Exception as e:
        print(f"✗ Syntax check error: {e}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("Testing BioSPPy GUI Installation")
    print("=" * 60)
    print()

    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("Module Imports", test_imports),
        ("Signal Manager", test_signal_manager),
        ("Plugin Manager", test_plugin_manager),
    ]

    results = []
    for name, test_func in tests:
        print(f"\nTest: {name}")
        print("-" * 60)
        results.append(test_func())

    print()
    print("=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n✓ All tests passed!")
        print("\nThe GUI code is correctly structured.")
        print("\nTo run the GUI (requires tkinter and display):")
        print("  python biosppy_gui.py")
        print("\nFor installation help:")
        print("  See biosppy/gui/INSTALLATION.md")
    else:
        print("\n✗ Some tests failed. Check the errors above.")
        sys.exit(1)
