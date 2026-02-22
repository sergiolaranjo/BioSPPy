"""
Plugin System
=============

Modular plugin architecture for extending GUI functionality.
"""

import importlib
import os
import sys
from pathlib import Path


class PluginManager:
    """Manager for loading and executing plugins.

    Plugins can extend the GUI with:
    - New signal processing algorithms
    - Custom visualizations
    - Additional analysis methods
    - Export formats
    - Data importers

    Plugin Structure:
    -----------------
    Each plugin is a Python module with:

    1. plugin_info dict with metadata:
       - name: Plugin name
       - version: Version string
       - author: Author name
       - description: Short description
       - category: 'processing', 'visualization', 'analysis', 'import', 'export'

    2. register(main_window) function:
       Called when plugin loads, can add menu items, buttons, etc.

    3. unregister(main_window) function:
       Called when plugin unloads

    Example Plugin:
    ---------------
    ```python
    # my_plugin.py

    plugin_info = {
        'name': 'Custom Filter',
        'version': '1.0',
        'author': 'John Doe',
        'description': 'Apply custom filtering algorithm',
        'category': 'processing'
    }

    def register(main_window):
        # Add menu item
        main_window.menubar.process_menu.add_command(
            label="Custom Filter",
            command=lambda: apply_custom_filter(main_window)
        )

    def unregister(main_window):
        # Clean up if needed
        pass

    def apply_custom_filter(main_window):
        # Implementation
        pass
    ```
    """

    def __init__(self, main_window):
        """Initialize plugin manager.

        Parameters
        ----------
        main_window : BioSPPyGUI
            Reference to main window.
        """
        self.main_window = main_window
        self.plugins = {}  # name -> plugin module
        self.plugin_dirs = []

        # Default plugin directories
        self._add_default_plugin_dirs()

    def _add_default_plugin_dirs(self):
        """Add default plugin directories."""
        # User plugins directory
        user_home = Path.home()
        user_plugins = user_home / '.biosppy' / 'plugins'
        if user_plugins.exists():
            self.plugin_dirs.append(str(user_plugins))

        # System plugins directory
        system_plugins = Path(__file__).parent / 'plugins'
        if system_plugins.exists():
            self.plugin_dirs.append(str(system_plugins))

    def add_plugin_dir(self, directory):
        """Add a plugin directory.

        Parameters
        ----------
        directory : str or Path
            Directory containing plugins.
        """
        directory = str(directory)
        if os.path.isdir(directory) and directory not in self.plugin_dirs:
            self.plugin_dirs.append(directory)
            if directory not in sys.path:
                sys.path.insert(0, directory)

    def discover_plugins(self):
        """Discover all available plugins.

        Returns
        -------
        list
            List of discovered plugin names.
        """
        discovered = []

        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue

            for item in os.listdir(plugin_dir):
                item_path = os.path.join(plugin_dir, item)

                # Check if it's a Python file
                if item.endswith('.py') and not item.startswith('_'):
                    plugin_name = item[:-3]
                    discovered.append((plugin_name, item_path))

                # Check if it's a package
                elif os.path.isdir(item_path):
                    init_file = os.path.join(item_path, '__init__.py')
                    if os.path.exists(init_file):
                        discovered.append((item, item_path))

        return discovered

    def load_plugin(self, plugin_name):
        """Load a plugin.

        Parameters
        ----------
        plugin_name : str
            Name of plugin to load.

        Returns
        -------
        bool
            True if loaded successfully.
        """
        if plugin_name in self.plugins:
            return True  # Already loaded

        try:
            # Import the plugin module
            module = importlib.import_module(plugin_name)

            # Validate plugin structure
            if not hasattr(module, 'plugin_info'):
                raise ValueError(f"Plugin {plugin_name} missing 'plugin_info'")

            if not hasattr(module, 'register'):
                raise ValueError(f"Plugin {plugin_name} missing 'register' function")

            # Register the plugin
            module.register(self.main_window)

            # Store plugin
            self.plugins[plugin_name] = module

            return True

        except Exception as e:
            print(f"Failed to load plugin {plugin_name}: {e}")
            return False

    def unload_plugin(self, plugin_name):
        """Unload a plugin.

        Parameters
        ----------
        plugin_name : str
            Name of plugin to unload.

        Returns
        -------
        bool
            True if unloaded successfully.
        """
        if plugin_name not in self.plugins:
            return False

        try:
            module = self.plugins[plugin_name]

            # Unregister if function exists
            if hasattr(module, 'unregister'):
                module.unregister(self.main_window)

            # Remove from plugins
            del self.plugins[plugin_name]

            return True

        except Exception as e:
            print(f"Failed to unload plugin {plugin_name}: {e}")
            return False

    def get_plugin_info(self, plugin_name):
        """Get plugin information.

        Parameters
        ----------
        plugin_name : str
            Plugin name.

        Returns
        -------
        dict or None
            Plugin info dictionary or None if not loaded.
        """
        if plugin_name in self.plugins:
            return self.plugins[plugin_name].plugin_info
        return None

    def list_loaded_plugins(self):
        """List all loaded plugins.

        Returns
        -------
        list
            List of loaded plugin names.
        """
        return list(self.plugins.keys())

    def load_all_plugins(self):
        """Discover and load all available plugins."""
        discovered = self.discover_plugins()

        for plugin_name, plugin_path in discovered:
            self.load_plugin(plugin_name)


# Example built-in plugins
class ExamplePlugin:
    """Example plugin demonstrating the plugin system."""

    plugin_info = {
        'name': 'Example Plugin',
        'version': '1.0.0',
        'author': 'BioSPPy Team',
        'description': 'Example plugin for demonstration',
        'category': 'processing'
    }

    @staticmethod
    def register(main_window):
        """Register plugin with main window."""
        # Could add menu items, toolbar buttons, etc.
        pass

    @staticmethod
    def unregister(main_window):
        """Unregister plugin from main window."""
        pass
