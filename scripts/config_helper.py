import yaml
import os
import re
import os
import yaml
import re

class ConfigLoader:
    """Handles configuration loading and path resolution."""

    @staticmethod
    def load_config(base_path, dataprep_path):
        """Loads and merges base and dataprep config files, resolving variables and absolute paths.

        Args:
            base_path (str): Path to base_config.yaml
            dataprep_path (str): Path to dataprep_config.yaml

        Returns:
            dict: Fully merged and processed config dict
        """
        with open(base_path, "r") as f:
            base_config = yaml.safe_load(f)

        with open(dataprep_path, "r") as f:
            dataprep_config = yaml.safe_load(f)

        # Merge configs - dataprep overrides base on conflicts
        config = ConfigLoader._deep_merge_dicts(base_config, dataprep_config)

        # Resolve variables and make absolute paths for 'paths' keys
        config["paths"] = ConfigLoader._process_paths(config["paths"], config)

        # Process colors, resolving references
        config["colors"] = ConfigLoader._process_colors(config["colors"], config)

        return config

    @staticmethod
    def _deep_merge_dicts(base, override):
        """Recursively merges two dicts, override takes precedence."""
        result = dict(base)
        for k, v in override.items():
            if (
                k in result
                and isinstance(result[k], dict)
                and isinstance(v, dict)
            ):
                result[k] = ConfigLoader._deep_merge_dicts(result[k], v)
            else:
                result[k] = v
        return result

    @staticmethod
    def _resolve_variables(value, reference_dict, depth=0):
        """Recursively resolve all ${var} patterns in the string using reference_dict."""
        if depth > 20:  # prevent infinite recursion
            return value
        if not isinstance(value, str):
            return value

        pattern = re.compile(r"\$\{([^\}]+)\}")
        matches = pattern.findall(value)
        for match in matches:
            keys = match.split(".")
            ref = reference_dict
            try:
                for k in keys:
                    ref = ref[k]
            except (KeyError, TypeError):
                # leave unresolved if key missing
                continue
            # Replace the ${...} with the resolved value (converted to str)
            value = value.replace(f"${{{match}}}", str(ref))

        # If after replacement there's still unresolved variables, recurse
        if "${" in value:
            return ConfigLoader._resolve_variables(value, reference_dict, depth + 1)
        return value

    @staticmethod
    def _process_paths(paths_dict, full_config):
        """Resolve variables in paths and convert to absolute paths."""
        processed = {}
        for key, val in paths_dict.items():
            resolved_val = ConfigLoader._resolve_variables(val, full_config)
            if isinstance(resolved_val, str) and resolved_val != "":
                processed[key] = os.path.abspath(resolved_val)
            else:
                processed[key] = resolved_val
        return processed

    @staticmethod
    def _process_colors(colors_dict, full_config):
        """Resolve color variables referencing others in colors dict."""
        processed = {}
        for k, v in colors_dict.items():
            resolved_val = ConfigLoader._resolve_variables(v, full_config)
            processed[k] = resolved_val
        return processed

