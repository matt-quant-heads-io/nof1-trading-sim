import os
import yaml
import logging
from typing import Dict, Any, Optional, Union, List


class ConfigAccessor:
    """
    Helper class to access nested configuration values using attribute notation.
    Allows for more intuitive access like config.system.mode instead of config.get('system.mode')
    """
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        
    def __getattr__(self, name: str) -> Any:
        """
        Access configuration values using attribute notation.
        
        Args:
            name: Configuration key
            
        Returns:
            Configuration value or another ConfigAccessor for nested dictionaries
        """
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return ConfigAccessor(value)
            return value
        raise AttributeError(f"Configuration has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """
        Access configuration values using dictionary notation.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
        """
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with a default fallback.
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)
    
    def items(self):
        """
        Return key-value pairs from the configuration.
        
        Returns:
            Iterator over key-value pairs
        """
        return self._config.items()
    
    def keys(self):
        """
        Return keys from the configuration.
        
        Returns:
            Iterator over keys
        """
        return self._config.keys()
    
    def values(self):
        """
        Return values from the configuration.
        
        Returns:
            Iterator over values
        """
        return self._config.values()
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in the configuration.
        
        Args:
            key: Configuration key
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self._config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration accessor back to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return self._config


class ConfigManager:
    """
    Manages configuration for the trading simulation system.
    Loads configuration from YAML files and provides access to configuration parameters 
    using both attribute notation and dictionary-style access.
    """
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file. If None, uses default config.
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or os.path.join("config", "default_config.yaml")
        self._config_dict = self._load_config()
        self.config = ConfigAccessor(self._config_dict)
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict containing configuration parameters.
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {self.config_path}: {e}")
            raise
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value for the given key path.
        
        Args:
            key_path: Configuration key in dot notation (e.g., 'system.mode')
            default: Default value to return if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config_dict
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value for the given key path.
        
        Args:
            key_path: Configuration key in dot notation (e.g., 'system.mode')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config_dict
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
                
        config[keys[-1]] = value
        
        # Refresh the accessor
        self.config = ConfigAccessor(self._config_dict)
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to a YAML file.
        
        Args:
            path: Path to save the configuration. If None, uses the current config path.
        """
        save_path = path or self.config_path
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self._config_dict, f, default_flow_style=False)
            self.logger.info(f"Saved configuration to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {save_path}: {e}")
            raise
    
    def merge_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Merge the given configuration dictionary with the current configuration.
        
        Args:
            config_dict: Configuration dictionary to merge
        """
        def _recursive_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    _recursive_update(d[k], v)
                else:
                    d[k] = v
        
        _recursive_update(self._config_dict, config_dict)
        # Refresh the accessor
        self.config = ConfigAccessor(self._config_dict)
        self.logger.info("Merged configuration with provided dictionary")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Return the full configuration as a dictionary.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return self._config_dict
    
    def keys(self, nested: bool = False) -> List[str]:
        """
        Get the top-level keys or all nested keys in the configuration.
        
        Args:
            nested: If True, returns all nested keys with dot notation
            
        Returns:
            List of keys
        """
        if not nested:
            return list(self._config_dict.keys())
        
        def _get_nested_keys(d, prefix=''):
            keys = []
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                keys.append(key)
                if isinstance(v, dict):
                    keys.extend(_get_nested_keys(v, key))
            return keys
        
        return _get_nested_keys(self._config_dict)
    
    def __getattr__(self, name: str) -> Any:
        """
        Allow direct attribute access to top-level config items.
        
        Args:
            name: Attribute name
            
        Returns:
            Configuration value or AttributeError
        """
        if name in self._config_dict:
            return self.config.__getattr__(name)
        raise AttributeError(f"ConfigManager has no attribute '{name}'")