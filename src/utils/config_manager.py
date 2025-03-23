import os
import yaml
import logging
from typing import Dict, Any, Optional

class ConfigManager:
    """
    Manages configuration for the trading simulation system.
    Loads configuration from YAML files and provides access to configuration parameters.
    """
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file. If None, uses default config.
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or os.path.join("config", "default_config.yaml")
        self.config = self._load_config()
        
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
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value for the given key.
        
        Args:
            key: Configuration key in dot notation (e.g., 'system.mode')
            default: Default value to return if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value for the given key.
        
        Args:
            key: Configuration key in dot notation (e.g., 'system.mode')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
                
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to a YAML file.
        
        Args:
            path: Path to save the configuration. If None, uses the current config path.
        """
        save_path = path or self.config_path
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
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
        
        _recursive_update(self.config, config_dict)
        self.logger.info("Merged configuration with provided dictionary")