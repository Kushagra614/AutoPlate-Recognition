import json
import os
import logging
from pathlib import Path

class Config:
    """Configuration loader and manager for the AutoPlate Recognition system."""
    
    def __init__(self, config_path="./config/config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration from JSON file with fallback defaults."""
        default_config = {
            "models": {
                "vehicle_detection": {
                    "model_path": "yolov8n.pt",
                    "confidence_threshold": 0.5,
                    "vehicle_classes": [2, 3, 5, 7]
                },
                "license_plate": {
                    "model_path": "./models/license_plate_detector.pt",
                    "confidence_threshold": 0.6
                }
            },
            "tracking": {
                "max_age": 1,
                "min_hits": 3,
                "iou_threshold": 0.3
            },
            "video": {
                "output_fps": 30,
                "resize_width": 1280,
                "resize_height": 720
            },
            "logging": {
                "level": "INFO",
                "save_path": "./logs"
            },
            "results": {
                "save_path": "./results",
                "save_visualizations": True,
                "save_detections": True
            },
            "ocr": {
                "min_confidence": 0.5,
                "languages": ["en"],
                "gpu": False
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults (loaded config overrides defaults)
                config = self._deep_merge(default_config, loaded_config)
                logging.info(f"Configuration loaded from {self.config_path}")
                return config
            except Exception as e:
                logging.warning(f"Error loading config from {self.config_path}: {e}")
                logging.info("Using default configuration")
                return default_config
        else:
            logging.info(f"Config file not found at {self.config_path}, using defaults")
            return default_config
    
    def _deep_merge(self, default, override):
        """Deep merge two dictionaries."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation (e.g., 'models.vehicle_detection.confidence_threshold')."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_config(self, model_type):
        """Get model-specific configuration."""
        return self.get(f"models.{model_type}", {})
    
    def get_tracking_config(self):
        """Get tracking configuration."""
        return self.get("tracking", {})
    
    def get_video_config(self):
        """Get video processing configuration."""
        return self.get("video", {})
    
    def get_logging_config(self):
        """Get logging configuration."""
        return self.get("logging", {})
    
    def get_results_config(self):
        """Get results configuration."""
        return self.get("results", {})
    
    def get_ocr_config(self):
        """Get OCR configuration."""
        return self.get("ocr", {})
    
    def save_config(self, path=None):
        """Save current configuration to file."""
        save_path = path or self.config_path
        
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logging.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logging.error(f"Error saving config to {save_path}: {e}")
            raise
    
    def update(self, key_path, value):
        """Update configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        logging.debug(f"Updated config: {key_path} = {value}")
