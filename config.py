import yaml
import os

class Config:
    def __init__(self, yaml_data):
        if isinstance(yaml_data, (str, bytes, os.PathLike)):
            with open(yaml_data, 'r') as file:
                self.__config = yaml.safe_load(file)
        elif isinstance(yaml_data, dict):
            self.__config = yaml_data
        else:
            raise TypeError("Expected a file path, bytes, os.PathLike or a dictionary")

    def __getattr__(self, name):
        if name in self.__config:
            return self.__config[name]
        else:
            raise AttributeError(f"Config attribute '{name}' not found")

