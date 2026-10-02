import os
import pytest
from common.config import load_config, merge_configs
import tempfile
import yaml

def test_load_config():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({"pipeline": {"seed": 42}}, f)
        temp_path = f.name
        
    try:
        config = load_config(temp_path)
        assert config["pipeline"]["seed"] == 42
    finally:
        os.remove(temp_path)
        
def test_load_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_file.yaml")

def test_merge_configs():
    base = {"db": {"host": "localhost", "port": 5432}, "log_level": "INFO"}
    override = {"db": {"port": 5433}, "new_key": "value"}
    
    merged = merge_configs(base, override)
    
    assert merged["db"]["host"] == "localhost"
    assert merged["db"]["port"] == 5433
    assert merged["log_level"] == "INFO"
    assert merged["new_key"] == "value"
