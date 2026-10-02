import json
from typing import List, Any
from pydantic import BaseModel

def write_jsonl(filepath: str, data: List[Any]):
    """Writes a list of objects (dicts or pydantic models) to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            if isinstance(item, BaseModel):
                # model_dump_json serializes nested models properly
                f.write(item.model_dump_json() + '\n')
            else:
                json.dump(item, f)
                f.write('\n')

def read_jsonl(filepath: str, cls: type = None) -> List[Any]:
    """Reads a JSONL file. If cls (Pydantic model) is provided, instantiates it."""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if cls and issubclass(cls, BaseModel):
                # model_validate_json handles nested dicts perfectly
                results.append(cls.model_validate_json(line))
            else:
                results.append(json.loads(line.strip()))
    return results
