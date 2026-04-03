import yaml

# For evaluating expression
import numpy as np
import math

def _eval_underscore_keys_in_nested_dict(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            processed_value = _eval_underscore_keys_in_nested_dict(value)
            
            target_key = key
            final_value = processed_value
            
            if isinstance(key, str) and key.startswith("_") and key.endswith("_"):
                if len(key) < 3:
                     raise ValueError(f"Invalid key format: '{key}'")
                
                target_key = key[1:-1]
                
                if target_key in new_dict:
                    raise KeyError(f"Conflict: Key '{target_key}' is defined twice (once as '{key}' and once as '{target_key}' or similar) in the dictionary.")
                
                if not isinstance(processed_value, str):
                     raise ValueError(f"Cannot evaluate non-string value for key '{key}'. Value: {processed_value}")

                try:
                    final_value = eval(processed_value)
                except Exception as e:
                    raise ValueError(f"Failed to evaluate key '{key}' (value: '{processed_value}'). Error: {e}")
            
            else:
                if target_key in new_dict:
                    raise KeyError(f"Conflict: Key '{target_key}' already exists in the dictionary.")
            
            new_dict[target_key] = final_value
            
        return new_dict
        
    elif isinstance(obj, list):
        return [_eval_underscore_keys_in_nested_dict(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_eval_underscore_keys_in_nested_dict(item) for item in obj)
    else:
        return obj
            
def safe_load(stream):
    return _eval_underscore_keys_in_nested_dict(yaml.safe_load(stream))