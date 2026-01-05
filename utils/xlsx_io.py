import pandas as pd
import os
from typing import Dict, Any

def write_dicts_to_xlsx(sheet_name_to_tabular_dict: Dict[str, Dict[Any, Dict[str, Any]]], filepath: str) -> None:
    try:
        os.makedirs(os.path.dirname(filepath))
    except FileExistsError:
        pass
    with pd.ExcelWriter(filepath) as writer:
        for sheet_name, tabular_dict in sheet_name_to_tabular_dict.items():
            df = pd.DataFrame.from_dict(tabular_dict, orient='index')
            df.index.name = "ID"
            df.reset_index(inplace=True)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return