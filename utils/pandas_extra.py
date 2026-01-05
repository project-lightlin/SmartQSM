import pandas as pd

def append(dataframe: pd.DataFrame) -> int:
    id: int = len(dataframe)
    dataframe.loc[id] = pd.NA
    return id