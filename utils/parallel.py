from typing import Any, Optional
import os
from joblib import Parallel
import warnings
try:
    import joblib.externals.loky.reusable_executor as reusable_executor
except ImportError:
    reusable_executor = None

def parallelize(
        iterable: Any,
        chunk_size: int,
        n_jobs: Optional[int] = None
) -> Any:
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="joblib") 
        if n_jobs == None:
            n_jobs = max(os.cpu_count() // 2, 1)
        
        with Parallel(
            n_jobs=n_jobs,
            batch_size=chunk_size,
            prefer="processes"
        ) as parallel:
            result = parallel(iterable)

    if reusable_executor:
        try:
            executor = reusable_executor.get_reusable_executor()
            executor.shutdown(wait=True, kill_workers=True)
        except Exception as e:
            pass 

    return result