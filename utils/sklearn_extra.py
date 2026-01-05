import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.exceptions import ConvergenceWarning
import warnings
import pandas as pd
from sklearn.model_selection import cross_val_score
from typing import List, Union, Any, Callable

def determine_optimal_components(
        X: np.ndarray,
        max_num_components: int = 10,
        use_cv_score: bool = False,
        use_bic: bool = True,
        use_aic: bool = False,
        use_log_likelihood: bool = False,
        cv: int = 5,
        scoring: Union[str, Callable] = lambda estimator, X, y=None: estimator.score(X),
        **kwargs
):
    columns: List[str] = ["num_components", "gmm"]
    
    if use_cv_score:
        columns.append('cv_score')
    if use_bic:
        columns.append('bic')
    if use_aic:
        columns.append('aic')
    if use_log_likelihood:
        columns.append('log_likelihood')
    
    max_num_components = min(max_num_components, len(X))
    if use_cv_score:
        max_num_components = min(max_num_components, int(np.floor(len(X) * (cv-1) / cv).astype(int)))
    
    data: List[List[Any]] = []
    
    for k in range(1, max_num_components + 1):
        gmm: GaussianMixture = GaussianMixture(
            n_components=k,
            **kwargs
        )

        row: List[Any] = [k, gmm]

        if use_cv_score:
            cv_score = cross_val_score(
                gmm, 
                X, 
                cv=cv, 
                scoring=scoring
            ).mean()
            row.append(cv_score)
        
        gmm.fit(X)

        if use_bic:
            row.append(gmm.bic(X))
        if use_aic:
            row.append(gmm.aic(X))
        if use_log_likelihood:
            row.append(gmm.score(X))

        data.append(row)

    return pd.DataFrame(data, columns=columns)

def calculate_best_gaussian_mixture(
        X: np.ndarray,
        auto: bool = False,
        max_num_components: int = 1,
        **kwargs
) -> Union[GaussianMixture, BayesianGaussianMixture]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning) 
        if auto:
            bgmm: BayesianGaussianMixture = BayesianGaussianMixture(
                n_components=min(max_num_components, len(X)),         
                **kwargs
            )
            bgmm.fit(X)
            return bgmm
        else:
            data_frame: pd.DataFrame = determine_optimal_components(
                X,
                max_num_components=max_num_components,
                use_cv_score=False,
                use_bic=True,
                use_aic=False,
                use_log_likelihood=False,
                **kwargs
            )
            best_gmm: GaussianMixture = data_frame['gmm'].iloc[data_frame['bic'].idxmin()]
            return best_gmm
