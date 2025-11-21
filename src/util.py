# util.py
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.tree import DecisionTreeClassifier


def compute_cuts(X, y, max_leaf_nodes=3, random_state=42):
    """
    Encontra pontos de corte ótimos usando uma árvore de decisão.
    Retorna um array de thresholds ordenados.
    """
    arvore = DecisionTreeClassifier(
        criterion="entropy",
        max_leaf_nodes=max_leaf_nodes,
        random_state=random_state,
    )
    arvore.fit(X, y)
    thresholds = arvore.tree_.threshold  # type: ignore
    thresholds = thresholds[thresholds != -2]
    thresholds = np.sort(thresholds)
    return thresholds


def mutual_information(col: pd.Series, target: pd.Series, max_leaf_nodes=3):
    """
    Calcula MI entre uma variável e o alvo (target).
    - Se col for categórica/objeto: usa diretamente mutual_info_score.
    - Se col for numérica: discretiza em bins definidos por árvore de decisão.
    """
    col = col.dropna()
    target = target.loc[col.index]

    # Categórica
    if col.dtype == "object" or col.dtype.name == "category":
        return mutual_info_score(col, target)

    # Numérica (discretiza com árvore)
    try:
        thresholds = compute_cuts(col.to_frame(), target, max_leaf_nodes=max_leaf_nodes)
        col_bin = pd.cut(col, bins=[-np.inf] + thresholds.tolist() + [np.inf])
        return mutual_info_score(col_bin, target)
    except Exception:
        return None
