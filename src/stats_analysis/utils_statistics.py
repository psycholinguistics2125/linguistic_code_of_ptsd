""" 
creadted_at: 2022, nov 5
description: this file regroups all statistical tools functions uses in this project
"""


from scipy.stats import ranksums
import pandas as pd
import itertools


import pingouin as pg
from pingouin import power_ttest2n
from pingouin import power_anova

from scipy.stats import ranksums


def compute_khi_2_table(
    data: pd.DataFrame, cible: list, col_list: list, seuil=0.01
) -> pd.DataFrame:
    """compute khi 2 stats for two sets of columns

    Args:
        data (pd.DataFrame): _description_
        cible (list): _description_
        col_list (list): _description_
        seuil (float, optional): _description_. Defaults to 0.01.

    Returns:
        pd.DataFrame: _description_
    """
    i = 0
    df_result = pd.DataFrame(
        columns=["x", "y", "chi2", "dof", "pval", "cramer", "power"]
    )
    for elt in list(itertools.product(cible, col_list)):
        x = elt[0]
        y = elt[1]
        expected, observed, stats = pg.chi2_independence(data, x=x, y=y)
        result = stats[stats["test"] == "pearson"][
            ["lambda", "chi2", "dof", "pval", "cramer", "power"]
        ].to_dict(orient="records")[0]
        result["x"] = x
        result["y"] = y
        if result["pval"] <= seuil:
            df_result.loc[i] = result
            i = i + 1
    return df_result


def compute_wilcoxon(data: pd.DataFrame, x: str, y: str, alt="two-sided") -> dict:
    """compute wilcowon test beween the col x and y

    Args:
        data (pd.DataFrame): _description_
        x (str): _description_
        y (str): _description_
        alt (str, optional): _description_. Defaults to 'two-sided'.

    Returns:
        dict: _description_
    """
    a = data[data[x] == 1][y]
    b = data[data[x] == 0][y]
    t = ranksums(a, b, alternative=alt)
    stats = {}
    stats["x"] = x
    stats["y"] = y
    stats["stats"] = t[0]
    stats["pval"] = t[1]
    stats["cohen"] = pg.compute_effsize(a, b, eftype="cohen")
    stats["power"] = power_ttest2n(nx=len(a), ny=len(b), d=stats["cohen"], alpha=0.01)
    return stats


def compute_wilc_table(
    data: pd.DataFrame, cible: list, col_list: list, seuil=0.01
) -> pd.DataFrame:
    i = 0
    df_result = pd.DataFrame(columns=["x", "y", "stats", "pval", "cohen", "power"])
    for elt in list(itertools.product(cible, col_list)):
        try:
            x = elt[0]
            y = elt[1]
            result = compute_wilcoxon(data, x, y)
            if result["pval"] <= seuil:
                df_result.loc[i] = result
                i = i + 1
        except:
            continue
    return df_result


def compute_anova_table(
    data: pd.DataFrame, cible: list, col_list: list, seuil=0.01
) -> pd.DataFrame:
    i = 0
    df_result = pd.DataFrame(columns=["x", "y", "p-unc", "np2", "power"])
    for elt in list(itertools.product(cible, col_list)):
        result = {}
        x = elt[0]
        y = elt[1]
        stats = pg.anova(data=data, dv=y, between=x, detailed=True).to_dict(
            orient="records"
        )[0]
        result["x"] = x
        result["y"] = y
        try:
            result["p-unc"] = stats["p-unc"]
            result["np2"] = stats["np2"]
            result["power"] = power_anova(
                eta_squared=result["np2"],
                k=len(set(data[x].tolist())),
                n=len(data),
                alpha=0.05,
            )
        except Exception as e:
            print(f"fail to compute anova becuase of {e}")
            result["p-unc"] = 1
            result["np2"] = 0
        if result["p-unc"] <= seuil:
            df_result.loc[i] = result
            i = i + 1

    return df_result


def compute_tttest(data: pd.DataFrame, x: str, y: str) -> dict:
    """compute one t test using pinguoins

    Args:
        data (pd.DataFrame): _description_
        x (str): _description_
        y (str): _description_

    Returns:
        dict: _description_
    """
    a = data[data[y] == 0][x]
    b = data[data[y] == 1][x]

    return pg.ttest(a, b, correction=False).to_dict(orient="records")[0]


def compute_mwu(data: pd.DataFrame, x: str, y: str) -> dict:
    """compute mann whitney u test using pinguoins

    Args:
        data (pd.DataFrame): _description_
        x (str): _description_
        y (str): _description_

    Returns:
        dict: _description_
    """
    a = data[data[y] == 0][x]
    b = data[data[y] == 1][x]
    stats = pg.mwu(a, b).to_dict(orient="records")[0]
    stats["cohen"] = pg.compute_effsize(a, b, eftype="cohen")
    stats["power"] = power_ttest2n(nx=len(a), ny=len(b), d=stats["cohen"], alpha=0.05)

    return stats


def compute_ttest_table(
    data: pd.DataFrame, cible: list, col_list: list, seuil=0.01
) -> pd.DataFrame:
    """compute t test for two sets of columns

    Args:
        data (pd.DataFrame): _description_
        cible (list): _description_
        col_list (list): _description_
        seuil (float, optional): _description_. Defaults to 0.01.

    Returns:
        pd.DataFrame: _description_
    """
    i = 0
    df_result = pd.DataFrame(
        columns=[
            "x",
            "y",
            "T",
            "dof",
            "alternative",
            "p-val",
            "CI95%",
            "cohen-d",
            "BF10",
            "power",
        ]
    )
    for elt in list(itertools.product(cible, col_list)):
        try:
            x = elt[0]
            y = elt[1]

            result = compute_tttest(data, y, x)
            result["x"] = x
            result["y"] = y

            if result["p-val"] <= seuil:
                df_result.loc[i] = result
                i = i + 1
        except Exception as e:
            print(e)
            continue
    return df_result


def compute_mwu_table(
    data: pd.DataFrame, cible: list, col_list: list, seuil=0.01
) -> pd.DataFrame:
    """compute mann whitney u test for two sets of columns

    Args:
        data (pd.DataFrame): _description_
        cible (list): _description_
        col_list (list): _description_
        seuil (float, optional): _description_. Defaults to 0.01.

    Returns:
        pd.DataFrame: _description_
    """
    i = 0
    df_result = pd.DataFrame(
        columns=[
            "x",
            "y",
            "T",
            "U-val",
            "alternative",
            "p-val",
            "RBC",
            "CLES",
            "cohen",
            "power",
        ]
    )
    for elt in list(itertools.product(cible, col_list)):
        try:
            x = elt[0]
            y = elt[1]

            result = compute_mwu(data, y, x)
            result["x"] = x
            result["y"] = y

            if result["p-val"] <= seuil:
                df_result.loc[i] = result
                i = i + 1
        except Exception as e:
            print(e)
            continue
    return df_result
