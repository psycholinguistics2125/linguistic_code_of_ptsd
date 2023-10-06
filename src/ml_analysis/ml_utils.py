"""
created at : 02, february 2023

"""

import numpy as np
import random
import pandas as pd
import logging
import os, yaml
import itertools
from collections import Counter


from src.ml_analysis.class_ml_analysis import MlAnalysis
from src.utils import flatten
from src.data_tools.data_utils import rename_features

import shap

# evaluation
from sklearn.metrics import roc_curve, RocCurveDisplay, auc
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    roc_auc_score,
    balanced_accuracy_score,
)


# interpretation
from rfpimp import *

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_tools.data_viz import add_median_labels


plt.rcParams.update({"font.size": 18})
plt.rcParams["figure.figsize"] = (13, 13)

targets_list = [
    "full_or_partial_PTSD",
    "CB_probable",
    "CC_probable",
    "CD_probable",
    "CE_probable",
    "CG_probable",
    "PTSD_probable",
]

models_names = {
    "rf": "Random Forest",
    "brf": "Imbalanced Random Forest",
    "ebm": "Explainable Boosting Machine",
    "lr": "Logistic Regression",
    "elasticnet": "Elascticnet Logsitic Regression",
    "lasso": "Lasso Logsitic Regression",
    "lgb": "Light Gradient Boosting Machine",
    "rc": "Ridge Classifier",
}


def compute_all_average(
    data:pd.DataFrame,
    config:dict,
    try_number=10,
    plot=False,
    save=False,
    top_features=20,
    models_list=["brf", "lr", "rf", "lasso", "elasticnet", "lgb", "ebm"],
    logger=logging.getLogger(),
    targets_list=targets_list,
):
    """ 
    compute average scores, average roc auc, 
    average interpretation and average error analysis for each target and each model


    Args:
        data (pd.DataFrame): _description_
        config (dict): _description_
        try_number (int, optional): _description_. Defaults to 10.
        plot (bool, optional): _description_. Defaults to False.
        save (bool, optional): _description_. Defaults to False.
        top_features (int, optional): _description_. Defaults to 20.
        models_list (list, optional): _description_. Defaults to ["brf", "lr", "rf", "lasso", "elasticnet", "lgb", "ebm"].
        logger (_type_, optional): _description_. Defaults to logging.getLogger().
        targets_list (_type_, optional): _description_. Defaults to targets_list.

    Returns:
        _type_: _description_
    """
    synthesis = {}

    for target in targets_list:
        logger.info(f"Starting target: {target}")
        # update config
        config["ml_analysis"]["target"] = target
        config["ml_analysis"]["test_size"] = 0.2

        # compute k time the models
        results = compute_k_experiences(
            data, config, k=try_number, model_list=models_list
        )
        # plot average roc aux curves
        compute_average_auc(results, config, save=True, selected_keys=models_list)
        for model_type in models_list:
            logger.info(f"Starting model: {model_type}")
            try:
                model_param, data_aug, scaler = load_models_param(
                    target, model_type, config
                )
            except Exception as e:
                data_aug = False
                if model_type in ["lr", "ebm", "lasso", "elasticnet"]:
                    scaler = True
                else:
                    scaler = False

            config["ml_analysis"]["scaler"] = scaler
            config["ml_analysis"]["data_aug"] = data_aug

            logger.info(
                f"Seting scaler to {scaler}; and data augmentation to {data_aug}"
            )

            logger.info(f"Intepretation.... ")
            int = average_interpretation(
                results, config, model_type, save=save, plot=plot, k=top_features
            )
            logger.info(f"Average scores.... ")
            # compute scores
            scores = compute_average_scores(
                results[model_type], config, plot=plot, save=save
            )
            # errors analysis
            logger.info(f"Error analysis .... ")
            errors = average_error_analysis(
                data, results, model_type, config, save=save, plot=plot, max_errors=20
            )
        # save config
        target = config["ml_analysis"]["target"]
        ml_folder = config["ml_analysis"]["ml_folder"]
        best_param = str(config["ml_analysis"]["best_param"])
        if save:
            logger.info(f"Saving... ")
            try:
                config_path = os.path.join(
                    ml_folder,
                    f"{target}_average_{try_number}_{best_param}",
                    "config.yaml",
                )
                yaml.dump(config, open(config_path, "w"))
            except Exception as e:
                logger.info(f"fail to save config because of {e}")

        synthesis[target] = results

    return synthesis


def compute_k_models(data:pd.DataFrame, config:dict, model_type:str, k=100):
    """compute k models for a given model type

    Args:
        data (pd.DataFrame): _description_
        config (dict): _description_
        model_type (str): _description_
        k (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    # initiate parameter
    data_aug = config["ml_analysis"]["data_aug"]
    target = config["ml_analysis"]["target"]
    i = 0
    # results
    result = pd.DataFrame(
        columns=[
            "model_type",
            "train",
            "train_label",
            "test",
            "test_label",
            "model",
            "seed",
        ]
    )

    for seed in random.sample(range(0, 1000), k):
        config["ml_analysis"]["seed"] = seed
        ml_exp = MlAnalysis(config)
        train, train_label, test, test_label = ml_exp.get_train_test(
            data, data_augmentation=data_aug
        )
        # print(train.columns)
        model = ml_exp.build_model(model_type=model_type)
        model = ml_exp.train_one_model(train, train_label, save=False)

        result.loc[i] = pd.Series(
            {
                "model_type": model_type,
                "train": train,
                "train_label": train_label,
                "test": test,
                "test_label": test_label,
                "model": model,
                "seed": seed,
            }
        )
        i += 1

    return result


def compute_k_experiences(
    data: pd.DataFrame,
    config: dict,
    k=100,
    model_list=["ebm", "brf", "lr", "rf", "lasso", "elasticnet", "dt", "lda"],
) -> pd.DataFrame:
    """An experience is an ensemble of model with a set of define from config


    Args:
        data (_type_): _description_
        config (_type_): _description_
        k (int, optional): _description_. Defaults to 100.
        model_list (list, optional): _description_. Defaults to ["ebm","brf",'lr','rf',"lasso","elasticnet","dt",'lda'].

    Returns:
        pd.DataFrame : _description_
    """

    results = {}
    for model_type in model_list:
        results[model_type] = compute_k_models(data, config, model_type, k=k)
    return results


def compute_average_scores(df_exps:pd.DataFrame, config:dict, plot=True, save=False, return_train=False):
    """compute average scores for a list of experiences in df_exps

    Args:
        df_exps (pd.DataFrame): _description_
        config (dict): _description_
        plot (bool, optional): _description_. Defaults to True.
        save (bool, optional): _description_. Defaults to False.
        return_train (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    target = config["ml_analysis"]["target"]
    scores = pd.DataFrame(
        columns=[
            "auc",
            "precision",
            "recall",
            "accuracy",
            "balanced_accuracy",
            "f1_weighted",
            "f1_macro",
            "f1_micro",
        ]
    )
    train_scores = pd.DataFrame(
        columns=[
            "auc",
            "precision",
            "recall",
            "accuracy",
            "balanced_accuracy",
            "f1_weighted",
            "f1_macro",
            "f1_micro",
        ]
    )

    for i in range(len(df_exps)):
        line = df_exps.loc[i]
        train, train_label, test, test_label = (
            line["train"],
            line["train_label"],
            line["test"],
            line["test_label"],
        )
        model = line["model"]
        model_type = line["model_type"]

        preds = model.predict(test)
        prec = precision_score(test_label, preds, average="weighted")
        rec = recall_score(test_label, preds, average="weighted")
        acc = accuracy_score(test_label, preds)
        try:
            auc = roc_auc_score(test_label, model.predict_proba(test)[:, 1])
        except:
            auc = 0
        bal_acc = balanced_accuracy_score(test_label, preds)
        f1_wei = f1_score(test_label, preds, average="weighted")
        f1_mac = f1_score(test_label, preds, average="macro")
        f1_mic = f1_score(test_label, preds, average="micro")

        scores.loc[i] = pd.Series(
            {
                "auc": auc,
                "precision": prec,
                "recall": rec,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "f1_weighted": f1_wei,
                "f1_macro": f1_mac,
                "f1_micro": f1_mic,
            }
        )

        train_preds = model.predict(train)
        prec = precision_score(train_label, train_preds, average="weighted")
        rec = recall_score(train_label, train_preds, average="weighted")
        acc = accuracy_score(train_label, train_preds)
        try:
            auc = roc_auc_score(train_label, model.predict_proba(train)[:, 1])
        except:
            auc = 0
        bal_acc = balanced_accuracy_score(train_label, train_preds)
        f1_wei = f1_score(train_label, train_preds, average="weighted")
        f1_mac = f1_score(train_label, train_preds, average="macro")
        f1_mic = f1_score(train_label, train_preds, average="micro")

        train_scores.loc[i] = pd.Series(
            {
                "auc": auc,
                "precision": prec,
                "recall": rec,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "f1_weighted": f1_wei,
                "f1_macro": f1_mac,
                "f1_micro": f1_mic,
            }
        )

    train_scores["dataset"] = "train"
    scores["dataset"] = "test"
    all_scores = pd.concat([train_scores, scores])
    if save:
        target = config["ml_analysis"]["target"]
        ml_folder = config["ml_analysis"]["ml_folder"]
        best_param = str(config["ml_analysis"]["best_param"])
        average_folder = os.path.join(
            ml_folder, f"{target}_average_{len(df_exps)}_{best_param}"
        )
        if not os.path.exists(average_folder):
            os.mkdir(average_folder)
        saving_path = os.path.join(
            average_folder, f"{model_type}_average_performances.csv"
        )
        scores.to_csv(saving_path)

    if plot:
        fig = plt.figure()
        df = all_scores.melt(
            id_vars=["dataset"],
            value_vars=[
                "auc",
                "precision",
                "recall",
                "accuracy",
                "balanced_accuracy",
                "f1_weighted",
                "f1_macro",
                "f1_micro",
            ],
            var_name="scores_names",
            value_name="scores",
        )
        ax = sns.boxplot(
            data=df, x="scores", y="scores_names", orient="h", hue="dataset"
        )
        add_median_labels(ax, fmt=".2f")
        ax.set_title(f"scores for {models_names[model_type]}")
        if save:
            plt.savefig(
                os.path.join(
                    average_folder, f"{model_type}_average_performancers_plots.jpeg"
                ),
                bbox_inches="tight",
                dpi=200,
            )
        else:
            plt.show()
    if return_train:
        return all_scores
    return scores


def compute_average_auc(
    results:pd.DataFrame, config, save=True, selected_keys=["ebm", "brf", "lr", "rf"]
):
    """compute average roc auc for a list of experiences in results

    Args:
        results (pd.DataFrame): _description_
        config (_type_): _description_
        save (bool, optional): _description_. Defaults to True.
        selected_keys (list, optional): _description_. Defaults to ["ebm", "brf", "lr", "rf"].

    Returns:
        _type_: _description_
    """
    target = config["ml_analysis"]["target"]
    if target == "full_and_partial_PTSD":
        return "multiclass ROC avecrage not implemented yet"

    if selected_keys == None:
        selected_keys = list(results.keys())

    # init visualisation
    _, ax = plt.subplots(figsize=(10, 10))
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

    for model_type in selected_keys:
        exp = results[model_type]

        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, len(exp))
        sns.set(font_scale=2)
        for i in range(len(exp)):
            line = exp.loc[i]
            train, train_label, test, test_label = (
                line["train"],
                line["train_label"],
                line["test"],
                line["test_label"],
            )
            model = line["model"]

            # vizualisation
            viz = RocCurveDisplay.from_estimator(
                model,
                test,
                test_label,
                name=None,  # f"ROC seed {seed}",
                alpha=0.3,
                lw=1,
                ax=ax,
                label="_ignore",
            )

            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax.plot(
            mean_fpr,
            mean_tpr,
            label=rf"Mean ROC {model_type} (AUC = %0.2f $\pm$ %0.2f)"
            % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            alpha=0.2,
            label=rf" {model_type}$\pm$ 1 std.",
        )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability\n(Positive label '{target}', {len(exp)} runs)",
    )
    ax.axis("square")
    plt.legend(loc="lower right")

    if save:
        target = config["ml_analysis"]["target"]
        ml_folder = config["ml_analysis"]["ml_folder"]
        best_param = str(config["ml_analysis"]["best_param"])
        average_folder = os.path.join(
            ml_folder, f"{target}_average_{len(exp)}_{best_param}"
        )
        if not os.path.exists(average_folder):
            os.mkdir(average_folder)
        saving_path = os.path.join(average_folder, f"average_roc_auc.jpeg")

        plt.savefig(saving_path, bbox_inches="tight", dpi=200)
    else:
        plt.show()


def average_error_analysis(
    data, results, model_type, config, save=False, plot=False, max_errors=20
):
    """compute average error analysis for a list of experiences in results"""

    df = results[model_type]
    ml_exp = MlAnalysis(config, save=False)
    errors_list = []
    for i in range(len(df)):
        line = df.loc[i]
        model = line["model"]
        test = line["test"]
        test_label = line["test_label"]
        df_errors = ml_exp.error_analysis(
            data, x_test=test, y_test=test_label, model=model
        )
        errors_list.append(df_errors["uuid"].tolist())

    ERRORS = pd.DataFrame(
        Counter(flatten(errors_list)).most_common(), columns=["uuid", "score"]
    ).sort_values(["score"], ascending=False)

    if save:
        target = config["ml_analysis"]["target"]
        ml_folder = config["ml_analysis"]["ml_folder"]
        best_param = str(config["ml_analysis"]["best_param"])
        average_folder = os.path.join(
            ml_folder, f"{target}_average_{len(df)}_{best_param}"
        )
        if not os.path.exists(average_folder):
            os.mkdir(average_folder)
        saving_path = os.path.join(average_folder, f"{model_type}_average_errors.csv")
        ERRORS.to_csv(saving_path)

    if plot:
        E = ERRORS.iloc[:max_errors]
        fig = plt.figure()
        fig = sns.barplot(data=E, x="score", palette="flare", orient="h", y="uuid")
        plt.title(f"Average erros over {len(df)} runs for {models_names[model_type]}")
        if save:
            plt.savefig(
                os.path.join(average_folder, f"{model_type}_error_plots.jpeg"),
                bbox_inches="tight",
                dpi=200,
            )
        else:
            plt.show()

    return ERRORS


def average_interpretation(
    results, config, model_type, save=False, plot=False, k=20, rename=True
):
    INTERPRETATIONS = pd.DataFrame()
    try:
        df = results[model_type]
    except Exception as e:
        print(f"fail to select the model type because of {e}")
        return INTERPRETATIONS

    if model_type == "ebm":
        INTERPRETATIONS = interpret_ebm(df)
        SHAP_INTERPRETATIONS = INTERPRETATIONS
    elif model_type in ["lr", "elasticnet", "lasso", "rc"]:
        INTERPRETATIONS = interpret_lr(df)
        SHAP_INTERPRETATIONS = interpret_with_shap(df, model_type= model_type)
    elif model_type in ["rf", "brf", "dt"]:
        INTERPRETATIONS = interpret_rf(df)
        SHAP_INTERPRETATIONS = interpret_with_shap(df, model_type= model_type)

    elif model_type in ["lgb"]:
        INTERPRETATIONS = interpret_lgb(df)
        SHAP_INTERPRETATIONS = interpret_with_shap(df, model_type= model_type)
    
    

    resume = pd.DataFrame(
        INTERPRETATIONS.apply(sum) / len(INTERPRETATIONS), columns=["scores"]
    ).sort_values("scores", ascending=False)

    resume_shap = pd.DataFrame(
        SHAP_INTERPRETATIONS.apply(sum) / len(SHAP_INTERPRETATIONS), columns=["scores"]
    ).sort_values("scores", ascending=False)

    best_features = resume.iloc[:k].index.tolist()
    best_features_shap = resume_shap.iloc[:k].index.tolist()


    INTERPRETATIONS = INTERPRETATIONS[best_features]
    SHAP_INTERPRETATIONS = SHAP_INTERPRETATIONS[best_features_shap]

    if rename:
        INTERPRETATIONS = rename_features(INTERPRETATIONS)
        SHAP_INTERPRETATIONS = rename_features(SHAP_INTERPRETATIONS)


    if save:
        target = config["ml_analysis"]["target"]
        ml_folder = config["ml_analysis"]["ml_folder"]
        best_param = str(config["ml_analysis"]["best_param"])
        average_folder = os.path.join(
            ml_folder, f"{target}_average_{len(df)}_{best_param}"
        )
        if not os.path.exists(average_folder):
            os.mkdir(average_folder)
        
        saving_path = os.path.join(
            average_folder, f"{model_type}_average_interpretations.csv"
        )
        saving_path_shap = os.path.join(
            average_folder, f"{model_type}_shap_average_interpretations.csv"
        )
        INTERPRETATIONS.to_csv(saving_path)
        SHAP_INTERPRETATIONS.to_csv(saving_path_shap)

    if plot:
        fig = plt.figure()
        sns.set(font_scale=3)
        fig = sns.boxplot(
            data=INTERPRETATIONS,
            palette="flare",
            orient="h",
        )

        # plt.title(f"Average features importances over {len(INTERPRETATIONS)} runs for {models_names[model_type]}")
        if save:
            plt.savefig(
                os.path.join(
                    average_folder, f"{model_type}_average_interpretations_plots.jpeg"
                ),
                bbox_inches="tight",
                dpi=200,
            )
        else:
            plt.show()

        fig2 = plt.figure()
        sns.set(font_scale=3)
        fig2 = sns.boxplot(
            data=SHAP_INTERPRETATIONS,
            palette="flare",
            orient="h",
        )

        # plt.title(f"Average features importances over {len(INTERPRETATIONS)} runs for {models_names[model_type]}")
        if save:
            plt.savefig(
                os.path.join(
                    average_folder, f"{model_type}_average_shap_interpretations_plots.jpeg"
                ),
                bbox_inches="tight",
                dpi=200,
            )
        else:
            plt.show()

        

    return INTERPRETATIONS


# specific interpretation functions


def interpret_lgb(df):
    cols = list(df.loc[0]["train"].columns)
    R = pd.DataFrame(columns=cols)
    for i in range(len(df)):
        line = df.loc[i]
        model = line["model"]
        res = dict(zip(model.feature_name_, model.feature_importances_))
        R.loc[i] = pd.Series(res)

    return R

def interpret_with_shap(df, model_type="rf"):
    cols = list(df.loc[0]["train"].columns)
    R = pd.DataFrame(columns=cols)
    for i in range(len(df)):
        line = df.loc[i]
        model = line["model"]
        check_additivity = False
        explainer = shap.Explainer(model, df.loc[i]["train"], check_additivity=check_additivity)
        
        shap_values = explainer(df.loc[i]["test"])
        if model_type == "rf":
            shap_values = shap_values[:,:,1]
        vals = abs(shap_values.values).mean(axis=0)
        res = dict(zip(cols, vals))
        R.loc[i] = pd.Series(res)
    return R

def interpret_rf(df):
    cols = list(df.loc[0]["train"].columns)
    R = pd.DataFrame(columns=cols)
    for i in range(len(df)):
        line = df.loc[i]
        model = line["model"]
        test_data = line["test"]
        test_label = line["test_label"]
        imp = importances(model, test_data, test_label)
        res = imp.to_dict()["Importance"]
        R.loc[i] = pd.Series(res)

    return R


def interpret_lr(df):
    cols = list(df.loc[0]["train"].columns)
    R = pd.DataFrame(columns=cols)
    for i in range(len(df)):
        line = df.loc[i]
        model = line["model"]
        res = dict(zip(cols, model.coef_[0]))
        R.loc[i] = pd.Series(res)

    return R


def interpret_ebm(df):
    cols = list(df.loc[0]["train"].columns)
    all_cols = cols + [
        elt[0] + " x " + elt[1] for elt in list(itertools.combinations(cols, 2))
    ]
    init_dict = dict.fromkeys(all_cols, 0)
    R = pd.DataFrame(columns=all_cols)
    for i in range(len(df)):
        line = df.loc[i]
        model = line["model"]
        res = dict(zip(model.term_names_, model.term_importances()))

        R.loc[i] = pd.Series({**init_dict, **res})

    return R


init_col_dict = {
    "first_personal_pronoun_sing": 0,
    "LCC": 0,
    "score_parenthetique_matches": 0,
    "repetitions_type_1_score": 0,
    "score_generical_connector_matches": 0,
    "ADP": 0,
    "degree_average": 0,
    "verb_conditionel": 0,
    "passive_count": 0,
    "NOUN": 0,
    "repetitions_type_5_score": 0,
    "PRESENT_ENNONCIATION": 0,
    "LSC": 0,
    "passive_count_norm": 0,
    "SCONJ": 0,
    "SYM": 0,
    "average_shrotest_path_g0": 0,
    "SPACE": 0,
    "PRESENT_HISTORIQUE": 0,
    "PRESENT_GENERIQUE": 0,
    "ADV": 0,
    "PRON": 0,
    "CORPS": 0,
    "AUX": 0,
    "L3": 0,
    "L2": 0,
    "ON_NOUS": 0,
    "L1": 0,
    "diameter_g0": 0,
    "PE": 0,
    "transitivity": 0,
    "NOM_PERCEPTIONS_SENSORIELLES": 0,
    "DET": 0,
    "CONJ": 0,
    "PART": 0,
    "ON_QUELQU_UN": 0,
    "third_personal_pronoun": 0,
    "PUNCT": 0,
    "passive_percentages": 0,
    "VERB_PERCEPTIONS_SENSORIELLES": 0,
    "CCONJ": 0,
    "PROPN": 0,
    "SENSATIONS_PHYSIQUES": 0,
    "verb_indicatif_present": 0,
    "verb_indicatif_future": 0,
    "NUM": 0,
    "INTJ": 0,
    "score_onomatopes_matches": 0,
    "score_troncations_matches": 0,
    "average_clustering": 0,
    "ADJ": 0,
    "first_personal_pronoun_plur": 0,
    "VERB": 0,
    "score_paticules_matches": 0,
    "repetitions_type_3_score": 0,
    "MORT_EXPLICITE": 0,
    "repetitions_type_4_score": 0,
    "verb_participe_passe": 0,
    "passive_sents_count": 0,
    "second_personal_pronoun": 0,
    "repetitions_type_2_score": 0,
    "repetitions_type_6_score": 0,
    "X": 0,
    "verb_indicatif_imparfait": 0,
    "ON_GENERIQUE": 0,
    "score_temporal_connector_matches": 0,
}
