import pandas as pd
import numpy as np

import logging
import os

import shutil
from src.utils import flatten
import random

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


# data
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


# models

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, KMeansSMOTE
from imblearn.pipeline import Pipeline

import joblib
from sklearn.dummy import DummyClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
from sklearn.gaussian_process.kernels import RBF
import lightgbm as lgb


# visualisation
import matplotlib.pyplot as plt
import seaborn as sns


# interpretation
from rfpimp import *


class MlAnalysis:
    def __init__(self, config, save=False, verbose=False) -> None:
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self.seed = self.config["ml_analysis"]["seed"]
        self.features_name = "_".join(self.config["ml_analysis"]["features"])
        self.experience_name = (
            f"{self.config['ml_analysis']['target']}_{self.features_name}"
        )
        self.data_aug = self.config["ml_analysis"]["data_aug"]

        if save:
            self.init_folder()

        self.verbose = verbose
        if self.verbose:
            self._logger.info(
                f"Initializing experience folder for {self.experience_name}"
            )

    def init_folder(self):
        """
        Create the folder and subfolders associated to experience
        """
        self.saving_folder = os.path.join(
            self.config["ml_analysis"]["ml_folder"], self.experience_name
        )
        if not os.path.exists(self.saving_folder):
            os.mkdir(self.saving_folder)
            if self.verbose:
                self._logger.info(f"Creating folder {self.saving_folder}")

        self.models_path = os.path.join(self.saving_folder, "models")
        if not os.path.exists(self.models_path):
            os.mkdir(self.models_path)
            if self.verbose:
                self._logger.info(f"Creating models folder {self.models_path}")

        self.plots_path = os.path.join(self.saving_folder, "plots")
        if not os.path.exists(self.plots_path):
            os.mkdir(self.plots_path)
            if self.verbose:
                self._logger.info(f"Creating plots folder {self.plots_path}")
        try:
            shutil.copy(
                "/home/robin/Code_repo/PTSD_analysis/config.yaml", self.saving_folder
            )
            if self.verbose:
                self._logger.info("config file succesfully copied")
        except Exception as e:
            if self.verbose:
                self._logger.info(f"Fail to copy config because of {e}")

    def get_features(self, data, features_names_list):
        selected_col = []

        if "coherence" in features_names_list:
            c = data.filter(regex=r"order*").columns.tolist()
            selected_col.append(c)

        if "readability" in features_names_list:
            for filters in [
                r"noum_ratio_score",
                "direct_discourse_score",
            ]:  # [r"discourse*","perplexity_score",'lexique_old20_score','duboix_score',"GFI_score","ARI_score","FRE_score","FKGL_score","SMOG_score","REL_score"]:
                c = data.filter(regex=filters).columns.tolist()
                selected_col.append(c)

        if "sentiments" in features_names_list:
            for filters in [
                r"textblob*",
                "labMT",
                r"feel_positive",
                r"liwc_émo*",
            ]:  # liwc_émo","liwc_mort","liw_corps",r"empath*", r"gobin*", r"feel*",r"polarimot*"
                c = data.filter(regex=filters).columns.tolist()
                selected_col.append(c)

        if "custom_ner" in features_names_list:
            for filters in [
                r"model_.*",
                r"PRESENT_.*",
                r"ON_.*",
                r".*PERCEPTIONS_.*",
                r"SENSATIONS.*",
            ]:
                c = data.filter(regex=filters).columns.tolist()
                selected_col.append(c)

        if "passive" in features_names_list:
            for filters in ["passive_count_norm"]:  # "passive_.*", passive_count
                c = data.filter(regex=filters).columns.tolist()
                selected_col.append(c)

        if "graph" in features_names_list:
            for filters in [
                "degree_average",
                "average_clustering",
                "average_shrotest_path_g0",
                r"^L.{1,2}",
                "diameter_g0",
                "transitivity",
                r"^PE",
            ]:
                c = data.filter(regex=filters).columns.tolist()
                selected_col.append(c)

        if "morph" in features_names_list:
            # for filters in [r".+(\:).+",r"PQP_score*"]:
            #    c = data.filter(regex =filters).columns.tolist()
            morph = [
                "first_personal_pronoun_sing",
                "first_personal_pronoun_plur",
                "second_personal_pronoun",
                "third_personal_pronoun",
                "verb_indicatif_present",
                "verb_indicatif_future",
                "verb_participe_passe",
                "verb_conditionel",
                "verb_indicatif_imparfait",
                "PQP_score",
            ]
            selected_col.append(morph)

        if "tag" in features_names_list:
            tag = [
                "ADP",
                "NOUN",
                "SPACE",
                "ADV",
                "PUNCT",
                "DET",
                "NUM",
                "PRON",
                "VERB",
                "PROPN",
                "SCONJ",
                "AUX",
                "X",
                "CCONJ",
                "ADJ",
                "CONJ",
                "INTJ",
                "SYM",
                "PART",
            ]

            selected_col.append(tag)

        if "dysfluences" in features_names_list:
            for filters in [
                r"repetition_type_1",
                "repetition_type_2",
                r".*_matches",
                "difluencies_score",
            ]:
                c = data.filter(regex=filters).columns.tolist()
                selected_col.append(c)

        if len(selected_col) == 0:
            if self.verbose:
                self._logger.info(
                    f"Custom filters are selected {self.config['ml_analysis']['custom']}"
                )
            for filters in [r".*_probable"]:  # self.config['ml_analysis']['custom']:
                c = data.filter(regex=filters).columns.tolist()
                selected_col.append(c)

        selected_col = list(set(flatten(selected_col)))

        selected_col = (
            data[selected_col].select_dtypes(include=["float", "int"]).columns.tolist()
        )

        return selected_col

    def get_train_test(
        self, data, selected_col=None, data_augmentation=None, scaler=None
    ) -> tuple:
        """
        split data in train and test set

        Returns:
            tuple: train, test
        """
        if data_augmentation == None:
            data_augmentation = self.data_aug
        # spliting configuration
        test_size = self.config["ml_analysis"]["test_size"]
        group_name = self.config["ml_analysis"]["group_name"]
        target = self.config["ml_analysis"]["target"]
        strati = self.config["ml_analysis"]["strati"]

        #
        if selected_col == None:
            if len(self.config["ml_analysis"]["selected_cols"]) == 0:
                selected_col = self.get_features(
                    data, self.config["ml_analysis"]["features"]
                )
            else:
                selected_col = self.config["ml_analysis"]["selected_cols"]
        if self.verbose:
            self._logger.info(f"{len(selected_col)} features were selected")

        if scaler == None:
            scaler = self.config["ml_analysis"]["scaler"]
        if scaler:
            data[selected_col] = StandardScaler().fit_transform(data[selected_col])

        # build target for the stratification_split
        data["strati_target"] = (
            data[target].astype(str)
            + "_"
            + data["sexe"].astype(str)
            + "_"
            + data[strati].astype(str)
        )  # data["full_and_partial_PTSD"].astype(str) + "_" + data[strati].astype(str)+ "_" + data["sexe"].astype(str)
        gss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=self.seed
        )

        for train_index, test_index in gss.split(data["uuid"], data["strati_target"]):
            train = data[selected_col].loc[
                train_index
            ]  # .reset_index().drop("index",axis=1)
            if self.verbose:
                self._logger.info(
                    f"Distribution of A criterion in train set: {data[group_name].loc[train_index].value_counts()}"
                )
            train_label = data[target].loc[train_index]
            test = data[selected_col].loc[test_index]  # .reset_index()
            test_label = data[target].loc[test_index]
        if self.verbose:
            self._logger.info(
                f"There are {len(train)}, training examples,  There are {len(test)} testing examples"
            )

        if data_augmentation:
            over_ratio = self.config["ml_analysis"]["over_ratio"]
            under_ratio = self.config["ml_analysis"]["under_ratio"]
            under = self.config["ml_analysis"]["under"]
            if self.verbose:
                self._logger.info(
                    f"Data augmentation set to True, using the ratio from config : over  = {over_ratio}; under = {under_ratio}"
                )

            over_smote = SMOTE(
                sampling_strategy={
                    0: int(len(train) * over_ratio),
                    1: int(len(train) * over_ratio),
                },
                random_state=self.seed,
            )

            under_random = RandomUnderSampler(
                sampling_strategy={
                    0: int(len(train) * under_ratio),
                    1: int(len(train) * under_ratio),
                },
                random_state=self.seed,
            )

            if under:
                steps = [("over", over_smote), ["under", under_random]]
            else:
                steps = [("over", over_smote)]

            pipeline = Pipeline(steps=steps)

            train, train_label = pipeline.fit_resample(train, train_label)
            if self.verbose:
                self._logger.info(
                    f"The new train data set is composed of {len(train)} elements"
                )

        if self.config["ml_analysis"]["add_random"]:
            train["random"] = [random.random() for n in range(len(train))]
            test["random"] = [random.random() for n in range(len(test))]

        return train, train_label, test, test_label

    def build_model(self, model_type, best_param=False):
        best_param = self.config["ml_analysis"]["best_param"]
        self.model_type = model_type
        if best_param:
            try:
                target = self.config["ml_analysis"]["target"]
                model_param, _, _ = load_models_param(
                    target, self.model_type, self.config
                )
                # print(model_param)
                # update config dict
                self.config["ml_analysis"][self.model_type] = model_param
            except Exception as e:
                model_param = self.config["ml_analysis"][self.model_type]
            # self._logger.warning(f"Fail to load best param for {self.model_type} model parameter because of {e}")
        else:
            model_param = self.config["ml_analysis"][self.model_type]

        if self.model_type == "rf":
            model = RandomForestClassifier(**model_param, random_state=self.seed)
        elif self.model_type == "brf":
            model = BalancedRandomForestClassifier(
                **model_param, random_state=self.seed
            )
        elif self.model_type == "ebm":
            model = ExplainableBoostingClassifier(**model_param, random_state=self.seed)
        elif self.model_type == "lr":
            model = LogisticRegression(**model_param, random_state=self.seed)
        elif self.model_type == "lasso":
            model = LogisticRegression(**model_param, random_state=self.seed)
        elif self.model_type == "elasticnet":
            model = LogisticRegression(**model_param, random_state=self.seed)
        elif self.model_type == "rc":
            model = RidgeClassifier(**model_param, random_state=self.seed)
        elif self.model_type == "dt":
            model = DecisionTreeClassifier(**model_param, random_state=self.seed)
        elif self.model_type == "lda":
            model = LinearDiscriminantAnalysis(**model_param)
        elif self.model_type == "qda":
            model = QuadraticDiscriminantAnalysis(**model_param)
        elif self.model_type == "gp":
            kernel = 1.0 * RBF(1.0)
            model = GaussianProcessClassifier(
                **model_param, random_state=self.seed, kernel=kernel
            )
        elif self.model_type == "lgb":
            model = lgb.LGBMClassifier(**model_param, random_state=self.seed)
        elif self.model_type == "dummy":
            model = DummyClassifier(**model_param)
        else:
            self._logger.warning(f"Model name : {self.model_type}, not supported yet")

        self.model = model
        return self.model

    def train_one_model(self, train_data, train_label, save=False):
        trained_model = self.model.fit(train_data, train_label)
        self.model = trained_model

        if save:
            model_saving_path = os.path.join(
                self.models_path, f"{self.model_type}_model.pkl"
            )
            joblib.dump(trained_model, model_saving_path)
            if self.verbose:
                self._logger.info(f"Model saved in {model_saving_path}")

        return trained_model

    def error_analysis(self, data, x_test=None, y_test=None, model=None):
        if y_test is None:
            x, y, x_test, y_test = self.get_train_test(data)
        if model is None:
            model = self.model

        complete_test_dataset = data.loc[x_test.index.tolist()]
        complete_test_dataset["preds"] = model.predict(x_test)

        error_list = complete_test_dataset[
            complete_test_dataset[self.config["ml_analysis"]["target"]]
            != complete_test_dataset["preds"]
        ]

        return error_list

    def evaluate_one_model(self, test_data, test_label, save=False):
        """evaluate a self.model

        Args:
            test_data (_type_): _description_
            test_label (_type_): _description_

        Returns:
            _type_: _description_
        """
        preds = self.model.predict(test_data)
        tn, fp, fn, tp = confusion_matrix(test_label, preds).ravel()
        specificity = tn / (tn + fp)
        if len(np.unique(test_label)) == 2:
            auc = (
                roc_auc_score(test_label, self.model.predict_proba(test_data)[:, 1]),
            )
        else:
            auc = 0

        results = {
            "model_type": self.model_type,
            "precision": precision_score(test_label, preds, average="weighted"),
            "recall": recall_score(test_label, preds, average="weighted"),
            "auc_score": auc,
            "specificity": specificity,
            "accuracy": accuracy_score(test_label, preds),
            "f1": f1_score(test_label, preds, average="weighted"),
            "report": classification_report(test_label, preds),
        }

        for name, score in results.items():
            if self.verbose:
                self._logger.info(f"{name} score is : {score}")
        if save:
            pd.DataFrame(results, index=results.keys()).to_csv(
                os.path.join(self.models_path, f"{self.model_type}_performances.csv")
            )
        return results

    def interpret_model(self, train_data, train_label, save=False):
        self.interpretation = pd.DataFrame(columns=["name", "importance"])

        if self.model_type in ["rf", "brf", "dt"]:
            self.interpretation = importances(self.model, train_data, train_label)
            self.interpretation["name"] = self.interpretation.index

        elif self.model_type in ["lr", "lasso", "elasticnet"]:
            self.interpretation["name"] = list(train_data.columns)
            self.interpretation["importance"] = self.model.coef_[0]

        elif self.model_type == "ebm":
            self.interpretation["name"] = self.model.term_names_
            self.interpretation["importance"] = self.model.term_importances()
            self.interpretation = self.interpretation.sort_values(
                "importance", ascending=False
            )

        if save:
            pd.DataFrame(self.interpretation).to_csv(
                os.path.join(self.models_path, f"{self.model_type}_interpretations.csv")
            )

        return pd.DataFrame(self.interpretation)

    def plot_roc_auc_curves(
        self, dummy_model, model_list, test_data, test_label, save=False
    ):
        disp = RocCurveDisplay.from_estimator(model, test_data, test_label, ax=disp.ax_)
        for model in model_list:
            RocCurveDisplay.from_estimator(model, test_data, test_label, ax=disp.ax_)

        plt.legend(fontsize=13, loc="best")
        plt.show()

        if save:
            fig_save_name = os.path.join(self.plots_path, f"roc_auc.png")
            plt.savefig(fig_save_name)


    def lauch_experience(
        self, data, selected_col=None, save=True, plot=False, data_augmentation=None
    ):
        if data_augmentation == None:
            data_augmentation = self.data_aug

        # split data set
        train_data, train_label, test_data, test_label = self.get_train_test(
            data, selected_col, data_augmentation=data_augmentation
        )

        # train all the interpretable models

        models_list = []
        dummy_model = None

        concat_result = {}
        for model_type in ["dummy", "brf", "rf", "ebm", "lr"]:
            self.model_type = model_type

            self.model = self.build_model(model_type)
            self.model = self.train_one_model(train_data, train_label, save)
            if self.model_type == "dummy":
                dummy_model = self.model
            else:
                models_list.append(self.model)
                concat_result[model_type] = self.evaluate_one_model(
                    test_data, test_label, save
                )
            interpret = self.interpret_model(train_data, train_label, save)
        if plot:
            self.plot_roc_auc_curves(
                dummy_model, models_list, test_data, test_label, save
            )
        if self.verbose:
            self._logger.info(f"Experience {self.experience_name} is done  ! ")

        return concat_result
