import pandas as pd
import os
import numpy as np
import random


def load_features_and_meta(config: dict) -> pd.DataFrame:
    try:
        meta_data = pd.read_csv(
            os.path.join(
                config["data"]["data_folder"], config["data"]["etude_1000_filename"]
            ),
            sep=",",
        )
        features = pd.read_csv(
            os.path.join(
                config["data"]["data_folder"], config["data"]["features_filename"]
            ),
            sep="\t",
        )
        data = meta_data.merge(features, on="uuid")
    except Exception as e:
        print(f"Fail to load merged data file because of {e}")
        data = pd.DataFrame()
    data = rename_features(data)
    return data


def get_features_name(config, analyse_type):
    subgroups = "A1_A2"
    features_choices = config["data"]["features_choices"]
    features_choices_str = "_".join(features_choices)
    stats_file_name = f"{subgroups}_{analyse_type}_{features_choices_str}.csv"
    features_file_name = f"{subgroups}_{features_choices_str}_features.csv"
    return features_file_name, stats_file_name


def rename_features(data: pd.DataFrame) -> pd.DataFrame:
    """for visualisation purposes

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    col_dict = {
        "MORT_EXPLICITE": "model_DEATH",
        "SENSATIONS_PHYSIQUES": "model_PHYSICAL_SENSATIONS",
        "CORPS": "model_BODY",
        "ON_GENERIQUE": "ON_generic",
        "ON_NOUS": "ON_we",
        "ON_QUELQU_UN": "ON_someone",
        "PRESENT_ENNONCIATION": "Enunciative_PRESENT",
        "PRESENT_GENERIQUE": "Generical_PRESENT",
        "PRESENT_HISTORIQUE": "Historical_PRESENT",
        "VERB_PERCEPTIONS_SENSORIELLES": "VERB_SENSORY_PERCEPTIONS",
        "NOUM_PERCEPTIONS_SENSORIELLES": "NOUM_SENSORY_PERCEPTIONS",
        "score_troncations_matches": "difluencies_score",
    }

    copy_data = data.copy()
    copy_data = copy_data.rename(columns=col_dict, errors="ignore")

    return copy_data


def compute_PTSD_and(x):
    if x.PTSD_probable == 1:
        return 2
    elif x.partial_PTSD_probable == 1:
        return 1
    else:
        return 0


def transform_lieu(x):
    if x == "STADE DE FRANCE/ASSAUT ST DENIS":
        return "ASSAUT ST DENIS"
    elif x == "LOISIR ET/OU TRAVAIL":
        return "BATACLAN"
    else:
        return x


def transform_diplome(x):
    if x == "other":
        return "high school or less"
    else:
        return x


def get_indices_list(word_list: list, seq_len: int, overlap: int) -> list:
    k = int(len(word_list) / seq_len)
    r = len(word_list) % seq_len
    indices_list = []
    for i in range(k):
        s = i * seq_len
        if s > overlap:
            s = s - overlap
        e = (i + 1) * seq_len
        indices_list.append((s, e))
    e = indices_list[-1][1]
    indices_list.append((e, r))
    return indices_list


def get_train_indices_list(word_list, seq_len, label=1, seed=45, upsampling_0=False):
    len_train_indices = int(15000 / seq_len)
    min_prop = 4
    np.random.seed(seed)
    random.seed(seed)

    max_len = len(word_list)
    if label == 0:
        if upsampling_0:
            n_indices = len_train_indices * min_prop  # we take
        else:
            n_indices = len_train_indices
    else:
        n_indices = len_train_indices
    I = np.sort(np.random.randint(0, max_len - seq_len, n_indices))
    indices_list = [(i, i + seq_len) for i in I]
    return indices_list


def get_sequence(
    data, target, seq_len, overlap, training=True, upsampling_0=False, seed=985
) -> pd.DataFrame:
    sequenced_data = pd.DataFrame()
    GROUP = []
    LABEL = []
    TEXT = []
    LEMMA = []
    POS = []
    MORPH = []
    for i in range(len(data)):
        line = data.iloc[i]
        word_list = line["token"]
        lemma_list = line["lemma"]
        pos_list = line["pos"]
        morph_list = line["morph"]
        label = line[target]
        code = line["code"]
        if training:
            indices_list = get_train_indices_list(
                word_list=word_list,
                seq_len=seq_len,
                label=label,
                upsampling_0=upsampling_0,
                seed=seed,
            )
            # print(code, indices_list)
        else:
            indices_list = get_indices_list(word_list=word_list)

        for indices in indices_list:
            start = indices[0]
            end = indices[1]
            text = " ".join(word_list[start:end])
            lemmas = lemma_list[start:end]
            poss = pos_list[start:end]
            morphs = morph_list[start:end]
            TEXT.append(text)
            LEMMA.append(lemmas)
            POS.append(poss)
            MORPH.append(morphs)
            LABEL.append(label)
            GROUP.append(code)
    # store in dataframe
    sequenced_data["text"], sequenced_data["label"], sequenced_data["group"] = (
        TEXT,
        LABEL,
        GROUP,
    )
    sequenced_data["lemma"], sequenced_data["pos"], sequenced_data["morph"] = (
        LEMMA,
        POS,
        MORPH,
    )

    return sequenced_data.sample(frac=1).reset_index(drop=True)


"""
['score_troncations_matches', 'ON_NOUS', 'PRESENT_ENNONCIATION',
       'passive_count', 'degree_average', 'passive_count_norm',
       'PRESENT_HISTORIQUE', 'verb_indicatif_present',
       'VERB_PERCEPTIONS_SENSORIELLES', 'PE', 'LSC', 'L1',
       'score_temporal_connector_matches', 'MORT_EXPLICITE', 'CORPS',
       'passive_sents_count', 'diameter_g0', 'average_shrotest_path_g0',
       'ON_GENERIQUE', 'SENSATIONS_PHYSIQUES', 'L3',
       'score_generical_connector_matches', 'L2', 'score_paticules_matches',
       'PRESENT_GENERIQUE']
    """
