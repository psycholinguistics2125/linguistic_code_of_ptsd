
""" 
Author: XXX
created date: 2022, 7 september
"""
import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np

import os

import logging
import logging.config
import src.utils as utils



from src.data_tools.data_utils import get_features_name, load_features_and_meta
from src.stats_analysis import utils_statistics

logging.config.dictConfig(utils.load_config("logging_config.yaml"))

if __name__ == "__main__":
    logger = logging.getLogger()
    #load the features data
    config = utils.load_config()
    data = load_features_and_meta(config)

    source_folder = config["data"]['data_folder']

    logger.info("#"*15)
    logger.info(f"Data loaded from file: {source_folder}")
    logger.info(f"Data shape: {data.shape}")
    logger.info("#"*15)

    cibles = config["stats_analysis"]['target']
    features_col_names = [col for col in data.select_dtypes(np.number).columns if col not in cibles+['uuid']+["exp_critereA","Unnamed: 0"]]
    seuil = config['stats_analysis']['seuil']
    power = config['stats_analysis']['power']
    saving_folder = config['stats_analysis']['saving_folder']

    logger.info("#"*15)
    logger.info(f"Computed statistics on {cibles} using {seuil} as pvalue limit")
    logger.info(f"results will be saved in {saving_folder}")
    logger.info("#"*15)
    
    for col_cible in cibles : 
        logger.info("#"*50)
        logger.info(col_cible)
        if col_cible == "full_and_partial_PTSD":
            result = utils_statistics.compute_anova_table(data,[col_cible],features_col_names,seuil =  seuil).sort_values("p-unc")
            name="anova"
        else :
            result = utils_statistics.compute_mwu_table(data,[col_cible],features_col_names,seuil =  seuil).sort_values("p-val")
            name= "mwu"

       
        _, result_name = get_features_name(config,name)
        result = result[result['power']>power]
        result.to_csv(os.path.join(saving_folder,f"{col_cible}_{result_name}"))