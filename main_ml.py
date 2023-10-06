
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import pandas as pd


import logging
import logging.config
import src.utils as utils

from src.data_tools.data_utils import  load_features_and_meta
from src.ml_analysis import ml_utils

logging.config.dictConfig(utils.load_config("logging_config.yaml"))



if __name__ == "__main__":
    logger = logging.getLogger()

    N_tries = int(sys.argv[1])
    if not isinstance(N_tries, int) :
        N_tries = 100

    logger.info(f"Training and saving the experiments for N = {N_tries}")
    #load the features data
    config = utils.load_config()
    data = load_features_and_meta(config)
    data['code'] = data['uuid']
    
 
    logger.info("#"*20)
    source_folder = config["data"]['data_folder']
    logger.info(f"Data loaded from file: {source_folder}")
    logger.info("#"*20)
    
    
    models_list =  ["lr","rf","ebm", 'lasso'] # ["brf",'lr','rf',"lasso","elasticnet"]
    targets_list =  ["full_or_partial_PTSD", 'CB_probable','CC_probable', 'CD_probable','CE_probable','CG_probable'] # "full_or_partial_PTSD", "full_or_partial_PTSD", 'CB_probable','CC_probable', 'CD_probable
    synthesis = ml_utils.compute_all_average(data,config,try_number = N_tries, plot = True, save = True, top_features = 20, models_list = models_list, logger = logger,targets_list = targets_list) 