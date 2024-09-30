
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
    #print(config)
    phase = "2"

    data = load_features_and_meta(config,phase=phase,exp_critereA = "A1A2")
    if "uuid" in data.columns :
        data['code'] = data['uuid']
    if "code" in data.columns :
        data['uuid'] = data['code']
   

    if config['ml_analysis']['filter']=="A1":
        data = data[data['exp_critereA']=="A1"].reset_index(drop=True)
    elif config['ml_analysis']['filter']=="A2":
        data = data[data['exp_critereA']=="A2"].reset_index(drop=True)
    else :
        pass
    
    # filter
    #data =  data[~data.apply(lambda row: row.astype(str).str.contains("STADE DE FRANCE").any(), axis=1)].reset_index(drop=True)

    logger.info("#"*20)
    logger.info("Data loaded")
    logger.info(f"There are {len(data)} samples")
    logger.info("#"*20)
 
    logger.info("#"*20)
    source_folder = config["data"]['data_folder']
    logger.info(f"Data loaded from file: {source_folder}")
    logger.info("#"*20)
    
    
    models_list =  ["lr","rf","ebm", 'lasso'] # ["brf",'lr','rf',"lasso","elasticnet"]
    targets_list =  ["full_or_partial_PTSD_probable",'CB_probable','CC_probable', 'CD_probable','CE_probable','CG_probable']#["full_or_partial_PTSD_probable", 'CE_probable','CG_probable'] # 'CB_probable','CC_probable', 'CD_probable' "full_or_partial_PTSD", "full_or_partial_PTSD", 'CB_probable','CC_probable', 'CD_probable
    synthesis = ml_utils.compute_all_average(data,config,try_number = N_tries, plot = True, save = True, top_features = 20, models_list = models_list, logger = logger,targets_list = targets_list,phase = phase) 

    for target in ["PTSD_probable"]:
        config["ml_analysis"]["strati"] = target
        targets_list = [target]
        synthesis = ml_utils.compute_all_average(data,config,try_number = N_tries, plot = True, save = True, top_features = 20, models_list = models_list, logger = logger,targets_list = targets_list,phase = phase) 