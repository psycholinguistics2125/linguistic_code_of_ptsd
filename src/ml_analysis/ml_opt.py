
import random
import logging
import os
import pandas as pd
import pickle

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

from src.ml_analysis.ml_utils import compute_k_models, compute_average_scores


rf_search_space = {'n_estimators':hp.randint('n_estimators',5,100),
                'max_depth': hp.randint('max_depth',5,45),
                'min_samples_split':hp.uniform('min_samples_split',0,1), 
                'min_samples_leaf':hp.randint('min_samples_leaf',1,20),
                'criterion':hp.choice('criterion',['gini','entropy']),
                'max_features':hp.choice('max_features',['sqrt', 'log2']),
                'class_weight':hp.choice('class_weight',['balanced', {0:1,1:5}, {0:1,1:10},None]),
                'data_aug':hp.choice('data_aug',[True,False]),
                'scaler':hp.choice('scaler',[True,False]),
                 }

brf_search_space = {'n_estimators':hp.randint('n_estimators',5,100),
                'max_depth': hp.randint('max_depth',5,25),
                'min_samples_split':hp.uniform('min_samples_split',0,1), 
                'min_samples_leaf':hp.randint('min_samples_leaf',1,20),
                'criterion':hp.choice('criterion',['gini','entropy']),
                'max_features':hp.choice('max_features',['sqrt', 'log2']),
                'class_weight':hp.choice('class_weight',['balanced', {0:1,1:5}, {0:1,1:10},None]),
                'data_aug':hp.choice('data_aug',[True,False]),
                'scaler':hp.choice('scaler',[True,False]),
                 }

lr_search_space = {'penalty':hp.choice('penalty',['l1', 'l2','elasticnet','none']),
                'C': hp.uniform('C',0,5),
                'fit_intercept':hp.choice('fit_intercept',[True,False]),
                'l1_ratio' : hp.uniform('l1_ratio',0,1),
                'solver':hp.choice('solver',['saga']),
                'class_weight':hp.choice('class_weight',['balanced', {0:1,1:5}, {0:1,1:10}]),
                'data_aug':hp.choice('data_aug',[True,False]),
                'scaler':hp.choice('scaler',[True]),
                 }

lgb_search_space = {'n_estimators':hp.randint('n_estimators',5,50),
                'max_depth': hp.randint('max_depth',5,35),
                'reg_alpha': hp.uniform('reg_alpha',0,1),
                'reg_lambda': hp.uniform('reg_lambda',0,1),
                'min_split_gain':hp.uniform('min_split_gain',0,1), 
                'min_child_samples':hp.randint('min_child_samples',5,30),
                'feature_fraction': hp.uniform('feature_fraction',0,1),
                'class_weight':hp.choice('class_weight',['balanced', {0:1,1:5}, {0:1,1:10}]),
                'data_aug':hp.choice('data_aug',[True,False]),
                'scaler':hp.choice('scaler',[True,False]),
                 }

ebm_search_space = {'max_bins':hp.randint('max_bins',5,50),
                'max_rounds':hp.choice('max_rounds',[100,200,300]),
                #'binning': hp.choice('binning',["uniform","quantile"]),
                'learning_rate': hp.uniform('learning_rate',0.0001,0.01),
                'interactions':hp.choice('interactions',[0,5,10]),
                'min_samples_leaf': hp.randint('min_samples_leaf',2,30),
                'max_leaves': hp.randint("max_leaves",2,10),
                "early_stopping_rounds" :  hp.randint("early_stopping_rounds",10,50),
                'data_aug':hp.choice('data_aug',[True]),
                'scaler':hp.choice('scaler',[True]),
                 }

dt_search_space = {
                'max_depth': hp.randint('max_depth',5,25),
                'min_samples_split':hp.uniform('min_samples_split',0,1), 
                'min_samples_leaf':hp.randint('min_samples_leaf',1,20),
                'criterion':hp.choice('criterion',['gini','entropy']),
                'ccp_alpha' :hp.uniform('ccp_alpha',0,1), 
                'max_features':hp.choice('max_features',['sqrt', 'log2']),
                'class_weight':hp.choice('class_weight',['balanced', {0:1,1:5}, {0:1,1:10},None]),
                'data_aug':hp.choice('data_aug',[True,False]),
                'scaler':hp.choice('scaler',[True,False]),
                 }

search_space_dict = {
                    "lr":lr_search_space,
                    "rf" :rf_search_space,
                    "brf" : brf_search_space,
                    "lgb" : lgb_search_space,
                    "ebm" : ebm_search_space,
                    "dt" : dt_search_space
                    }




def objective(search_space, config, data, model_type, score_name = "balanced_accuracy",phase = 1):
    #print(search_space)
    config['ml_analysis']['scaler'] = search_space['scaler']
    config['ml_analysis']['data_aug'] = search_space['data_aug']
    config['ml_analysis'][model_type] = {key:search_space[key] for key in list(search_space.keys()) if key not in ['data_aug','scaler']}
    config["ml_analysis"]["best_param"] = False
    results = compute_k_models(data,config,model_type,k = 100,phase=phase)
    scores = compute_average_scores(results,config, plot = False , save = False)
    
    obj =  scores[score_name].mean()

    return {'loss': -obj, 'status': STATUS_OK}





def finetune_models_on_target(models_list, target, config, data, max_evals = 200,logger = logging.getLogger(),  score_name = "balanced_accuracy",phase =1) :
    # setting up the configuration
    config['ml_analysis']["target"] = target
    saving_folder = os.path.join(config['ml_analysis']['ml_folder'],f"{target}_finetunning")
    if not os.path.exists(saving_folder) :
        os.mkdir(saving_folder)
        logger.info(f"folder, {saving_folder} has been created")
    
    logger.info(f"folder, {saving_folder} will be used for saving results ! ")

    best_scores = []
    synthesis = pd.DataFrame()
    for model_type in models_list :
        logger.info(f"Beginning optimization for {model_type} on {target} ")
        trials = Trials()
        search_space = search_space_dict[model_type]
        
        fmin_objective = lambda x : objective(x, config = config, data = data, model_type = model_type,score_name= score_name,phase = phase)
        
        best_params = fmin(
                fn=fmin_objective, 
                space=search_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

        best_param = pd.DataFrame(space_eval(search_space, best_params), index =[model_type])
        best_param.to_csv(os.path.join(saving_folder,f"{model_type}_best_parameters_{score_name}.csv"),index =False)
        trial_path = os.path.join(saving_folder,f"{model_type}_trials.pkl")
        pickle.dump(trials, open(trial_path, "wb"))
        
        logger.info(f'Best parameters and trials save in {saving_folder} !')
        best_scores.append(- fmin_objective(space_eval(search_space, best_params))['loss'])

    synthesis['models_type'] = models_list
    synthesis['scores'] = best_scores
    models_names = "_".join(models_list)
    synthesis.to_csv(os.path.join(saving_folder,f"{models_names}_synthesis_best_parameters_{score_name}.csv"),index = False)

    print(synthesis)
