# read the loaded data
# save it in the data/raw for further process
import os
from get_data import read_params
import argparse
import pandas as pd
import logging
from joblib import Parallel, delayed
from statsmodels.stats.outliers_influence import variance_inflation_factor
import time

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def preprocess_and_save(config_path):
    """
    This functions drops those columns whose values are more than threshold VIF passed as parameter.

    Parameters
    ----------
    vif_thresh: This is the threshold VIF value above which dataset column will be dropped.

    Returns:
    ----------
    None.
    """
    try:
        config = read_params(config_path)
        vif_thresh = config["base"]["vif_threshold"]
        logging.info('All features with VIF more than {} will be dropped from the dataset.'.format(vif_thresh))
        raw_data_path = config["load_data"]["raw_dataset_csv"]
        preprocessed_data_path = config["preprocess_data"]["preprocessed_dataset_csv"]
        df = pd.read_csv(raw_data_path, sep=",")
        df_copy = df.copy()
        target = config["base"]["target_col"].replace(" ", "_")
        variables = [df_copy.columns[i] for i in range(df_copy.shape[1])]
        dropped = True
        while dropped:
            dropped = False
            vif = Parallel(n_jobs=-1, verbose=5)(
                delayed(variance_inflation_factor)(df_copy[variables].values, ix) for ix in range(len(variables)))

            maxloc = vif.index(max(vif))
            if max(vif) > vif_thresh:
                print(df_copy[variables].columns[maxloc])
                print(target)
                if df_copy[variables].columns[maxloc] != target:
                    logging.info(
                        time.ctime() + ' dropping \'' + df_copy[variables].columns[maxloc] + '\' at index: ' + str(
                            maxloc))
                    variables.pop(maxloc)
                    dropped = True

        logging.info('Remaining variables:')
        logging.info([variables])
        final_df = df_copy[[i for i in variables]]
        final_df.to_csv(preprocessed_data_path, sep=",", index=False)
    except Exception as e:
        logging.error(
            "{} occured while droping some of the feature from dataset based on vif threshold.".format(str(e)))

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    preprocess_and_save(config_path=parsed_args.config)