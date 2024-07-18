
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import os

def read_csv(
    file_name: str,
    *args,
    **kwargs,
):
    """
    Reads the file_name.csv which is specified.
    Note that a dotenv here helps if the tables are stored in an external directory.
    """
    file_path = f"{file_name}.csv" if not file_name.endswith('.csv') else file_name
    
    load_dotenv(override=True)
    if 'TABLES_DIR' in os.environ:
        file_path = os.path.join(os.environ['TABLES_DIR'], file_path)
    
    return pd.read_csv(file_path, *args, **kwargs)
        
def parse_table(df):
    """
    This code parses the tables that have a "name" column associated with OOD tasks.
    
    For example, if you have a value in the name column equal to "emnist_vs_emnist_test_...", this means that
    you have the results for a task where it is being tested on emnist_test datapoints with the emnist being the reference
    dataset to consider.
    
    What this piece of code does is that it parses the entire table and returns a dictionary of tables where for each task
    "dataset1"_vs_"dataset2"
    ret[dataset1][dataset2] is a table containing a columns "name" with the following values:
    
    1. "train": This relates to the datapoints in the train split of dataset1
    2. "test": This relates to the datapoints in the test split of dataset1
    3. "generated": This relates to the datapoints generated with the model
    4. "ood": This relates to the out_of_distribution datapoints
    """
    all_first_part_keys = set()
    all_second_part_keys = set()
    for x in df["name"]:
        first_part = x.split("_")[0]
        all_first_part_keys.add(first_part)
        second_part = x.split("_")[2]
        all_second_part_keys.add(second_part) 
        
    
    ret = {}
    rng = tqdm(all_first_part_keys)
    for first_part in rng:
        idx = 0
        for second_part in all_second_part_keys:
            if first_part == second_part or second_part == "dgm-generated":
                continue
            idx += 1
            tt = len(all_second_part_keys)
            
            rng.set_description(f"{idx}/{tt}")
            # filter the rows of df so that the name follows the convention: "first_part"_vs_"second_part"_...
            filter = df["name"].str.startswith(first_part + "_vs_" + second_part)
            filter = filter | df["name"].str.startswith(first_part + "_vs_" + first_part + "_train")
            filter = filter | df["name"].str.startswith(first_part + "_vs_" + first_part + "_test")
            filter = filter | df["name"].str.startswith(first_part + "_vs_dgm-generated")
            filtered_df = df[filter]
        
            df_content = {}
            for column_name in filtered_df.columns:
                if column_name in ["_defaultColorIndex", "id"]:
                    continue
                df_content[column_name] = []
                
            for i in range(len(filtered_df)):
                row = filtered_df.iloc[i]
                name = row["name"]
                first, _, second, train_or_test, _ =  name.replace('random_image', 'random-image').split("_")
                if second == first:
                    if train_or_test == "train":
                        df_content["name"].append("train")
                    else:
                        df_content["name"].append("test")
                elif second == "dgm-generated":
                    df_content["name"].append("generated")
                else:
                    df_content["name"].append("ood")
                
                for key in df_content.keys():
                    if key == "name":
                        continue
                    df_content[key].append(row[key])

            if first_part not in ret:
                ret[first_part] = {}
            ret[first_part][second_part] = pd.DataFrame(df_content)
    
    return ret
