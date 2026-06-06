from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
import os
import numpy as np
import pandas as pd
from load_datar import bkg_fit, x_train, y_train, x_test, y_test
from config import model_dir, plot_dir, root_dir, log_dir


# loading of the trained model
model_hdl = ModelHandler()
model_hdl.load_model_handler(os.path.join(model_dir, "model_p.pkl"))

# creation of parquet file for real data
parquet_file = os.path.join(model_dir,"p_output.parquet.gzip")

#  Check if the parquet file already exists, if not create it
if not os.path.exists(parquet_file):
    print("\nCreating parquet file...\n")
    handler = bkg_fit
    handler.apply_model_handler(model_hdl, output_margin=False)
    handler.write_df_to_parquet_files(os.path.join(model_dir, "p_output"))
    print("\nParquet created.\n")


# creation of parquet file for test data
test_parquet = os.path.join(model_dir,"testp_output_templates.parquet.gzip")
if not os.path.exists(test_parquet):
    print("\nCreating test parquet file...\n")
    df_test = pd.DataFrame(x_test.copy())
    df_test["label"] = y_test
    df_test = df_test.copy()
    test_hdl = TreeHandler()
    test_hdl.set_data_frame(df_test)
    test_hdl.apply_model_handler(model_hdl, output_margin=False)
    test_hdl.write_df_to_parquet_files(os.path.join(model_dir, "testp_output"))
    print("\nTest parquet created.\n")
else:
    print("\nLoading test parquet...\n")
    df_test = pd.read_parquet(test_parquet)
