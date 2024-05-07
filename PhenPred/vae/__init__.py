import os
import sys

proj_dir = os.getcwd()

if proj_dir not in sys.path:
    sys.path.append(proj_dir)

files_folder = f"{proj_dir}/data/"
data_folder = f"{proj_dir}/data/clines/"
plot_folder = f"{proj_dir}/reports/vae/"
shap_folder = plot_folder
