import os

# Data dir path - INESC-ID server
data_folder = "/data/benchmarks/clines/"

if not os.path.exists(data_folder):
    data_folder = "data/clines/"

# Plot dir path - INESC-ID server
plot_folder = "/home/egoncalves/PhenPred/reports/vae/"

if not os.path.exists(plot_folder):
    plot_folder = "reports/vae/"

# SHAP dir path - on mounted drive
shap_folder = "/home/scai/projects/E0160_P01_PhenPred/reports/vae/"
if not os.path.exists(shap_folder):
    shap_folder = plot_folder

# Files folder path - INESC-ID server
files_folder = "/home/egoncalves/PhenPred/data/"

if not os.path.exists(files_folder):
    files_folder = "data/"
