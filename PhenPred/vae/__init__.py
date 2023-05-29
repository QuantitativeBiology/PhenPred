import os

# Data dir path - INESC-ID server
data_folder = "/data/benchmarks/clines/"

if not os.path.exists(data_folder):
    data_folder = "data/clines/"

# Plot dir path - INESC-ID server
plot_folder = "/home/scai/PhenPred/reports/vae/"

if not os.path.exists(plot_folder):
    plot_folder = "reports/vae/"

# Files folder path - INESC-ID server
files_folder = "/home/scai/PhenPred/data/"

if not os.path.exists(files_folder):
    files_folder = "data/"

# Log folder path - INESC-ID server
logs_folder = "/home/scai/PhenPred/logs/"

if not os.path.exists(logs_folder):
    logs_folder = "logs/"
