import os

# Data dir path - INESC-ID server
data_folder = "/data/benchmarks/clines/"

if not os.path.exists(data_folder):
    data_folder = "data/clines/"

# Plot dir path - INESC-ID server
plot_folder = "/home/egoncalves/PhenPred/reports/vae/"

if not os.path.exists(plot_folder):
    plot_folder = "reports/vae/"
