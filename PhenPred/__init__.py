#!/usr/bin/env python
# Copyright (C) 2022 Emanuel Goncalves

import matplotlib.pyplot as plt

# Matplotlib config
plt.rcParams["figure.figsize"] = [4, 4]
plt.rcParams["figure.dpi"] = 300

# Matplotlib set font to sans-times
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

# Matplotlib set main axis font size
plt.rcParams["axes.titlesize"] = 7

# Matplotlib set legend font size
plt.rcParams["legend.fontsize"] = 6

# Matplotlib set legend title font size
plt.rcParams["legend.title_fontsize"] = 6

# Matplotlib set tick label font size
plt.rcParams["axes.labelsize"] = 6

# Matplotlib set tick label font size
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6

# Matplotlib set grid line width
plt.rcParams["grid.linewidth"] = 0.5

# Matplotlib ommit top and right spines
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Matplotlib set grid line
plt.rcParams["axes.grid"] = True

# Matplotlib set grid line style
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.15

# Matplotlib set grid line color
plt.rcParams["grid.color"] = "black"

# Matplotlib set grid line alpha
plt.rcParams["grid.alpha"] = 0.5

# Matplotlib set legend frameon
plt.rcParams["legend.frameon"] = False

# Matplotlib set legend loc
plt.rcParams["legend.loc"] = "best"

# Matplotlib set axis below true
plt.rcParams["axes.axisbelow"] = True

# Matplotlib illustrator export
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Palette - tissue type
PALETTE_TTYPE = {
    "Lung": "#007fff",
    "Prostate": "#665d1e",
    "Stomach": "#ffbf00",
    "Central Nervous System": "#fbceb1",
    "Skin": "#ff033e",
    "Bladder": "#ab274f",
    "Haematopoietic and Lymphoid": "#d5e6f7",
    "Kidney": "#7cb9e8",
    "Thyroid": "#efdecd",
    "Soft Tissue": "#8db600",
    "Head and Neck": "#e9d66b",
    "Ovary": "#b284be",
    "Bone": "#b2beb5",
    "Endometrium": "#10b36f",
    "Breast": "#6e7f80",
    "Pancreas": "#ff7e00",
    "Peripheral Nervous System": "#87a96b",
    "Cervix": "#c9ffe5",
    "Large Intestine": "#9f2b68",
    "Liver": "#00ffff",
    "Vulva": "#008000",
    "Esophagus": "#cd9575",
    "Biliary Tract": "#72a0c1",
    "Other tissue": "#a32638",
    "Small Intestine": "#9966cc",
    "Placenta": "#f19cbb",
    "Testis": "#e32636",
    "Adrenal Gland": "#3b7a57",
    "Uterus": "#7a3b5e",
    "Unknown": "#a32638",
    "Eye": "#ff1493",
}

OMIC_PALETTE = {
    "conditionals": "#4c72b0",
    "copynumber": "#dd8452",
    "drugresponse": "#55a868",
    "metabolomics": "#c44e52",
    "proteomics": "#8172b3",
    "crisprcas9": "#937860",
    "transcriptomics": "#da8bc3",
    "methylation": "#8c8c8c",
}


def save_figure(path, extensions=["png", "pdf"], dpi=300, transparent=False, **kwargs):
    for e in extensions:
        plt.savefig(
            path + "." + e,
            bbox_inches="tight",
            dpi=dpi,
            transparent=transparent,
            **kwargs,
        )
    plt.close("all")
