import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PhenPred import OMIC_PALETTE, OMIC_NAMES
from PhenPred.vae import plot_folder, data_folder


def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if np.pi / 2 <= angle <= 3 * np.pi / 2:
        alignment = "right"
        rotation = rotation + 180
    else:
        alignment = "left"
    return rotation, alignment


def add_labels(angles, values, labels, offset, ax):
    # This is the space between the end of the bar and the label
    padding = 1

    # Iterate over angles, values, and labels, to add all of them.
    for (
        angle,
        value,
        label,
    ) in zip(angles, values, labels):
        angle = angle

        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle,
            y=value + padding,
            s=label,
            ha=alignment,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=5,
        )


if __name__ == "__main__":
    # Import feature importance
    df = pd.read_csv(f"{plot_folder}/latent_shap_top10_no_tag.csv")
    df = df.rename(
        columns={"feature": "name", "importance": "value", "omic_layer": "group"}
    )
    df["name"] = [
        v.split(";")[1] if ";" in v else v
        for v in df["name"].replace(
            {
                "tissue_Haematopoietic and Lymphoid": "Haem. and Lymph.",
                "tissue_Lung": "Lung",
                "tissue_Skin": "Skin",
            }
        )
    ]

    hue_order = list(
        df.sort_values("value", ascending=False)
        .groupby("group")
        .first()
        .sort_values("value", ascending=False)
        .index
    )
    df = pd.concat(
        [
            df.query(f"group =='{g}'").sort_values("value", ascending=False)
            for g in hue_order
        ]
    )

    # Values
    VALUES = df["value"].values
    LABELS = df["name"].values

    # Grab the group values
    GROUP = df["group"].values

    # Add three empty bars to the end of each group
    PAD = 1
    ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
    WIDTH = (2 * np.pi) / len(ANGLES)

    # Obtain size of each group
    GROUPS_SIZE = [len(i[1]) for i in df.groupby("group")]

    # Determines where to place the first bar.
    # By default, matplotlib starts at 0 (the first bar is horizontal)
    # but here we say we want to start at pi/2 (90 deg)
    OFFSET = 0

    # Obtaining the right indexes is now a little more complicated
    offset = 0
    IDXS = []
    for size in GROUPS_SIZE:
        IDXS += list(range(offset + PAD, offset + size + PAD))
        offset += size + PAD

    # Same layout as above
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"}, dpi=300)

    ax.set_theta_offset(OFFSET)
    ax.set_ylim(-5, 10)
    ax.set_frame_on(False)

    ax.xaxis.grid(False)
    ax.set_xticks([])

    ax.yaxis.grid(True, color="gray", alpha=0.5, ls="-", lw=0.3, zorder=-1)
    ax.set_yticks([0, 5])
    ax.set_yticklabels([0, 5], fontsize=6)
    # set y tick labels in the horizontal line
    ax.set_rlabel_position(0)

    # Use different colors for each group!
    GROUPS_SIZE = [len(i[1]) for i in df.groupby("group")]
    COLORS = [
        OMIC_PALETTE[g]
        for i, g in enumerate(df["group"].unique())
        for _ in range(GROUPS_SIZE[i])
    ]

    # And finally add the bars.
    # Note again the `ANGLES[IDXS]` to drop some angles that leave the space between bars.
    ax.bar(
        ANGLES[IDXS],
        VALUES,
        width=WIDTH,
        color=COLORS,
        edgecolor="white",
        linewidth=0.5,
    )

    add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)

    # add legend middle of the plot
    legend = ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=OMIC_PALETTE[g]) for g in np.unique(GROUP)
        ],
        labels=[OMIC_NAMES[g] for g in np.unique(GROUP)],
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=1,
        frameon=False,
        fontsize=7,
    ).remove()

    # for t in legend.get_texts():
    #     t.set_ha("center")  # ha is alias for horizontalalignment

    PhenPred.save_figure(f"{plot_folder}/latent_shap_circular_top10")
