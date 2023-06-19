import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from math import sqrt
from natsort import natsorted
from adjustText import adjust_text
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.interpolate import interpn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class CrispyPlot:
    # PLOTING PROPS
    SNS_RC = {
        "axes.linewidth": 0.3,
        "xtick.major.width": 0.3,
        "ytick.major.width": 0.3,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }

    # PALETTES
    PAL_DBGD = {0: "#656565", 1: "#F2C500", 2: "#E1E1E1", 3: "#0570b0"}
    PAL_SET1 = sns.color_palette("Set1", n_colors=9).as_hex()
    PAL_SET2 = sns.color_palette("Set2", n_colors=8).as_hex()
    PAL_DTRACE = {
        0: "#E1E1E1",
        1: PAL_SET2[1],
        2: "#656565",
        3: "#2b8cbe",
        4: "#de2d26",
    }

    PAL_YES_NO = {"No": "#E1E1E1", "Yes": PAL_SET2[1]}

    PAL_TISSUE = {
        "Lung": "#c50092",
        "Haematopoietic and Lymphoid": "#00ca5b",
        "Large Intestine": "#c100ce",
        "Central Nervous System": "#c3ce44",
        "Skin": "#d869ff",
        "Breast": "#496b00",
        "Head and Neck": "#ff5ff2",
        "Bone": "#004215",
        "Esophagus": "#fa006f",
        "Ovary": "#28d8fb",
        "Peripheral Nervous System": "#ff2c37",
        "Kidney": "#5ea5ff",
        "Stomach": "#fcbb3b",
        "Bladder": "#3e035f",
        "Pancreas": "#f6bc62",
        "Liver": "#df93ff",
        "Thyroid": "#ae7400",
        "Soft Tissue": "#f5aefa",
        "Cervix": "#008464",
        "Endometrium": "#980018",
        "Biliary Tract": "#535d3c",
        "Prostate": "#ff80b6",
        "Uterus": "#482800",
        "Vulva": "#ff7580",
        "Placenta": "#994800",
        "Testis": "#875960",
        "Small Intestine": "#fb8072",
        "Adrenal Gland": "#ff8155",
        "Eye": "#fa1243",
        "Other": "#000000",
    }

    PAL_TISSUE_2 = dict(
        zip(
            *(
                natsorted(list(PAL_TISSUE)),
                sns.color_palette("tab20c").as_hex()
                + sns.color_palette("tab20b").as_hex(),
            )
        )
    )

    CANCER_TYPES = [
        "Esophageal Carcinoma",
        "Colorectal Carcinoma",
        "Hepatocellular Carcinoma",
        "Esophageal Squamous Cell Carcinoma",
        "Ovarian Carcinoma",
        "Other Solid Cancers",
        "Glioma",
        "Pancreatic Carcinoma",
        "Thyroid Gland Carcinoma",
        "Neuroblastoma",
        "Non-Cancerous",
        "Melanoma",
        "Oral Cavity Carcinoma",
        "Small Cell Lung Carcinoma",
        "Head and Neck Carcinoma",
        "Endometrial Carcinoma",
        "Glioblastoma",
        "Kidney Carcinoma",
        "Plasma Cell Myeloma",
        "Prostate Carcinoma",
        "T-Cell Non-Hodgkin's Lymphoma",
        "Breast Carcinoma",
        "Mesothelioma",
        "Non-Small Cell Lung Carcinoma",
        "T-Lymphoblastic Leukemia",
        "Biliary Tract Carcinoma",
        "Ewing's Sarcoma",
        "Squamous Cell Lung Carcinoma",
        "Gastric Carcinoma",
        "B-Lymphoblastic Leukemia",
        "Chondrosarcoma",
        "Bladder Carcinoma",
        "Cervical Carcinoma",
        "Osteosarcoma",
        "Acute Myeloid Leukemia",
        "B-Cell Non-Hodgkin's Lymphoma",
        "Other Sarcomas",
        "Chronic Myelogenous Leukemia",
        "Hodgkin's Lymphoma",
        "Burkitt's Lymphoma",
        "Other Blood Cancers",
        "Rhabdomyosarcoma",
        "Unknown",
        "Acute Monocytic Leukemia",
    ]
    PAL_CANCER_TYPE = dict(
        zip(
            *(
                natsorted(CANCER_TYPES),
                sns.color_palette("tab20c").as_hex()
                + sns.color_palette("tab20b").as_hex()
                + sns.light_palette(PAL_SET1[1], n_colors=5).as_hex()[:-1],
            )
        )
    )
    PAL_CANCER_TYPE["Pancancer"] = PAL_SET1[5]

    PAL_MODEL_TYPE = {**PAL_CANCER_TYPE, **PAL_TISSUE_2}

    PAL_GROWTH_CONDITIONS = {
        "Adherent": "#fb8072",
        "Semi-Adherent": "#80b1d3",
        "Suspension": "#fdb462",
        "Unknown": "#d9d9d9",
    }

    PAL_MSS = {"MSS": "#d9d9d9", "MSI": "#fb8072"}

    SV_PALETTE = {
        "tandem-duplication": "#377eb8",
        "deletion": "#e41a1c",
        "translocation": "#984ea3",
        "inversion": "#4daf4a",
        "inversion_h_h": "#4daf4a",
        "inversion_t_t": "#ff7f00",
    }

    PPI_PAL = {
        "T": "#fc8d62",
        "1": "#656565",
        "2": "#7c7c7c",
        "3": "#949494",
        "4": "#ababab",
        "5+": "#c3c3c3",
        "-": "#2b8cbe",
        "X": "#2ca02c",
    }

    PPI_ORDER = ["T", "1", "2", "3", "4", "5+", "-"]

    GENESETS = ["essential", "nonessential", "nontargeting"]
    GENESETS_PAL = dict(
        essential="#e6550d", nonessential="#3182bd", nontargeting="#31a354"
    )

    # BOXPLOT PROPOS
    BOXPROPS = dict(linewidth=1.0)
    WHISKERPROPS = dict(linewidth=1.0)
    MEDIANPROPS = dict(linestyle="-", linewidth=1.0, color="red")
    FLIERPROPS = dict(
        marker="o",
        markerfacecolor="black",
        markersize=2.0,
        linestyle="none",
        markeredgecolor="none",
        alpha=0.6,
    )

    # CORRELATION PLOT PROPS
    ANNOT_KWS = dict(stat="R")
    MARGINAL_KWS = dict(kde=False, hist_kws={"linewidth": 0})

    LINE_KWS = dict(lw=1.0, color=PAL_DBGD[1], alpha=1.0)
    SCATTER_KWS = dict(edgecolor="w", lw=0.3, s=10, alpha=0.6, color=PAL_DBGD[0])
    JOINT_KWS = dict(lowess=True, scatter_kws=SCATTER_KWS, line_kws=LINE_KWS)

    @staticmethod
    def get_palette_continuous(n_colors, color=PAL_DBGD[0]):
        pal = sns.light_palette(color, n_colors=n_colors + 2).as_hex()[2:]
        return pal

    @staticmethod
    def density_interpolate(xx, yy, dtype="gaussian"):
        if dtype == "gaussian":
            xy = np.vstack([xx, yy])
            zz = gaussian_kde(xy)(xy)

        else:
            data, x_e, y_e = np.histogram2d(xx, yy, bins=20)

            zz = interpn(
                (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                data,
                np.vstack([xx, yy]).T,
                method="splinef2d",
                bounds_error=False,
            )

        return zz

    @classmethod
    def get_palettes(cls, samples, samplesheet):
        samples = set(samples).intersection(samplesheet.index)

        pal_tissue = {
            s: cls.PAL_TISSUE_2[samplesheet.loc[s, "tissue"]] for s in samples
        }

        pal_growth = {
            s: cls.PAL_GROWTH_CONDITIONS[samplesheet.loc[s, "growth_properties"]]
            for s in samples
        }

        palettes = pd.DataFrame(dict(tissue=pal_tissue, media=pal_growth))

        return palettes

    @classmethod
    def triu_plot(cls, x, y, color, label, **kwargs):
        df = pd.DataFrame(dict(x=x, y=y)).dropna()
        ax = plt.gca()
        ax.hexbin(
            df["x"], df["y"], cmap="Spectral_r", gridsize=30, mincnt=1, bins="log", lw=0
        )
        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    @classmethod
    def triu_scatter_plot(cls, x, y, color, label, **kwargs):
        df = pd.DataFrame(dict(x=x, y=y)).dropna()
        df["z"] = cls.density_interpolate(df["x"], df["y"], dtype="interpolate")
        df = df.sort_values("z")

        ax = plt.gca()

        ax.scatter(
            df["x"],
            df["y"],
            c=df["z"],
            marker="o",
            edgecolor="",
            s=5,
            alpha=0.8,
            cmap="Spectral_r",
        )

        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    @classmethod
    def diag_plot(cls, x, color, label, **kwargs):
        sns.distplot(
            x[~np.isnan(x)], label=label, color=CrispyPlot.PAL_DBGD[0], kde=False
        )

    @classmethod
    def attenuation_scatter(
        cls,
        x,
        y,
        plot_df,
        z="cluster",
        zorder=None,
        pal=None,
        figsize=(2.5, 2.5),
        plot_reg=True,
        ax_min=None,
        ax_max=None,
    ):
        if ax_min is None:
            ax_min = plot_df[[x, y]].min().min() * 1.1

        if ax_max is None:
            ax_max = plot_df[[x, y]].max().max() * 1.1

        if zorder is None:
            zorder = ["High", "Low"]

        if pal is None:
            pal = dict(High=cls.PAL_DTRACE[1], Low=cls.PAL_DTRACE[0])

        g = sns.jointplot(
            x,
            y,
            plot_df,
            "scatter",
            color=CrispyPlot.PAL_DTRACE[0],
            xlim=[ax_min, ax_max],
            ylim=[ax_min, ax_max],
            space=0,
            s=5,
            edgecolor="w",
            linewidth=0.0,
            marginal_kws={"hist": False, "rug": False},
            stat_func=None,
            alpha=0.1,
        )

        for n in zorder[::-1]:
            df = plot_df.query(f"{z} == '{n}'")
            g.x, g.y = df[x], df[y]

            g.plot_joint(
                sns.regplot,
                color=pal[n],
                fit_reg=False,
                scatter_kws={"s": 3, "alpha": 0.5, "linewidth": 0},
            )

            if plot_reg:
                g.plot_joint(
                    sns.kdeplot,
                    cmap=sns.light_palette(pal[n], as_cmap=True),
                    legend=False,
                    fill=False,
                    fill_lowest=False,
                    n_levels=9,
                    alpha=0.8,
                    lw=0.1,
                )

            g.plot_marginals(sns.kdeplot, color=pal[n], fill=True, legend=False)

        handles = [
            mpatches.Circle([0, 0], 0.25, facecolor=pal[s], label=s) for s in pal
        ]
        g.ax_joint.legend(
            loc="upper left",
            handles=handles,
            title="Protein\nattenuation",
            frameon=False,
        )

        g.ax_joint.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
        g.ax_joint.plot([ax_min, ax_max], [ax_min, ax_max], "k--", lw=0.3)

        plt.gcf().set_size_inches(figsize)

        return g


class GIPlot(CrispyPlot):
    MARKERS = ["o", "X", "v", "^"]

    @classmethod
    def gi_regression_no_marginals(
        cls,
        x_gene,
        y_gene,
        plot_df,
        alpha=1.0,
        hue=None,
        style=None,
        lowess=False,
        palette=None,
        plot_reg=True,
        figsize=(3, 3),
        plot_style_legend=True,
        plot_hue_legend=True,
        ax=None,
    ):
        pal = cls.PAL_DTRACE if palette is None else palette

        plot_df = plot_df.dropna(subset=[x_gene, y_gene])

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize, dpi=600)

        markers_handles = dict()

        # Joint
        for t, df in [("none", plot_df)] if hue is None else plot_df.groupby(hue):
            for i, (n, df_style) in enumerate(
                [("none", df)] if style is None else df.groupby(style)
            ):
                ax.scatter(
                    x=df_style[x_gene],
                    y=df_style[y_gene],
                    edgecolor="w",
                    lw=0.1,
                    s=3,
                    c=pal[2] if palette is None else pal[t],
                    alpha=alpha,
                    marker=cls.MARKERS[i],
                )
                markers_handles[n] = cls.MARKERS[i]

        if plot_reg:
            sns.regplot(
                x=x_gene,
                y=y_gene,
                data=plot_df,
                line_kws=dict(lw=1.0, color=cls.PAL_DTRACE[1]),
                marker="",
                lowess=lowess,
                truncate=True,
                ax=ax,
            )

        ax.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)

        cor, pval = spearmanr(plot_df[x_gene], plot_df[y_gene])
        rmse = sqrt(mean_squared_error(plot_df[x_gene], plot_df[y_gene]))
        annot_text = f"Spearman's R={cor:.2g}, p-value={pval:.1e}; RMSE={rmse:.2f}"
        ax.text(0.95, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="right")

        if plot_hue_legend and (palette is not None):
            hue_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    label=t,
                    mew=0,
                    markersize=3,
                    markerfacecolor=c,
                    lw=0,
                )
                for t, c in pal.items()
            ]
            hue_legend = ax.legend(
                handles=hue_handles,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                prop={"size": 3},
                frameon=False,
            )
            ax.add_artist(hue_legend)

        if plot_style_legend and (style is not None):
            style_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=m,
                    label=n,
                    mew=0,
                    markersize=3,
                    markerfacecolor=cls.PAL_DTRACE[2],
                    lw=0,
                )
                for n, m in markers_handles.items()
            ]
            ax.legend(
                handles=style_handles, loc="upper left", frameon=False, prop={"size": 3}
            )

        return ax

    @classmethod
    def gi_regression(
        cls,
        x_gene,
        y_gene,
        plot_df=None,
        size=None,
        size_range=None,
        size_inverse=False,
        size_legend_loc="best",
        size_legend_title=None,
        hue=None,
        style=None,
        lowess=False,
        palette=None,
        plot_reg=True,
        plot_annot=True,
        hexbin=False,
        color=None,
        label=None,
        a=0.75,
    ):
        pal = cls.PAL_DTRACE if palette is None else palette

        if plot_df is None:
            plot_df = pd.concat([x_gene, y_gene], axis=1)
            x_gene, y_gene = x_gene.name, y_gene.name

        plot_df = plot_df.dropna(subset=[x_gene, y_gene])

        if size is not None:
            plot_df = plot_df.dropna(subset=[size])

            feature_range = [1, 10] if size_range is None else size_range
            s_transform = MinMaxScaler(feature_range=feature_range)
            s_transform = s_transform.fit(
                (plot_df[[size]] * -1) if size_inverse else plot_df[[size]]
            )

        grid = sns.JointGrid(x=x_gene, y=y_gene, data=plot_df, space=0)

        # Joint
        if plot_reg:
            grid.plot_joint(
                sns.regplot,
                data=plot_df,
                line_kws=dict(lw=1.0, color=cls.PAL_DTRACE[1]),
                marker="",
                lowess=lowess,
                truncate=True,
            )

        hue_df = plot_df.groupby(hue) if hue is not None else [(None, plot_df)]
        for i, (h, h_df) in enumerate(hue_df):
            style_df = h_df.groupby(style) if style is not None else [(None, h_df)]

            for j, (s, s_df) in enumerate(style_df):
                if hexbin:
                    grid.ax_joint.hexbin(
                        s_df[x_gene],
                        s_df[y_gene],
                        cmap="Spectral_r",
                        gridsize=100,
                        mincnt=1,
                        bins="log",
                        lw=0,
                        alpha=1,
                    )

                else:
                    if size is None:
                        s = 3
                    elif size_inverse:
                        s = s_transform.transform(s_df[[size]] * -1)
                    else:
                        s = s_transform.transform(s_df[[size]])

                    sc = grid.ax_joint.scatter(
                        x=s_df[x_gene],
                        y=s_df[y_gene],
                        edgecolor="w",
                        lw=0.1,
                        s=s,
                        c=pal[2] if h is None else pal[h],
                        alpha=a,
                        marker=cls.MARKERS[0] if s is None else cls.MARKERS[j],
                        label=s,
                    )

                grid.x = s_df[x_gene].rename("")
                grid.y = s_df[y_gene].rename("")
                grid.plot_marginals(
                    sns.kdeplot,
                    # hist_kws=dict(linewidth=0, alpha=a),
                    cut=0,
                    legend=False,
                    fill=True,
                    color=pal[2] if h is None else pal[h],
                    label=h,
                )

        grid.ax_joint.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)

        if plot_annot:
            cor, pval = spearmanr(plot_df[x_gene], plot_df[y_gene])
            rmse = sqrt(mean_squared_error(plot_df[x_gene], plot_df[y_gene]))
            annot_text = f"Spearman's R={cor:.2g}, p-value={pval:.1e}; RMSE={rmse:.2f}"
            grid.ax_joint.text(
                0.95,
                0.05,
                annot_text,
                fontsize=4,
                transform=grid.ax_joint.transAxes,
                ha="right",
            )

        if style is not None:
            grid.ax_joint.legend(prop=dict(size=4), frameon=False, loc=2)

        if hue is not None:
            grid.ax_marg_y.legend(
                prop=dict(size=4),
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )

        if size is not None:

            def inverse_transform_func(x):
                arg_x = np.array(x).reshape(-1, 1)
                res = s_transform.inverse_transform(arg_x)[:, 0]
                return res

            handles, labels = sc.legend_elements(
                prop="sizes",
                num=8,
                func=inverse_transform_func,
            )
            grid.ax_joint.legend(
                handles,
                labels,
                title=size_legend_title,
                loc=size_legend_loc,
                frameon=False,
                prop={"size": 3},
            ).get_title().set_fontsize("3")

        plt.gcf().set_size_inches(2, 2)

        return grid

    @staticmethod
    def _marginal_boxplot(_, xs=None, ys=None, zs=None, vertical=False, **kws):
        if vertical:
            ax = sns.boxplot(x=zs, y=ys, orient="v", **kws)
        else:
            ax = sns.boxplot(x=xs, y=zs, orient="h", **kws)

        ax.set_ylabel("")
        ax.set_xlabel("")

    @classmethod
    def gi_regression_marginal(
        cls,
        x,
        y,
        z,
        plot_df,
        style=None,
        scatter_kws=None,
        line_kws=None,
        legend_title=None,
        discrete_pal=None,
        hue_order=None,
        annot_text=None,
        add_hline=False,
        add_vline=False,
        plot_reg=True,
        plot_annot=True,
        marginal_notch=False,
    ):
        # Defaults
        if scatter_kws is None:
            scatter_kws = dict(edgecolor="w", lw=0.3, s=8)

        if line_kws is None:
            line_kws = dict(lw=1.0, color=cls.PAL_DBGD[0])

        if discrete_pal is None:
            discrete_pal = cls.PAL_DTRACE

        if hue_order is None:
            hue_order = natsorted(set(plot_df[z]))

        #
        grid = sns.JointGrid(x=x, y=y, data=plot_df, space=0, ratio=8)

        grid.plot_marginals(
            cls._marginal_boxplot,
            palette=discrete_pal,
            data=plot_df,
            linewidth=0.3,
            fliersize=1,
            notch=marginal_notch,
            saturation=1.0,
            xs=x,
            ys=y,
            zs=z,
            showcaps=False,
            boxprops=cls.BOXPROPS,
            whiskerprops=cls.WHISKERPROPS,
            flierprops=cls.FLIERPROPS,
            medianprops=dict(linestyle="-", linewidth=1.0),
        )

        if plot_reg:
            sns.regplot(
                x=x,
                y=y,
                data=plot_df,
                color=discrete_pal[0],
                truncate=True,
                fit_reg=True,
                scatter=False,
                line_kws=line_kws,
                ax=grid.ax_joint,
            )

        for j, feature in enumerate(hue_order):
            dfs = plot_df[plot_df[z] == feature]
            dfs = (
                dfs.assign(style=1).groupby("style")
                if style is None
                else dfs.groupby(style)
            )

            for i, (mtype, df) in enumerate(dfs):
                sns.regplot(
                    x=x,
                    y=y,
                    data=df,
                    color=discrete_pal[feature],
                    fit_reg=False,
                    scatter_kws=scatter_kws,
                    label=mtype if j == 0 else None,
                    marker=cls.MARKERS[i],
                    ax=grid.ax_joint,
                )

        if style is not None:
            grid.ax_joint.legend(prop=dict(size=4), frameon=False, loc=2)

        # Annotation
        if plot_annot:
            if annot_text is None:
                df_corr = plot_df.dropna(subset=[x, y, z])
                cor, pval = pearsonr(df_corr[x], df_corr[y])
                annot_text = f"R={cor:.2g}, p={pval:.1e}"

            grid.ax_joint.text(
                0.95,
                0.05,
                annot_text,
                fontsize=4,
                transform=grid.ax_joint.transAxes,
                ha="right",
            )

        if add_hline:
            grid.ax_joint.axhline(0, ls="-", lw=0.3, c=cls.PAL_DBGD[0], alpha=0.2)

        if add_vline:
            grid.ax_joint.axvline(0, ls="-", lw=0.3, c=cls.PAL_DBGD[0], alpha=0.2)

        handles = [
            mpatches.Circle(
                (0.0, 0.0),
                0.25,
                facecolor=discrete_pal[t],
                label=f"{t} (N={(plot_df[z] == t).sum()})",
            )
            for t in hue_order
        ]

        grid.ax_marg_y.legend(
            handles=handles,
            title=z if legend_title is None else legend_title,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )
        grid.ax_marg_y.get_legend().get_title().set_fontsize("6")

        grid.ax_joint.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)

        plt.gcf().set_size_inches(1.5, 1.5)

        return grid

    @classmethod
    def gi_classification(
        cls,
        x_gene,
        y_gene,
        plot_df,
        hue=None,
        palette=None,
        orient="v",
        stripplot=True,
        notch=True,
        order=None,
        hue_order=None,
        plot_legend=True,
        legend_kws=None,
        ax=None,
    ):
        pal = cls.PAL_DTRACE if palette is None else palette

        if ax is None and orient == "v":
            figsize = (0.2 * len(set(plot_df[x_gene])), 2)

        elif ax is None and orient == "h":
            figsize = (2, 0.2 * len(set(plot_df[y_gene])))

        else:
            figsize = None

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize, dpi=600)

        if stripplot:
            sns.stripplot(
                x=x_gene,
                y=y_gene,
                order=order,
                hue=hue,
                hue_order=hue_order,
                data=plot_df,
                dodge=True,
                orient=orient,
                jitter=0.3,
                size=1.5,
                linewidth=0.1,
                alpha=0.5,
                edgecolor="white",
                palette=pal,
                ax=ax,
                zorder=0,
            )

        bp = sns.boxplot(
            x=x_gene,
            y=y_gene,
            order=order,
            hue=hue,
            hue_order=hue_order,
            data=plot_df,
            orient=orient,
            notch=notch,
            boxprops=dict(linewidth=0.3),
            whiskerprops=dict(linewidth=0.3),
            medianprops=cls.MEDIANPROPS,
            flierprops=cls.FLIERPROPS,
            palette=pal,
            showcaps=False,
            sym="" if stripplot else None,
            saturation=1.0,
            ax=ax,
        )

        for patch in bp.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.1))

        ax.grid(
            True,
            axis="y" if orient == "v" else "x",
            ls="-",
            lw=0.1,
            alpha=1.0,
            zorder=0,
        )

        if plot_legend and (hue is not None):
            hue_nfeatures = len(set(plot_df[hue]))
            handles, labels = bp.get_legend_handles_labels()
            ax.legend(
                handles[: (hue_nfeatures - 1)],
                labels[: (hue_nfeatures - 1)],
                frameon=False,
                **legend_kws,
            )

        elif ax.get_legend() is not None:
            ax.get_legend().remove()

        return ax

    @classmethod
    def gi_tissue_plot(
        cls,
        x,
        y,
        plot_df=None,
        hue="tissue",
        pal=CrispyPlot.PAL_TISSUE_2,
        plot_reg=True,
        annot=True,
        lowess=False,
        figsize=(2, 2),
    ):
        if plot_df is None:
            plot_df = pd.concat([x, y], axis=1)
            x, y = x.name, y.name

        plot_df = plot_df.dropna(subset=[x, y, hue])

        fig, ax = plt.subplots(figsize=figsize, dpi=600)

        for t, df in plot_df.groupby(hue):
            ax.scatter(
                df[x].values,
                df[y].values,
                c=pal[t],
                marker="o",
                linewidths=0,
                s=5,
                label=t,
                alpha=0.8,
            )

        if plot_reg:
            sns.regplot(
                x,
                y,
                data=plot_df,
                line_kws=dict(lw=1.0, color=cls.PAL_DTRACE[1]),
                marker="",
                lowess=lowess,
                truncate=True,
                ax=ax,
            )

        if annot:
            cor, pval = spearmanr(plot_df[x], plot_df[y])
            annot_text = f"Spearman's R={cor:.2g}, p-value={pval:.1e}"
            ax.text(
                0.95, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="right"
            )

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            prop={"size": 3},
            frameon=False,
            title=hue,
        ).get_title().set_fontsize("3")

        return ax

    @classmethod
    def gi_continuous_plot(
        cls,
        x,
        y,
        z,
        plot_df,
        cmap="Spectral_r",
        joint_alpha=0.8,
        mid_point_norm=True,
        mid_point=0,
        cbar_label=None,
        lowess=False,
        plot_reg=False,
        corr_annotation=True,
        ax=None,
    ):
        df = plot_df.dropna(subset=[x, y, z])

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=600)

        sc = ax.scatter(
            df[x],
            df[y],
            c=df[z],
            marker="o",
            edgecolor=None,
            s=5,
            linewidths=0,
            cmap=cmap,
            alpha=joint_alpha,
            norm=MidpointNormalize(midpoint=mid_point) if mid_point_norm else None,
        )

        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel(
            z if cbar_label is None else cbar_label, rotation=270, va="bottom"
        )

        if plot_reg:
            sns.regplot(
                x,
                y,
                data=plot_df,
                line_kws=dict(lw=1.0, color=cls.PAL_DTRACE[1]),
                scatter=False,
                lowess=lowess,
                truncate=True,
                ax=ax,
            )

        if corr_annotation:
            cor, pval = spearmanr(df[x], df[y])
            annot_text = f"R={cor:.2g}, p={pval:.1e}"
            ax.text(
                0.95, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="right"
            )

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

        return ax
