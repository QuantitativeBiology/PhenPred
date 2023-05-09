from turtle import color
import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from math import sqrt
from datetime import datetime
from scipy.special import stdtr
from scipy.stats import chi2
from sklearn.metrics import mean_squared_error
from PhenPred.vae import data_folder, plot_folder
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler


class GenomicsBenchmark:
    def __init__(self, timestamp):
        self.timestamp = timestamp

        # Original dataset
        self.genomics = pd.read_csv(f"{data_folder}/genomics.csv", index_col=0).T

        # Original drug response values
        self.drespo = pd.read_csv(f"{data_folder}/drugresponse.csv", index_col=0).T

        # Fully generated drug response autoencoder dataset
        self.drespo_vae = pd.read_csv(
            f"{plot_folder}/files/{timestamp}_imputed_drugresponse.csv.gz", index_col=0
        )

        # Sample sheet
        self.ss = pd.read_csv(f"{data_folder}/cmp_model_list_20230307.csv", index_col=0)

        # Samples overlap
        self.samples = list(
            set(self.genomics.index)
            .intersection(set(self.drespo.index))
            .intersection(set(self.drespo_vae.index))
            .intersection(set(self.ss.index))
        )

        # Covariates
        self.covs = pd.concat(
            [
                self.ss["growth_properties"].str.get_dummies(),
                self.ss["tissue"].str.get_dummies()["Haematopoietic and Lymphoid"],
            ],
            axis=1,
        ).reindex(index=self.samples)

        # Impute missing values
        self.drespo_imputed = self.drespo.reindex(
            index=self.samples, columns=self.drespo_vae.columns, copy=True
        )
        self.drespo_imputed = self.drespo_imputed.fillna(
            self.drespo_vae.reindex(
                index=self.samples, columns=self.drespo_imputed.columns
            )
        )

        # Subset original
        self.drespo = self.drespo.reindex(
            index=self.samples, columns=self.drespo_vae.columns
        )

        # Filter low occurance genomic features
        self.genomics = self.genomics.loc[:, self.genomics.count() > 3]

    def run(self):
        lm_res = self.associations()
        lm_res.to_csv(f"{plot_folder}/genomics/{self.timestamp}_lm_res.csv")

    def associations(self):
        lm_drug_orig = pd.concat(
            [
                LModel(
                    Y=self.drespo.loc[self.samples, [d]].dropna(),
                    X=self.genomics.loc[self.samples],
                    M=self.covs.loc[self.samples],
                ).fit_matrix()
                for d in self.drespo_imputed.columns
            ]
        )
        lm_drug_orig = LModel.multipletests(lm_drug_orig).sort_values("fdr")
        lm_drug_orig = lm_drug_orig.set_index(["y_id", "x_id"])

        lm_drug_vae = LModel(
            Y=self.drespo_imputed.loc[self.samples],
            X=self.genomics.loc[self.samples],
            M=self.covs.loc[self.samples],
        ).fit_matrix()
        lm_drug_vae = LModel.multipletests(lm_drug_vae).sort_values("fdr")
        lm_drug_vae = lm_drug_vae.set_index(["y_id", "x_id"])

        lm_drug = pd.concat(
            [
                lm_drug_orig.add_suffix("_orig"),
                lm_drug_vae.add_suffix("_vae"),
            ],
            axis=1,
        ).dropna()

        return lm_drug

    def plot_associations(self, lm_res):
        plot_df = lm_res.query("fdr_orig < 0.05 | fdr_var < 0.05").copy()
        plot_df = plot_df.sort_values("pval_orig").reset_index()

        _, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=600)

        sns.scatterplot(
            x=plot_df.index,
            y=-np.log10(plot_df["pval_orig"]),
            color="#E1E1E1",
            lw=0,
            s=1,
            zorder=1,
            rasterized=True,
            ax=ax,
        )

        ax.set(
            title=f"Pharmacogenomics associations (N={plot_df.shape[0]:,})",
            xlabel="Ranked drug ~ genomic feature associations",
            ylabel="Log-ratio p-value (log10)",
        )

        plt.savefig(
            f"{plot_folder}/genomics/{self.timestamp}_lm_assoc_pval_scatter.pdf",
            bbox_inches="tight",
        )
        plt.close()

        d_drespo_original = self.drespo[["1373;Dabrafenib;GDSC2"]]
        d_drespo_original_scl = pd.DataFrame(
            StandardScaler().fit_transform(d_drespo_original),
            index=d_drespo_original.index,
        )

        d_drespo_vae = self.drespo_vae[["1373;Dabrafenib;GDSC2"]]
        d_drespo_vae_scl = pd.DataFrame(
            StandardScaler().fit_transform(d_drespo_vae), index=d_drespo_vae.index
        )

        plot_df = pd.concat(
            [
                d_drespo_original_scl[0].rename("original"),
                d_drespo_vae_scl[0].rename("vae"),
                self.genomics["BRAF_mut"],
            ],
            axis=1,
        ).dropna()

        sns.scatterplot(
            x=plot_df["original"], y=plot_df["vae"], hue=plot_df["BRAF_mut"]
        )
        plt.show()


class LModel:
    def __init__(
        self,
        Y,
        X,
        M,
        M2=None,
        fit_intercept=True,
        copy_X=True,
        n_jobs=-1,
        verbose=0,
    ):
        self.samples = list(
            set.intersection(
                set(Y.index),
                set(X.index),
                set(M.index),
                set(Y.index) if M2 is None else set(M2.index),
            )
        )

        self.X = X.loc[self.samples]
        self.X = self.X.loc[:, self.X.count() > (M.shape[1] + (1 if M2 is None else 2))]
        self.X_ma = np.ma.masked_invalid(self.X.values)

        self.Y = Y.loc[self.samples]
        self.Y = self.Y.loc[:, self.Y.std() > 0]

        self.M = M.loc[self.samples]

        self.M2 = M2.loc[self.samples, self.X.columns] if M2 is not None else M2

        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs

        self.verbose = verbose

    def model_regressor(self):
        regressor = LinearRegression(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            n_jobs=self.n_jobs,
        )
        return regressor

    @staticmethod
    def loglike(y_true, y_pred):
        nobs = len(y_true)
        nobs2 = nobs / 2.0

        ssr = np.power(y_true - y_pred, 2).sum()

        llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / nobs) - nobs2

        return llf

    @staticmethod
    def multipletests_per(
        associations, method="fdr_bh", field="pval", fdr_field="fdr", index_cols=None
    ):
        index_cols = ["y_id"] if index_cols is None else index_cols

        d_unique = {tuple(i) for i in associations[index_cols].values}

        df = associations.set_index(index_cols)

        df = pd.concat(
            [
                df.loc[i]
                .assign(fdr=multipletests(df.loc[i, field], method=method)[1])
                .rename(columns={"fdr": fdr_field})
                for i in d_unique
            ]
        ).reset_index()

        return df

    def fit_matrix(self):
        lms = []

        for x_idx, x_var in enumerate(self.X):
            if self.verbose > 0:
                print(f"LM={x_var} ({x_idx})")

            # Mask NaNs
            x_ma = np.ma.mask_rowcols(self.X_ma[:, [x_idx]], axis=0)

            # Build matrices
            x = self.X.iloc[~x_ma.mask.any(axis=1), [x_idx]]
            y = self.Y.iloc[~x_ma.mask.any(axis=1), :]

            # Covariate matrix (remove invariable features and add noise)
            m = self.M.iloc[~x_ma.mask.any(axis=1), :]
            if self.M2 is not None:
                m2 = self.M2.iloc[~x_ma.mask.any(axis=1), [x_idx]]
                m = pd.concat([m2, m], axis=1)
            m = m.loc[:, m.std() > 0]
            m += np.random.normal(0, 1e-4, m.shape)

            # Fit covariate model
            lm_small = self.model_regressor().fit(m, y)
            lm_small_ll = self.loglike(y, lm_small.predict(m))

            # Fit full model: covariates + feature
            lm_full_x = np.concatenate([m, x], axis=1)
            lm_full = self.model_regressor().fit(lm_full_x, y)
            lm_full_ll = self.loglike(y, lm_full.predict(lm_full_x))

            # Log-ratio test
            lr = 2 * (lm_full_ll - lm_small_ll)
            lr_pval = chi2(1).sf(lr)

            # Assemble + append results
            res = pd.DataFrame(
                dict(
                    y_id=y.columns,
                    x_id=x_var,
                    n=y.attrs["nan_mask"].loc[y.columns, x.index].sum(1)
                    if "nan_mask" in y.attrs
                    else len(x),
                    beta=lm_full.coef_[:, -1],
                    lr=lr.values,
                    covs=m.shape[1],
                    pval=lr_pval,
                    fdr=multipletests(lr_pval, method="fdr_bh")[1],
                )
            )

            lms.append(res)

        lms = pd.concat(lms, ignore_index=True).sort_values("pval")

        return lms

    @staticmethod
    def lm_residuals(y, x, fit_intercept=True, add_intercept=False):
        # Prepare input matrices
        ys = y.dropna()

        xs = x.loc[ys.index].dropna()
        xs = xs.loc[:, xs.std() > 0]

        ys = ys.loc[xs.index]

        if ys.shape[0] <= xs.shape[1]:
            return None

        # Linear regression models
        lm = LinearRegression(fit_intercept=fit_intercept).fit(xs, ys)

        # Calculate residuals
        residuals = ys - lm.predict(xs) - lm.intercept_

        # Add intercept
        if add_intercept:
            residuals += lm.intercept_

        return residuals

    @staticmethod
    def multipletests(
        parsed_results, pval_method="fdr_bh", field="pval", idx_cols=None
    ):
        idx_cols = ["y_id"] if idx_cols is None else idx_cols

        parsed_results_adj = []

        for idx, df in parsed_results.groupby(idx_cols):
            df = df.assign(fdr=multipletests(df[field], method=pval_method)[1])
            parsed_results_adj.append(df)

        parsed_results_adj = pd.concat(parsed_results_adj, ignore_index=True)

        return parsed_results_adj
