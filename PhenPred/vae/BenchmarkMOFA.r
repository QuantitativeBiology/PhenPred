library(MOFA2)

## ---------- Number of factors
n_factors <- 100

## ---------- Model, Data e Info Cell Lines

# concat names path with n_factors
model <- load_model(paste0("data/clines/mofa/MOFA_depmap23Q2_Factors", n_factors, ".hdf5"))
data <- get_data(model, as.data.frame = T)
head(data, n=3)

## ---------- Data imputation and prediction
data_imputed <- impute(model, views = "all", factors = "all")
data_predicted <- predict(model, views = "all", factors = "all")

## ---------- Write imputed / predicted data
dpath_prefix <- "data/clines/mofa/MOFA_depmap23Q2_Factors"

write.csv(data_imputed@imputed_data$crisprcas9$groupA, paste0(dpath_prefix, n_factors, "_imputed_crisprcas9.csv"), row.names=TRUE)
write.csv(data_imputed@imputed_data$drugresponse$groupA, paste0(dpath_prefix, n_factors, "_imputed_drugresponse.csv"), row.names=TRUE)
write.csv(data_imputed@imputed_data$metabolomics$groupA, paste0(dpath_prefix, n_factors, "_imputed_metabolomics.csv"), row.names=TRUE)
write.csv(data_imputed@imputed_data$methylation$groupA, paste0(dpath_prefix, n_factors, "_imputed_methylation.csv"), row.names=TRUE)
write.csv(data_imputed@imputed_data$proteomics$groupA, paste0(dpath_prefix, n_factors, "_imputed_proteomics.csv"), row.names=TRUE)
write.csv(data_imputed@imputed_data$transcriptomics$groupA, paste0(dpath_prefix, n_factors, "_imputed_transcriptomics.csv"), row.names=TRUE)
write.csv(data_imputed@imputed_data$copynumber$groupA, paste0(dpath_prefix, n_factors, "_imputed_copynumber.csv"), row.names=TRUE)

write.csv(data_predicted$crisprcas9$groupA, paste0(dpath_prefix, n_factors, "_predicted_crisprcas9.csv"), row.names=TRUE)
write.csv(data_predicted$drugresponse$groupA, paste0(dpath_prefix, n_factors, "_predicted_drugresponse.csv"), row.names=TRUE)
write.csv(data_predicted$metabolomics$groupA, paste0(dpath_prefix, n_factors, "_predicted_metabolomics.csv"), row.names=TRUE)
write.csv(data_predicted$methylation$groupA, paste0(dpath_prefix, n_factors, "_predicted_methylation.csv"), row.names=TRUE)
write.csv(data_predicted$proteomics$groupA, paste0(dpath_prefix, n_factors, "_predicted_proteomics.csv"), row.names=TRUE)
write.csv(data_predicted$transcriptomics$groupA, paste0(dpath_prefix, n_factors, "_predicted_transcriptomics.csv"), row.names=TRUE)
write.csv(data_predicted$copynumber$groupA, paste0(dpath_prefix, n_factors, "_predicted_copynumber.csv"), row.names=TRUE)