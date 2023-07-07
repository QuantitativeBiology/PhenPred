library(MOFA2)

## ---------- Model, Data e Info Cell Lines
model <- load_model("data/clines/mofa/MOFA_depmap23Q2_Factors50.hdf5")
data <- get_data(model, as.data.frame = T)
head(data, n=3)

## ---------- Data imputation and prediction
data_imputed <- impute(model, views = "all", factors = "all")
data_predicted <- predict(model, views = "all", factors = "all")

## ---------- Write imputed / predicted data
write.csv(data_imputed@imputed_data$crisprcas9$groupA, "data/clines/mofa/MOFA_depmap23Q2_Factors50_imputed_crisprcas9.csv", row.names=TRUE)
write.csv(data_imputed@imputed_data$drugresponse$groupA, "data/clines/mofa/MOFA_depmap23Q2_Factors50_imputed_drugresponse.csv", row.names=TRUE)
write.csv(data_imputed@imputed_data$metabolomics$groupA, "data/clines/mofa/MOFA_depmap23Q2_Factors50_imputed_metabolomics.csv", row.names=TRUE)
write.csv(data_imputed@imputed_data$methylation$groupA, "data/clines/mofa/MOFA_depmap23Q2_Factors50_imputed_methylation.csv", row.names=TRUE)
write.csv(data_imputed@imputed_data$proteomics$groupA, "data/clines/mofa/MOFA_depmap23Q2_Factors50_imputed_proteomics.csv", row.names=TRUE)
write.csv(data_imputed@imputed_data$transcriptomics$groupA, "data/clines/mofa/MOFA_depmap23Q2_Factors50_imputed_transcriptomics.csv", row.names=TRUE)
write.csv(data_imputed@imputed_data$copynumber$groupA, "data/clines/mofa/MOFA_depmap23Q2_Factors50_imputed_copynumber.csv", row.names=TRUE)

write.csv(data_predicted$crisprcas9$groupA, "data/clines/mofa/MOFA_depmap23Q2_Factors50_predicted_crisprcas9.csv", row.names=TRUE)
write.csv(data_predicted$drugresponse$groupA, "data/clines/mofa/MOFA_depmap23Q2_Factors50_predicted_drugresponse.csv", row.names=TRUE)
write.csv(data_predicted$metabolomics$groupA, "data/clines/mofa/MOFA_depmap23Q2_Factors50_predicted_metabolomics.csv", row.names=TRUE)
write.csv(data_predicted$methylation$groupA, "data/clines/mofa/MOFA_depmap23Q2_Factors50_predicted_methylation.csv", row.names=TRUE)
write.csv(data_predicted$proteomics$groupA, "data/clines/mofa/MOFA_depmap23Q2_Factors50_predicted_proteomics.csv", row.names=TRUE)
write.csv(data_predicted$transcriptomics$groupA, "data/clines/mofa/MOFA_depmap23Q2_Factors50_predicted_transcriptomics.csv", row.names=TRUE)
write.csv(data_predicted$copynumber$groupA, "data/clines/mofa/MOFA_depmap23Q2_Factors50_predicted_copynumber.csv", row.names=TRUE)