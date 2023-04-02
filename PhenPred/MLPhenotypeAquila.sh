#!/bin/bash
python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml methylation tissue
python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml genomics tissue
python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml transcriptomics tissue
python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml proteomics tissue
python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml metabolomics tissue

python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml methylation metabolomics
python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml genomics metabolomics
python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml transcriptomics metabolomics
python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml proteomics metabolomics
python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml tissue metabolomics

python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml methylation None
python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml genomics None
python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml transcriptomics None
python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml proteomics None
python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml metabolomics None
python3 PhenPred/MLPhenotype.py config/config_rf_drug_aquila.yml tissue None