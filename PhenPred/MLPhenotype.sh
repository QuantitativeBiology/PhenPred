#!/bin/bash
python3 PhenPred/MLPhenotype.py config/config_rf.yml methylation
python3 PhenPred/MLPhenotype.py config/config_rf.yml genomics
python3 PhenPred/MLPhenotype.py config/config_rf.yml transcriptomics
python3 PhenPred/MLPhenotype.py config/config_rf.yml proteomics
python3 PhenPred/MLPhenotype.py config/config_rf.yml metabolomics
# python3 PhenPred/MLPhenotype.py config/config_rf.yml tissue
python3 PhenPred/MLPhenotypePlot.py