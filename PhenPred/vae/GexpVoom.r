library(limma)

setwd("/Users/emanuel/Projects/PhenPred/data/clines/depmap23Q2/")

# Read csv as OmicsExpressionProteinCodingGenesTPMLogp1
data <- read.csv("OmicsExpressionGenesExpectedCountProfile.csv", row.names=1)
data <- t(data)
cat('Total genes and cell lines:', dim(data), '\n')

# There are no NA in these datasets
data[ which(is.na(data)) ] = 0
cat('Total genes and cell lines:', dim(data), '\n')

cat('Remove samples exeding genes with 0\n')
zero.count = apply(data, 2, function(x) sum(x==0) ) / nrow(data)
summary(zero.count)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.1251  0.2148  0.2330  0.2376  0.2527  0.6088 

data = data[ , ! colnames(data) %in% names(zero.count)[ zero.count > .4 ] ]
cat('Total genes and cell lines:', dim(data), '\n')
# Total genes and cell lines: 21823 1985
