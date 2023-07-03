library(limma)
library(edgeR)

setwd("/Users/emanuel/Projects/PhenPred/data/clines/depmap23Q2/")

# Read samplesheet from OmicsProfiles.csv set first column as row names
samplesheet <- read.csv("OmicsProfiles.csv", header=TRUE, stringsAsFactors=FALSE, row.names=1)

# read protein coding genes list from ProteinCodingGenes.txt
genes <- read.table("ProteinCodingGenes.txt", header=FALSE, stringsAsFactors=FALSE)
genes <- genes[,1]

# Read csv as OmicsExpressionProteinCodingGenesTPMLogp1
data <- read.csv("OmicsExpressionGenesExpectedCountProfile.csv", row.names=1)
data <- t(data)
cat('Total genes and cell lines:', dim(data), '\n')

data = data[ ! grepl('^ENSG0', rownames(data)), ]
rownames(data) = gsub('\\..*', '', rownames(data))
data = data[ rownames(data) %in% genes, ]

cat('Duplicate row names:', length( rownames(data) ) - length( unique( rownames(data) ) ), '\n')
# aggregate duplicated rows preserving the row_names
data = aggregate(data, by=list(rownames(data)), FUN=sum)
rownames(data) = data[,1]
data = data[ , -1]
cat('Total genes and cell lines:', dim(data), '\n')

cat('Remove samples exeding genes with 0\n')
zero.count = apply(data, 2, function(x) sum(x==0) ) / nrow(data)
summary(zero.count)

data = data[ , ! colnames(data) %in% names(zero.count)[ zero.count > .4 ] ]
cat('Total genes and cell lines:', dim(data), '\n')

message('Remove non-expressed genes with average CPM <= 0 ') # "The limma-voom method assumes that rows with zero or very low counts have been removed"
keep = aveLogCPM(data) > 0
data = data[ keep, ]
cat('Total genes and cell lines:', dim(data), '\n')

cat('Normalization\n')
# Although it is also possible to give a matrix of counts directly to voom without TMM normalization, limma package reommends it
data = DGEList(counts=data,genes=rownames(data))
data = calcNormFactors(data, method = 'TMM')

cat('Voom transformation\n')
data_voom = voom(data, plot = T)$E # $E extract the "data corrected for variance"

# map column names to sample names using samplesheet
colnames(data_voom) = samplesheet[ colnames(data_voom), 'ModelID' ]

# export to csv, round to 5 decimals, not use quotes
write.csv(round(data_voom, 5), file="OmicsExpressionGenesExpectedCountProfileVoom.csv", quote=FALSE)


