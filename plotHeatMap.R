## Import (packages)
library(ggplot2)
library(tidyr)
library(dplyr)

if (!require("gplots")) {
   install.packages("gplots", dependencies = TRUE)
   library(gplots)
   }
if (!require("RColorBrewer")) {
   install.packages("RColorBrewer", dependencies = TRUE)
   library(RColorBrewer)
   }

## Read in data
engData <- read.csv("english.csv", header=T, row.names=1)
finData <- read.csv("finnish.csv", header=T, row.names=1)
turData <- read.csv("turkish.csv", header=T, row.names=1)

engMatrixData <- as.matrix(engData)
finMatrixData <- as.matrix(finData)
turMatrixData <- as.matrix(turData)

engMatrixData
finMatrixData
turMatrixData

# creates a own color palette from red to green
my_palette <- colorRampPalette(c("red", "yellow", "green"))(n = 299)

eng_col_breaks = c(seq(0,0.083,length=100),  # for red
  seq(0.084,0.14,length=100),           # for yellow
  seq(0.141,1,length=100))             # for green

finTur_col_breaks = c(seq(0,0.062,length=100),  # for red
  seq(0.063,0.14,length=100),           # for yellow
  seq(0.141,1,length=100))             # for green

png("english-heatmap.png", height=900,width=900)
heatmap.2(engMatrixData, Rowv = NA, Colv = NA, main = "English Vowel PMI",
	dendrogram="none",
  trace=c("none"),
  density.info=c("none"),
  lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(1.5, 4, 2 ),
	col=my_palette,       # use on color palette defined earlier
  breaks=eng_col_breaks    # enable color transition at specified limits
  )
dev.off()

png("finnish-heatmap.png", height=900,width=900)
heatmap.2(finMatrixData, Rowv = NA, Colv = NA, main = "Finnish Vowel PMI",
  dendrogram="none",
  trace=c("none"),
  density.info=c("none"),
  lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(1.5, 4, 2 ),
	col=my_palette,       # use on color palette defined earlier
  breaks=finTur_col_breaks    # enable color transition at specified limits
  )
dev.off()

png("turkish-heatmap.png", height=900,width=900)
heatmap.2(turMatrixData, Rowv = NA, Colv = NA, main = "Turkish Vowel PMI",
  dendrogram="none",
  trace=c("none"),
  density.info=c("none"),
  lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(1.5, 4, 2 ),
	col=my_palette,       # use on color palette defined earlier
  breaks=finTur_col_breaks    # enable color transition at specified limits
  )
dev.off()