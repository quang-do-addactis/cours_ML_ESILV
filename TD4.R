library(data.table)
library(xgboost)
library(randomForest)
library(caret)


digit <- fread("C:/Users/xuan-quang.do/Documents/ESILV/Machine Learning/github_share/digit_recognizer.csv")

names(digit)
sqrt(784) #28*28

set.seed(1)
digit = copy(digit[sample(.N, 0.1 * nrow(digit))])
digit[, .N / nrow(digit) * 100, label][order(label)]

hist(digit$label, breaks = seq(-0.5, 9.5, by = 1))




#Visualization 
plotdigit<- function(datarow){
  # function will produce an image of the data given
  # the function takes away the first value because it is the target
  
  rotate <- function(x) t(apply(x, 2, rev))
  
  title<- datarow[, 1] # get actual value from training data
  datarow <- datarow[, -1] # remove the target column
  datarow<- as.numeric(datarow) # convert to numeric
  
  x<- rep(0:27)/27 
  y<- rep(0:27)/27
  z<- matrix(unlist(datarow), ncol=28, byrow=T)
  z<- rotate(z)
  
  
  main_ti = paste0("Actual Value:", title)
  image(x,y,z, main=main_ti, col=gray.colors(255, start=1, end=0), asp=1,
        xlim=c(0,1), ylim=c(-0.1,1.1), useRaster = T, axes=F, xlab='', ylab='')
}


par(mfrow=c(3, 4))
set.seed(1)
rows = sample(1:nrow(digit), size = 12)
for (i in rows){
  plotdigit(digit[i, ])
}



set.seed(1)
datum = digit[label == 1]
rows = sample(1:nrow(datum), size = 12)
for (i in rows){
  plotdigit(datum[i, ])
}