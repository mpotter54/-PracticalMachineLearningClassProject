### load libraries
library(knitr)
library(caret)
library(Hmisc)
library(gridExtra)
library(ggplot2)
library(caret)
library(rattle)
library(RGtk2)
library(RGtk2Extras)
library(ElemStatLearn)
library(ISLR)
library(gbm)
library(quantmod)
library(xts)
library(forecast)
library(pander)
library(dplyr)
### pander options
panderOptions('table.style', 'multiline')
panderOptions("table.split.table", 105)
### load data
mainDir <- "C:/PracticalMachineLearningClassProject"
setwd(mainDir)
fileTraining <- "pml-training.csv"
training <- read.csv(file=fileTraining, head=TRUE, sep=",",
                     na.strings=c('#DIV/0', '', 'NA'), 
                     stringsAsFactors = FALSE)
fileTesting <- "pml-testing.csv"
testing <- read.csv(file=fileTesting, head=TRUE, sep=",",
                    na.strings=c('#DIV/0', '', 'NA'), 
                    stringsAsFactors = FALSE)
### clean data
drops <- colnames(training)[colSums(is.na(training)) > 19000]
drops <- c("X", "user_name", "cvtd_timestamp", "raw_timestamp_part_1", "raw_timestamp_part_2", 
           "new_window", "num_window", drops)
train2 <- training[, !(names(training) %in% drops)]
tsFN <- testing[, !(names(training) %in% drops)]
inTrain <- createDataPartition(train2$classe,p=0.7,list=FALSE)
trFN <- train2[inTrain,]
vaFN <- train2[-inTrain,]
trFN$classe <- as.factor(trFN$classe)
vaFN$classe <- as.factor(vaFN$classe)
### experiment with different models
n <- 1200
sample <- sample_n(trFN, n)
tc <- trainControl(method="cv", number=10)
mc <- "Accuracy"
### LDA
set.seed(n)
model.lda <- train(classe~., data=sample, method="lda", metric=mc, trControl=tc)
### CART
set.seed(n)
model.cart <- train(classe~., data=sample, method="rpart", metric=mc, trControl=tc)
### KNN
set.seed(n)
model.knn <- train(classe~., data=sample, method="knn", metric=mc, trControl=tc)
### SVM
set.seed(n)
model.svm <- train(classe~., data=sample, method="svmRadial", metric=mc, trControl=tc)
### Random Forest
set.seed(n)
model.rf <- train(classe~., data=sample, method="rf", metric=mc, trControl=tc)
### Naive Bayes
set.seed(n)
model.nb <- train(classe~., data=sample, method="nb", metric=mc, trControl=tc)
### Gradient Boosting
set.seed(n)
model.gbm <- train(classe~., data=sample, method="gbm", metric=mc, trControl=tc)
### Select the best model
results <- resamples(list(lda=model.lda, 
                          cart=model.cart, 
                          knn=model.knn, 
                          svm=model.svm, 
                          rf=model.rf, 
                          nb=model.nb, 
                          gbm=model.gbm))
summary(results)
dotplot(results)
### Random Forest
n <- 2400
set.seed(n)
model2.rf <- train(classe~., data=trFN, method="rf", metric=mc, trControl=tc)
### Gradient Boosting
set.seed(n)
model2.gbm <- train(classe~., data=trFN, method="gbm", metric=mc, trControl=tc)
### Select the best model
results2 <- resamples(list(rf=model2.rf, gbm=model2.gbm))
summary(results2)
### Predict with random forest against validation set
pred.rf <- predict(model2.rf, newdata = vaFN)
confusionMatrix(pred.rf, vaFN$classe)
### Predict against test set
pred.rft <- predict(model2.rf, newdata = tsFN)
print(pred.rft)
### Columns Removed
drops
### Columns Classes
str(trFN)
### classe levels
prTR <- prop.table(table(trFN$classe)) * 100
trLevels <- cbind(freq=table(trFN$classe), percentage=prTR)
pander(trLevels)
### Box Plot Belt
belt <- colnames(trFN)[grep("_belt",colnames(trFN))]
featurePlot(x=trFN[,belt], y=trFN[,53], plot="box", layout=c(4,4), auto.key = list(columns = 5))
### Density Plot Belt
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=trFN[,belt], y=trFN[,53], plot="density", scales=scales, layout=c(4,4), auto.key = list(columns = 5))
### Box Plot Arm
arm <- colnames(trFN)[grep("_arm",colnames(trFN))]
featurePlot(x=trFN[,arm], y=trFN[,53], plot="box", layout=c(4,4), auto.key = list(columns = 5))
### Density Plot Arm
featurePlot(x=trFN[,arm], y=trFN[,53], plot="density", scales=scales, layout=c(4,4), auto.key = list(columns = 5))
### Box Plot Dumbbell
dumbbell <- colnames(trFN)[grep("_dumbbell",colnames(trFN))]
featurePlot(x=trFN[,dumbbell], y=trFN[,53], plot="box", layout=c(4,4), auto.key = list(columns = 5))
### Density Plot Dumbbell
featurePlot(x=trFN[,dumbbell], y=trFN[,53], plot="density", scales=scales, layout=c(4,4), auto.key = list(columns = 5))
### Box Plot Forearm
forearm <- colnames(trFN)[grep("_forearm",colnames(trFN))]
featurePlot(x=trFN[,forearm], y=trFN[,53], plot="box", layout=c(4,4), auto.key = list(columns = 5))
### Density Plot Forearm
featurePlot(x=trFN[,forearm], y=trFN[,53], plot="density", scales=scales, layout=c(4,4), auto.key = list(columns = 5))
