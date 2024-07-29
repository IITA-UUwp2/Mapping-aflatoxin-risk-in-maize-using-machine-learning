### A workflow for predicting categorical aflatoxin risk classes in South and East Africa
### Various ensemble and non-ensemble machine learning methods are used
### VSURF was used to eliminate redundant variables
### The model uses both space-time (ST-CV) and random (CV)
### Created by Stella M Gachoki
### June/July 2024

##### ********************* INITIALIZE ******************** ####
#### Create a function to load all the libraries required
load_libraries <- function() {
  libs <- c("caret", "randomForest", "sf", "CAST", "gbm", "xgboost", "tidyquant",
            "VSURF", "glmnet", "reshape2", "ggplot2", "dplyr", "tidyr", 
            "corrplot", "igraph", "themis", "MLmetrics", "adabag", "C50", 
            "tidyquant", "viridis", "gridExtra", "stringr", "raster", 
            "terra", "stars", "rasterVis", "sp", "virtualspecies","ggdist","BiodiversityR", "pdp")
  sapply(libs, require, character.only = TRUE)
}

load_libraries()

######## Exploratory data analysis #####
# Load the temporally matched database
d.temp <- read.csv("cleaned_afla_FINALmonthly24_nonzero.csv")
names(d.temp)
d.temp$PrecClus <- as.factor(d.temp$PrecClus)
d.temp$Year2 <- as.factor(d.temp$Year2)
d.temp$class <- as.factor(d.temp$class)

#### Plot boxplots showing the aflatoxin prevalence
png(file = "Aflatoxin_prevalence.png", width = 7000, height =5000, units = "px", res = 700, type = "cairo")
ggplot(d.temp, aes(x=factor(Year2), y=Aflatoxin, fill=Country)) + 
  geom_boxplot() + ylim(0,300)+ facet_grid(Country ~ .) + 
  labs(title="",x="", y="Aflatoxin prevalence") +
  theme_ggdist()  + theme(legend.position = "none",text = element_text(size = 15, color = "black",face="bold"),
                          axis.text.y = element_text(size = 14, face="bold",color="black"),
                          axis.text.x = element_text(angle = 90, hjust = 1, size=14, color="black",face="bold"),
                          plot.title = element_text(hjust = 0.5,size=15, color="black",face="bold"))
dev.off()


#### Plot the distribution of the various risk classes per year and country
png(file = "Distribution_of_sample_locations.png", width = 15000, height =12000, units = "px", res = 900, type = "cairo")
ggplot(d.temp, aes(x=class, fill=class)) + 
  geom_bar() + facet_grid(Country ~ Year2) + labs(title="",x = "",y = "Number of locations") +
  theme_bw()  + theme(legend.position = "top",text = element_text(size = 20, color = "black",face="bold"),
                      axis.text.y = element_text(size = 20, face="bold",color="black"),
                      axis.text.x = element_text(angle = 90, hjust = 1, size=20, color="black",face="bold"),
                      plot.title = element_text(hjust = 0.5,size=20, color="black",face="bold"))
dev.off()

##### Data partition using the various response variables as stratas ####
set.seed(123)
trainIndex.temp <- createDataPartition(d.temp$class, p = .8, list = FALSE)
trainData.temp <- d.temp[trainIndex.temp,]
testData.temp <- d.temp[-trainIndex.temp,]

##### Extraction of predictors (x) and response variables (y) for each set
# Split data into predictors (X) and response variable (y)
x.temp <- trainData.temp[, c(10:111)]
y.temp <- trainData.temp$class

##### Feature elimination using VSURF  #####
### Retain the prediction step variables
set.seed(1234)
##Temporal matched
vsurf.temp <- VSURF(x.temp,trainData.temp$class,data=trainData.temp ,mtry=15,ntree.thres = 100,
                    nfor.thres = 10,ntree.interp = 100,nfor.interp = 10)
vsurf.temp$varselect.interp

thres.var <- data.frame(x.temp[,c(50, 42 ,28, 58, 12, 85, 36 ,72 ,10,  8, 84)])
names(thres.var)

# Define the formula for model fitting
fm.temp <- class ~ stempJun+stempMay+mintempMar+pdsiSep+mintempApr+precOct+mintempJan+stempJul+stempJan+mintempJul+pdsiJan+mintempOct

##### MODEL TRAINING with SPACE-TIME CV $$$$$$$$$$$$ #####
folds.temp <- CreateSpacetimeFolds(trainData.temp, spacevar = "PrecClus",timevar="Year2", k = 3, class = "class")

ctrl.temp <- trainControl(method = "repeatedcv", number = 5, repeats=3, savePredictions = "all", verboseIter = TRUE,index = folds.temp$index,
                          classProbs = TRUE, summaryFunction = multiClassSummary, selectionFunction = "best", allowParallel = TRUE)

###### Temporally matched with space-time cross validation
models.temp <- list(
  ranger.temp = caret::train(fm.temp, data = trainData.temp, method = "ranger", trControl = ctrl.temp,importance="permutation",
                             tuneGrid=expand.grid(.mtry = 5, .splitrule = "extratrees", .min.node.size = 1),
                             num.trees = 300,max.depth=6,min.bucket=1),
  adaboost.temp = caret::train(fm.temp, data = trainData.temp, method = "AdaBoost.M1", 
                               tuneGrid=expand.grid(mfinal=3, maxdepth=6, coeflearn="Breiman"),trControl = ctrl.temp),
  xgbTree.temp = caret::train(fm.temp, data = trainData.temp, method = "xgbTree", trControl = ctrl.temp,
                              tuneGrid=expand.grid(nrounds = 3, max_depth = 6, eta = 0.02, min_child_weight = 4,
                                                   subsample = 1, gamma = 1, colsample_bytree = 0.002)),
  knn.temp = caret::train(fm.temp, data = trainData.temp, method = "knn",trControl = ctrl.temp,tuneGrid=expand.grid(k=6)),
  svm.temp = caret::train(fm.temp, data = trainData.temp, method = "svmRadial", trControl = ctrl.temp,tuneGrid=expand.grid(C=1, sigma=0.4)),
  nnet.temp = caret::train(fm.temp, data = trainData.temp, method = "nnet", trControl = ctrl.temp,
                           tuneGrid=expand.grid(size=5, decay=0.1), linout=TRUE, trace=FALSE),
  gbm.temp = caret::train(fm.temp, data = trainData.temp, method = "gbm", trControl = ctrl.temp,
                          tuneGrid=expand.grid(n.trees=300, interaction.depth=3, shrinkage=0.02, n.minobsinnode=10),
                          verbose=FALSE),
  naivesbayes.temp = caret::train(fm.temp, data = trainData.temp, method = "naive_bayes", trControl = ctrl.temp,
                                  tuneGrid=expand.grid(laplace=0, usekernel=TRUE, adjust=1))
)

### Training metrics for temporally matched with Space-time CV
results.temp <- resamples(models.temp)
summary(results.temp)
metrics.temp <- as.data.frame(results.temp$values)
metrics_long.temp <- metrics.temp %>% pivot_longer(cols = -Resample, names_to = "Model", values_to = "Value") %>%
  separate(Model, into = c("Model", "Metric"), sep = "~")

as.data.frame(metrics_long.temp)
metrics_long.temp$Model <- str_extract(metrics_long.temp$Model, "[^.]+")

accuracy.temp <- metrics_long.temp %>% filter(Metric == "Accuracy")
av_acc.temp <- metrics_long.temp %>%filter(Metric == "Accuracy") %>%group_by(Model) %>%
  summarise(Avg_accuracy = mean(Value, na.rm = TRUE))

bal_accuracy.temp <- metrics_long.temp %>% filter(Metric == "Mean_Balanced_Accuracy")
bal_av_acc.temp <- metrics_long.temp %>%filter(Metric == "Mean_Balanced_Accuracy") %>%group_by(Model) %>%
  summarise(bal_Avg_accuracy = mean(Value, na.rm = TRUE))

auc.temp <- metrics_long.temp %>% filter(Metric == "prAUC")
auc_av_acc.temp <- metrics_long.temp %>%filter(Metric == "prAUC") %>%group_by(Model) %>%
  summarise(auc_Avg_accuracy = mean(Value, na.rm = TRUE))

f1.temp <- metrics_long.temp %>% filter(Metric == "Mean_F1")
f1_av_acc.temp <- metrics_long.temp %>%filter(Metric == "Mean_F1") %>%group_by(Model) %>%
  summarise(f1_Avg_accuracy = mean(Value, na.rm = TRUE))

acc.temp.plt <- ggplot(accuracy.temp, aes(x = Model, y = Value)) +
  stat_halfeye(adjust = 0.5, justification = -0.2, .width = 0, point_colour = NA) +
  geom_boxplot(fill = "lightblue", color = "darkblue",width = 0.2, outlier.color = NA, alpha = NA) +
  stat_dots(side = "left", justification = 1, binwidth = NA, dotsize = 0.1, overlaps = "nudge") +
  theme_bw() +geom_text(data = av_acc.temp, aes(x = Model, y = Avg_accuracy, label = round(Avg_accuracy, 2)),
                        vjust = -5, hjust = 0.5, size =5, color = "navy") +
  labs(title = "Overall accuracy", x = "", y = "") + ylim(0.3,0.6)+
  theme(legend.position = "none", text = element_text(size = 14, color = "black"), 
        axis.title = element_text(size = 14, color = "black"), 
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
        axis.text.x = element_text(size = 14, color = "black",angle = 90, hjust = 1),
        axis.text.y = element_text(size = 14, color = "black"))

bal_acc.temp.plt <- ggplot(bal_accuracy.temp, aes(x = Model, y = Value)) +
  stat_halfeye(adjust = 0.5, justification = -0.2, .width = 0, point_colour = NA) +
  geom_boxplot(fill = "lightblue", color = "darkblue",width = 0.2, outlier.color = NA, alpha = NA) +
  stat_dots(side = "left", justification = 1, binwidth = NA, dotsize = 0.1, overlaps = "nudge") +
  theme_bw() +geom_text(data = bal_av_acc.temp, aes(x = Model, y = bal_Avg_accuracy, label = round(bal_Avg_accuracy, 2)),
                        vjust = -5, hjust = 0.5, size = 5, color = "navy") +
  labs(title = "Balanced accuracy", x = "", y = "") + ylim(0.3,0.6)+
  theme(legend.position = "none", text = element_text(size = 14, color = "black"), 
        axis.title = element_text(size = 14, color = "black"), 
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
        axis.text.x = element_text(size = 14, color = "black",angle = 90, hjust = 1),
        axis.text.y = element_text(size = 14, color = "black"))

auc_acc.temp.plt <- ggplot(auc.temp, aes(x = Model, y = Value)) +
  stat_halfeye(adjust = 0.5, justification = -0.2, .width = 0, point_colour = NA) +
  geom_boxplot(fill = "lightblue", color = "darkblue",width = 0.2, outlier.color = NA, alpha = NA) +
  stat_dots(side = "left", justification = 1, binwidth = NA, dotsize = 0.1, overlaps = "nudge") +
  theme_bw() +geom_text(data = auc_av_acc.temp, aes(x = Model, y = auc_Avg_accuracy, label = round(auc_Avg_accuracy, 2)),
                        vjust = -5, hjust = 0.5, size =5, color = "navy") +
  labs(title = "pr-Area under curve", x = "", y = "") + ylim(0.1,0.6)+
  theme(legend.position = "none", text = element_text(size = 14, color = "black"), 
        axis.title = element_text(size = 14, color = "black"), 
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
        axis.text.x = element_text(size = 14, color = "black",angle = 90, hjust = 1),
        axis.text.y = element_text(size = 14, color = "black"))

f1_acc.temp.plt <- ggplot(f1.temp, aes(x = Model, y = Value)) +
  stat_halfeye(adjust = 0.5, justification = -0.2, .width = 0, point_colour = NA) +
  geom_boxplot(fill = "lightblue", color = "darkblue",width = 0.2, outlier.color = NA, alpha = NA) +
  stat_dots(side = "left", justification = 1, binwidth = NA, dotsize = 0.1, overlaps = "nudge") +
  theme_bw() +geom_text(data = f1_av_acc.temp, aes(x = Model, y = f1_Avg_accuracy, label = round(f1_Avg_accuracy, 2)),
                        vjust = -5, hjust = 0.5, size =5, color = "navy") +
  labs(title = "F1-score", x = "", y = "") + ylim(0.1,0.6)+
  theme(legend.position = "none", text = element_text(size = 14, color = "black"), 
        axis.title = element_text(size = 14, color = "black"), 
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
        axis.text.x = element_text(size = 14, color = "black",angle = 90, hjust = 1),
        axis.text.y = element_text(size = 14, color = "black"))
#getModelInfo()$multinom$parameters

png(file = "Training_Metrics_Aflatoxin_ST-CV.png", width = 14000, height =9000, units = "px", res = 850, type = "cairo")
grid.arrange(acc.temp.plt, bal_acc.temp.plt, auc_acc.temp.plt, f1_acc.temp.plt,ncol=2)
dev.off()

###### TESTING Evaluation metrics
#### Temporarly matched ST-CV
# Initialize an empty list to store predictions
predictions.temp <- list()

# Loop through each model in the models.temp list to make predictions
for(model_name in names(models.temp)) {
  model <- models.temp[[model_name]]
  predictions.temp[[model_name]] <- predict(model, newdata = testData.temp)
}
outcome_variable_name <- "class"  # Change this to your actual outcome variable name

evaluation_results.temp <- list()
for(model_name in names(predictions.temp)) {
  prediction <- predictions.temp[[model_name]]
  actual <- testData.temp[[outcome_variable_name]]
  evaluation_results.temp[[model_name]] <- confusionMatrix(prediction, actual)
}

write.csv(evaluation_results.temp$naivesbayes.temp$byClass,"rangertemp.csv")



### Load the test metrics and make plots
test.ST.temp <- read.csv("TestMetrics_STCV.csv")

png(file = "Test_Metrics_Aflatoxin_STCV.png", width = 14000, height =9000, units = "px", res = 850, type = "cairo")
ggplot(test.ST.temp, aes(x = Metric, y = Class, fill = Value)) +
  geom_tile(color = "white") + facet_wrap(~ Model) +scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "", x = "", y = "") + theme_minimal() +
  theme(legend.position = "none",text = element_text(size = 20, color = "black"),axis.title = element_text(size = 14, color = "black"),
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),axis.text.x = element_text(size = 18, color = "black", hjust = 0.5),
        axis.text.y = element_text(size = 18, color = "black")) +
  geom_text(aes(label = sprintf("%.2f", Value)), size = 6, color = "black")+
  scale_x_discrete(labels = c("Balanced accuracy" = "BA","F1-score" = "F1"))
dev.off()

#### Variable importance and partial dependence plots
### Variable importance for temporally matched GBM model
imp.adaboost.temp <- varImp(models.temp$adaboost)
imp.adaboost.temp2 <- as.data.frame(imp.adaboost.temp$importance)
imp.adaboost.temp2$varnames <- rownames(imp.adaboost.temp2)

png(file = "VarImp_Aflatoxin.png", width = 6000, height =5000, units = "px", res = 600, type = "cairo")
ggplot(imp.adaboost.temp2, aes(x=reorder(varnames, Overall), y=Overall)) +  geom_point(color="blue",size=4)+
  ggtitle("Adaptive gradient boosting")+ xlab("") + ylab("")+ coord_flip()+theme_tq() + 
  theme(plot.title = element_text(size=14, hjust = 0.5, color="black"),
        text = element_text(size = 14, face="bold",color = "black"))
dev.off()

##### partial dependence plots
library(pdp)
pdp.temp = topPredictors(models.temp$adaboost,n=3)
pd.temp.c1 <- NULL
for (i in pdp.temp) {
  tmp.c1 <- partial(models.temp$adaboost, pred.var = i, data = trainData.temp, type = "classification",which.class = 1L)
  names(tmp.c1) <- c("x", "y")
  pd.temp.c1 <- rbind(pd.temp.c1, cbind(tmp.c1, predictor = i))
}
pd.temp.c1$predictor <- factor(pd.temp.c1$predictor, levels = unique( pd.temp.c1$predictor))


pd.temp.c1.plt <- ggplot(pd.temp.c1, aes(x, y)) + geom_line(linewidth=0.5) + theme_classic() +
  theme(text = element_text(size = 14, face="bold",
                            color = "black"),axis.text.y = element_text(size=14, face="bold",color = "black"),
        plot.title = element_text(hjust = 0.5,color="navy"),axis.text = element_text(size = 14,color = "black"))+
  ggtitle("GBM - low risk")+ ylab("log-odds") +xlab("") + facet_wrap(~ predictor, scales = "free")

### Class 2
pd.temp.c2 <- NULL
for (i in pdp.temp) {
  tmp.c2 <- partial(models.temp$adaboost, pred.var = i, data = trainData.temp, type = "classification",which.class = 2L)
  names(tmp.c2) <- c("x", "y")
  pd.temp.c2 <- rbind(pd.temp.c2, cbind(tmp.c2, predictor = i))
}
pd.temp.c2$predictor <- factor(pd.temp.c2$predictor, levels = unique( pd.temp.c2$predictor))


pd.temp.c2.plt <- ggplot(pd.temp.c2, aes(x, y)) + geom_line(linewidth=0.5) + theme_classic() +
  theme(text = element_text(size = 14, face="bold",
                            color = "black"),axis.text.y = element_text(size=14, face="bold",color = "black"),
        plot.title = element_text(hjust = 0.5,color="navy"),axis.text = element_text(size = 14,color = "black"))+
  ggtitle("GBM - medium risk")+ ylab("log-odds") +xlab("") + facet_wrap(~ predictor, scales = "free")

### Class 3
pd.temp.c3 <- NULL
for (i in pdp.temp) {
  tmp.c3 <- partial(models.temp$adaboost, pred.var = i, data = trainData.temp, type = "classification",which.class = 3L)
  names(tmp.c3) <- c("x", "y")
  pd.temp.c3 <- rbind(pd.temp.c3, cbind(tmp.c3, predictor = i))
}
pd.temp.c3$predictor <- factor(pd.temp.c3$predictor, levels = unique( pd.temp.c3$predictor))

pd.temp.c3.plt <- ggplot(pd.temp.c3, aes(x, y)) + geom_line(linewidth=0.5) + theme_classic() +
  theme(text = element_text(size = 14, face="bold",
                            color = "black"),axis.text.y = element_text(size=14, face="bold",color = "black"),
        plot.title = element_text(hjust = 0.5,color="navy"),axis.text = element_text(size = 14,color = "black"))+
  ggtitle("GBM - high risk")+ ylab("log-odds") +xlab("") + facet_wrap(~ predictor, scales = "free")


png(file = "PartialPlots_Aflatoxin_STCV.png", width = 8000, height =8000, units = "px", res = 600, type = "cairo")
grid.arrange(pd.temp.c1.plt,pd.temp.c2.plt,pd.temp.c3.plt,nrow= 3)
dev.off()


#### extrapolate the predictions and calculate novel conditions
gbm.ras.temp18 <- rast("retainedpreds_Afla_masked_2018.tif")
names(gbm.ras.temp18) <- c("stempJun","stempMay","mintempMar","pdsiSep","mintempApr","precOct","mintempJan","stempJul", 
                           "stempJan","mintempJul","pdsiJan","mintempOct")
pred.gbm.temp18 <- predict(object=gbm.ras.temp18,model=models.temp$adaboost,na.rm=T)
gc()
writeRaster(pred.gbm.temp18,"prediction_afla_wet2018.tif",overwrite=TRUE)

names.novel.temp <- c("stempJun","stempMay","mintempMar","pdsiSep","mintempApr","precOct","mintempJan","stempJul",  
                      "stempJan","mintempJul","pdsiJan","mintempOct")
d.novel.temp.frame <- d.temp[,c(names.novel.temp)]
novel.test.temp <- ensemble.novel.object(d.novel.temp.frame, name="noveltest")

rastack.novel.temp18 <- stack("retainedpreds_Afla_masked_2018.tif")
names(rastack.novel.temp18) <- c("stempJun","stempMay","mintempMar","pdsiSep","mintempApr","precOct","mintempJan","stempJul", 
                           "stempJan","mintempJul","pdsiJan","mintempOct")
novel.raster.temp18 <- ensemble.novel(x=rastack.novel.temp18, novel.object=novel.test.temp)
writeRaster(novel.raster.temp18,"prediction_afla_wet2018_novel.tif",overwrite=TRUE)

### 2021
gbm.ras.temp21 <- rast("retainedpreds_Afla_masked_2021.tif")
names(gbm.ras.temp21) <- c("stempJun","stempMay","mintempMar","pdsiSep","mintempApr","precOct","mintempJan","stempJul", 
                           "stempJan","mintempJul","pdsiJan","mintempOct")
pred.gbm.temp21 <- predict(object=gbm.ras.temp21,model=models.temp$adaboost,na.rm=T)
gc()
writeRaster(pred.gbm.temp21,"prediction_afla_dry2021.tif",overwrite=TRUE)

rastack.novel.temp21 <- stack("retainedpreds_Afla_masked_2021.tif")
names(rastack.novel.temp21) <- c("stempJun","stempMay","mintempMar","pdsiSep","mintempApr","precOct","mintempJan","stempJul", 
                                 "stempJan","mintempJul","pdsiJan","mintempOct")
novel.raster.temp21 <- ensemble.novel(x=rastack.novel.temp21, novel.object=novel.test.temp)
writeRaster(novel.raster.temp21,"prediction_afla_wet2021_novel.tif",overwrite=TRUE)


##### Future projections assuming increase in temp by 1% and decrease in rainfall by 1%
## Wet year 2018
gbm.ras.temp18F <- rast("retainedpreds_Afla_masked_2018.tif")
names(gbm.ras.temp18F) <- c("stempJun","stempMay","mintempMar","pdsiSep","mintempApr","precOct","mintempJan","stempJul", 
                            "stempJan","mintempJul","pdsiJan","mintempOct")
gbm.ras.temp18F$precOct <- gbm.ras.temp18F$precOct*0.99
pred.gbm.temp18F <- predict(object=gbm.ras.temp18F,model=models.temp$adaboost,na.rm=T)
gc()
writeRaster(pred.gbm.temp18F,"prediction_afla_wet2018F.tif",overwrite=TRUE)

rastack.novel.temp18F <- stack("retainedpreds_Afla_masked_2018.tif")
names(rastack.novel.temp18F) <- c("stempJun","stempMay","mintempMar","pdsiSep","mintempApr","precOct","mintempJan","stempJul", 
                                  "stempJan","mintempJul","pdsiJan","mintempOct")
rastack.novel.temp18F$precOct <- rastack.novel.temp18F$precOct*0.99
novel.raster.temp18F <- ensemble.novel(x=rastack.novel.temp18F, novel.object=novel.test.temp)
writeRaster(novel.raster.temp18F,"prediction_afla_wet2018F_novel.tif",overwrite=TRUE)

##Dry year 2021
gbm.ras.temp21F <- rast("retainedpreds_Afla_masked_2021.tif")
names(gbm.ras.temp21F) <- c("stempJun","stempMay","mintempMar","pdsiSep","mintempApr","precOct","mintempJan","stempJul", 
                            "stempJan","mintempJul","pdsiJan","mintempOct")
gbm.ras.temp21F$precOct <- gbm.ras.temp21F$precOct*0.99
pred.gbm.temp21F <- predict(object=gbm.ras.temp21F,model=models.temp$adaboost,na.rm=T)
gc()
writeRaster(pred.gbm.temp21F,"prediction_afla_dry2021F.tif",overwrite=TRUE)

rastack.novel.temp21F <- stack("retainedpreds_Afla_masked_2021.tif")
names(rastack.novel.temp21F) <- c("stempJun","stempMay","mintempMar","pdsiSep","mintempApr","precOct","mintempJan","stempJul", 
                                  "stempJan","mintempJul","pdsiJan","mintempOct")
rastack.novel.temp21F$precOct <- rastack.novel.temp21F$precOct*0.99
novel.raster.temp21F <- ensemble.novel(x=rastack.novel.temp21F, novel.object=novel.test.temp)
writeRaster(novel.raster.temp21F,"prediction_afla_dry2021F_novel.tif",overwrite=TRUE)

##############************************THE END**********************#################

