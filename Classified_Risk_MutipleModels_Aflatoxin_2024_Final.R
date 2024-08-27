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
            "terra", "stars", "rasterVis", "sp", "virtualspecies","ggdist","BiodiversityR", "pdp","Boruta")
  sapply(libs, require, character.only = TRUE)
}

load_libraries()

#### Set the working directory
setwd("C:/Users/gstel/OneDrive/Desktop/IITA/Maize Afalatoxin 2024/Projects/Aflatoxin_Risk_in_Maize")

######## Data aggregation to reduce spatial noise #####
# Load the temporally matched database
d.temp <- read.csv("cleaned_afla_FINALmonthly24_nonzero.csv")
names(d.temp)
d.temp_sf <- st_as_sf(d.temp, coords = c("Long", "Lat"), crs = 4326)
d.temp_sf <- st_transform(d.temp_sf, crs = 32736)  # UTM zone 36S

# Create grid covering the extent of the data
grid_res <- 5000  # 5 km grid resolution
bbox <- st_bbox(d.temp_sf)
grid <- st_make_grid(st_as_sfc(bbox), cellsize = grid_res, square = TRUE)
grid_sf <- st_as_sf(grid, crs = 32736)

# Add unique ID to each grid cell
grid_sf <- grid_sf %>% mutate(grid_id = row_number())

# Plot grid cells to verify
ggplot() + geom_sf(data = grid_sf, fill = NA, color = "blue") +
  geom_sf(data = d.temp_sf, color = "red") + labs(title = "Grid Cells and Points") +
  theme_minimal()

# Ensure that grid cells and points are in the same CRS
grid_sf <- st_transform(grid_sf, crs = st_crs(d.temp_sf))
#st_write(grid_sf, "grid_sf10km.shp")

# Assign each point to the nearest grid cell
d.temp_sf <- d.temp_sf %>% st_join(grid_sf, join = st_nearest_feature)

# Verify assignment
print(d.temp_sf %>% st_drop_geometry() %>% head())

# Perform aggregation by grid cell ID and year
majority <- function(x) {as.numeric(names(sort(table(x), decreasing = TRUE))[1])}
d.temp_agg <- d.temp_sf %>% group_by(grid_id, Year2) %>%
  summarize(
    Country = first(Country),
    District = first(District),
    Aflatoxin = mean(Aflatoxin, na.rm = TRUE),
    PrecClus = majority(PrecClus),
    Lat = mean(avgLat), 
    Long = mean(avgLong), 
    across(PH:smoistDec, ~mean(.x, na.rm = TRUE)),
    .groups = 'drop' ) %>% mutate(
    class = case_when(
      Aflatoxin < 5 ~ "c1",
      Aflatoxin >= 5 & Aflatoxin <= 20 ~ "c2",
      Aflatoxin > 20 ~ "c3",
      TRUE ~ NA_character_))

#### Remove Geometry before saving the CSV file
if ("geometry" %in% names(d.temp_agg)) {d.temp_agg <- d.temp_agg %>%
    st_drop_geometry()}

# Save aggregated data to CSV
#write.csv(d.temp_agg, "aggregated5km_afla_mean.csv")

###### *********************** THE END *****************************************

####### MODELLING PART #########
# Load the aggregated data and convert the categorical columns as needed
d.agg <- read.csv("aggregated5km_afla_mean.csv")
names(d.agg)
d.agg$class <- as.factor(d.agg$class)
d.agg$Year <- as.factor(d.agg$Year)
d.agg$PrecClus <- as.factor(d.agg$PrecClus)

#### Plot boxplots showing the aflatoxin prevalence
png(file = "Aflatoxin_prevalence.png", width = 7000, height =5000, units = "px", res = 700, type = "cairo")
ggplot(d.agg, aes(x=factor(Year), y=Aflatoxin, fill=Country)) + 
  geom_boxplot() + ylim(0,300)+ facet_grid(Country ~ .) + 
  labs(title="",x="", y="Aflatoxin prevalence") +
  theme_ggdist()  + theme(legend.position = "none",text = element_text(size = 15, color = "black",face="bold"),
                          axis.text.y = element_text(size = 14, face="bold",color="black"),
                          axis.text.x = element_text(angle = 90, hjust = 1, size=14, color="black",face="bold"),
                          plot.title = element_text(hjust = 0.5,size=15, color="black",face="bold"))
dev.off()

#### Plot the distribution of the various risk classes per year and country
png(file = "Distribution_of_sample_locations.png", width = 15000, height =12000, units = "px", res = 900, type = "cairo")
ggplot(d.agg, aes(x=class, fill=class)) + 
  geom_bar() + facet_grid(Country ~ Year) + labs(title="",x = "",y = "Number of locations") +
  theme_bw()  + theme(legend.position = "top",text = element_text(size = 20, color = "black",face="bold"),
                      axis.text.y = element_text(size = 20, face="bold",color="black"),
                      axis.text.x = element_text(angle = 90, hjust = 1, size=20, color="black",face="bold"),
                      plot.title = element_text(hjust = 0.5,size=20, color="black",face="bold"))
dev.off()

##### Data partition using space-time partitioning ####
set.seed(123)  # For reproducibility
# Create stratified sampling indices
trainIndex <- createDataPartition(d.agg$class, p = 0.8, list = FALSE, times = 1)
trainData.agg <- d.agg[trainIndex, ]
testData.agg <- d.agg[-trainIndex, ]

# Verify the split
print(dim(trainData.agg))
print(dim(testData.agg))
print(table(trainData.agg$class))
print(table(testData.agg$class))

##### Extraction of predictors (x) and response variables (y) for each set
# Split data into predictors (X) and response variable (y)
x.agg <- trainData.agg[, c(10:111)]
y.agg <- trainData.agg$class

##### Feature elimination using VSURF  #####
set.seed(123)
vsurf.agg <- VSURF(x=x.agg,y=trainData.agg$class ,mtry=10,ntree.thres = 300,
                   nfor.thres = 30,ntree.interp = 300,nfor.interp = 30)
vsurf.agg$varselect.pred

thres.var <- data.frame(x.agg[,c(22, 44,  4, 26, 38)])
names(thres.var)

# Define the formula for model fitting
fm.agg <- class ~ mintempMar+dem+stempMay+precMar+stempJun


##### MODEL TRAINING with SPACE-TIME CV $$$$$$$$$$$$ #####
folds.agg <- CreateSpacetimeFolds(trainData.agg, spacevar = "PrecClus", timevar="Year",k = 3, class = "class")

ctrl.agg <- trainControl(method = "repeatedcv", number = 5, repeats=3, savePredictions = "all", verboseIter = TRUE,index = folds.agg$index,
                         classProbs = TRUE, summaryFunction = multiClassSummary,selectionFunction = "best",allowParallel = TRUE)

###### Temporally matched with space-time cross validation
models.agg <- list(
  ranger.agg = caret::train(fm.agg, data = trainData.agg, method = "ranger", trControl = ctrl.agg,importance="permutation",
                            tuneGrid = expand.grid(.mtry =3, .splitrule = "extratrees", .min.node.size = 5), num.trees = 500,max.depth=6,min.bucket=1),
  adaboost.agg = caret::train(fm.agg, data = trainData.agg, method = "AdaBoost.M1", 
                              tuneGrid=expand.grid(mfinal=3, maxdepth=6, coeflearn="Breiman"),trControl = ctrl.agg),
  xgbTree.agg = caret::train(fm.agg, data = trainData.agg, method = "xgbTree", trControl = ctrl.agg,tuneGrid = expand.grid(nrounds = 10, 
                                  max_depth = 8, eta =  0.05, min_child_weight = 5,subsample = 0.8, gamma = 1,colsample_bytree =  0.7)),
  knn.agg = caret::train(fm.agg, data = trainData.agg, method = "knn",trControl = ctrl.agg,tuneGrid=expand.grid(k=6)),
  svm.agg = caret::train(fm.agg, data = trainData.agg, method = "svmRadial", trControl = ctrl.agg,tuneGrid=expand.grid(C=1, sigma=0.3)),
  nnet.agg = caret::train(fm.agg, data = trainData.agg, method = "nnet", trControl = ctrl.agg,tuneGrid=expand.grid(size=6, decay=0.2), 
                          linout=FALSE, trace=FALSE),
  gbm.agg = caret::train(fm.agg, data = trainData.agg, method = "gbm", trControl = ctrl.agg,tuneGrid = expand.grid(n.trees = 500, 
                                                interaction.depth = 6, shrinkage =0.03, n.minobsinnode = 10),verbose=FALSE),
  naivesbayes.agg = caret::train(fm.agg, data = trainData.agg, method = "naive_bayes", trControl = ctrl.agg,tuneGrid=expand.grid(laplace=1,
                                                                                      usekernel=TRUE, adjust=1))
)

### Training metrics for temporally matched with Space-time CV
results.agg <- resamples(models.agg)
summary(results.agg)
metrics.agg <- as.data.frame(results.agg$values)
metrics_long.agg <- metrics.agg %>% pivot_longer(cols = -Resample, names_to = "Model", values_to = "Value") %>%
  separate(Model, into = c("Model", "Metric"), sep = "~")

as.data.frame(metrics_long.agg)
metrics_long.agg$Model <- str_extract(metrics_long.agg$Model, "[^.]+")

accuracy.agg <- metrics_long.agg %>% filter(Metric == "Accuracy")
av_acc.agg <- metrics_long.agg %>%filter(Metric == "Accuracy") %>%group_by(Model) %>%
  summarise(Avg_accuracy = median(Value, na.rm = TRUE))

bal_accuracy.agg <- metrics_long.agg %>% filter(Metric == "Mean_Balanced_Accuracy")
bal_av_acc.agg <- metrics_long.agg %>%filter(Metric == "Mean_Balanced_Accuracy") %>%group_by(Model) %>%
  summarise(bal_Avg_accuracy = median(Value, na.rm = TRUE))

auc.agg <- metrics_long.agg %>% filter(Metric == "prAUC")
auc_av_acc.agg <- metrics_long.agg %>%filter(Metric == "prAUC") %>%group_by(Model) %>%
  summarise(auc_Avg_accuracy = median(Value, na.rm = TRUE))

f1.agg <- metrics_long.agg %>% filter(Metric == "Mean_F1")
f1_av_acc.agg <- metrics_long.agg %>%filter(Metric == "Mean_F1") %>%group_by(Model) %>%
  summarise(f1_Avg_accuracy = median(Value, na.rm = TRUE))

acc.agg.plt <- ggplot(accuracy.agg, aes(x = Model, y = Value)) +
  stat_halfeye(adjust = 0.5, justification = -0.2, .width = 0, point_colour = NA) +
  geom_boxplot(fill = "lightblue", color = "darkblue",width = 0.2, outlier.color = NA, alpha = NA) +
  stat_dots(side = "left", justification = 1, binwidth = NA, dotsize = 0.1, overlaps = "nudge") +
  theme_bw() +geom_text(data = av_acc.agg, aes(x = Model, y = Avg_accuracy, label = round(Avg_accuracy, 2)),
                        vjust = -5, hjust = 0.5, size =6, color = "navy") +
  labs(title = "Overall accuracy", x = "", y = "") + ylim(0.3,0.7)+
  theme(legend.position = "none", text = element_text(size = 16, color = "black"), 
        axis.title = element_text(size = 16, color = "black"), 
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
        axis.text.x = element_text(size = 16, color = "black",angle = 90, hjust = 1),
        axis.text.y = element_text(size = 16, color = "black"))

bal_acc.agg.plt <- ggplot(bal_accuracy.agg, aes(x = Model, y = Value)) +
  stat_halfeye(adjust = 0.5, justification = -0.2, .width = 0, point_colour = NA) +
  geom_boxplot(fill = "lightblue", color = "darkblue",width = 0.2, outlier.color = NA, alpha = NA) +
  stat_dots(side = "left", justification = 1, binwidth = NA, dotsize = 0.1, overlaps = "nudge") +
  theme_bw() +geom_text(data = bal_av_acc.agg, aes(x = Model, y = bal_Avg_accuracy, label = round(bal_Avg_accuracy, 2)),
                        vjust = -5, hjust = 0.5, size = 6, color = "navy") +
  labs(title = "Balanced accuracy", x = "", y = "") + ylim(0.3,0.7)+
  theme(legend.position = "none", text = element_text(size = 16, color = "black"), 
        axis.title = element_text(size = 16, color = "black"), 
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
        axis.text.x = element_text(size = 16, color = "black",angle = 90, hjust = 1),
        axis.text.y = element_text(size = 16, color = "black"))

auc_acc.agg.plt <- ggplot(auc.agg, aes(x = Model, y = Value)) +
  stat_halfeye(adjust = 0.5, justification = -0.2, .width = 0, point_colour = NA) +
  geom_boxplot(fill = "lightblue", color = "darkblue",width = 0.2, outlier.color = NA, alpha = NA) +
  stat_dots(side = "left", justification = 1, binwidth = NA, dotsize = 0.1, overlaps = "nudge") +
  theme_bw() +geom_text(data = auc_av_acc.agg, aes(x = Model, y = auc_Avg_accuracy, label = round(auc_Avg_accuracy, 2)),
                        vjust = -5, hjust = 0.5, size =6, color = "navy") +
  labs(title = "pr-Area under curve", x = "", y = "") + ylim(0.2,0.6)+
  theme(legend.position = "none", text = element_text(size = 16, color = "black"), 
        axis.title = element_text(size = 16, color = "black"), 
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
        axis.text.x = element_text(size = 16, color = "black",angle = 90, hjust = 1),
        axis.text.y = element_text(size = 16, color = "black"))

f1_acc.agg.plt <- ggplot(f1.agg, aes(x = Model, y = Value)) +
  stat_halfeye(adjust = 0.5, justification = -0.2, .width = 0, point_colour = NA) +
  geom_boxplot(fill = "lightblue", color = "darkblue",width = 0.2, outlier.color = NA, alpha = NA) +
  stat_dots(side = "left", justification = 1, binwidth = NA, dotsize = 0.1, overlaps = "nudge") +
  theme_bw() +geom_text(data = f1_av_acc.agg, aes(x = Model, y = f1_Avg_accuracy, label = round(f1_Avg_accuracy, 2)),
                        vjust = -5, hjust = 0.5, size =6, color = "navy") +
  labs(title = "F1-score", x = "", y = "") + ylim(0.2,0.6)+
  theme(legend.position = "none", text = element_text(size = 16, color = "black"), 
        axis.title = element_text(size = 16, color = "black"), 
        plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
        axis.text.x = element_text(size = 16, color = "black",angle = 90, hjust = 1),
        axis.text.y = element_text(size = 16, color = "black"))

#getModelInfo()$svmLinear$parameters

png(file = "Training_Metrics_Aflatoxin_ST-CV.png", width = 14000, height =10000, units = "px", res = 850, type = "cairo")
grid.arrange(acc.agg.plt, bal_acc.agg.plt, auc_acc.agg.plt, f1_acc.agg.plt,ncol=2)
dev.off()

###### TESTING Evaluation metrics
#### Temporarly matched ST-CV
# Initialize an empty list to store predictions
predictions.agg <- list()

# Loop through each model in the models.agg list to make predictions
for(model_name in names(models.agg)) {
  model <- models.agg[[model_name]]
  predictions.agg[[model_name]] <- predict(model, newdata = testData.agg)
}
outcome_variable_name <- "class"  # Change this to your actual outcome variable name

evaluation_results.agg <- list()
for(model_name in names(predictions.agg)) {
  prediction <- predictions.agg[[model_name]]
  actual <- testData.agg[[outcome_variable_name]]
  evaluation_results.agg[[model_name]] <- confusionMatrix(prediction, actual)
}

write.csv(evaluation_results.agg$adaboost.agg$byClass,"adaboosttemp.csv")

### Load the test metrics and make plots
test.ST.agg <- read.csv("TestMetrics_STCV.csv")

png(file = "Test_Metrics_Aflatoxin_STCV.png", width = 14000, height =9000, units = "px", res = 850, type = "cairo")
ggplot(test.ST.agg, aes(x = Metric, y = Class, fill = Value)) +
  geom_tile(color = "white") + facet_wrap(~ Model) +scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "", x = "", y = "") + theme_minimal() +
  theme(legend.position = "none",text = element_text(size = 20, color = "black"),axis.title = element_text(size = 14, color = "black"),
        plot.title = element_text(size = 20, hjust = 0.5, face = "bold"),axis.text.x = element_text(size = 18, color = "black", hjust = 0.5),
        axis.text.y = element_text(size = 20, color = "black")) +
  geom_text(aes(label = sprintf("%.2f", Value)), size = 8, color = "black")+
  scale_x_discrete(labels = c("Balanced accuracy" = "BA","F1-score" = "F1", "Precision"="PR"))
dev.off()

#### Variable importance and partial dependence plots
imp.ranger.agg <- varImp(models.agg$ranger.agg)
imp.ranger.agg2 <- as.data.frame(imp.ranger.agg$importance)
imp.ranger.agg2$varnames <- rownames(imp.ranger.agg2)

png(file = "VarImp_Aflatoxin.png", width = 6000, height =5000, units = "px", res = 600, type = "cairo")
ggplot(imp.ranger.agg2, aes(x=reorder(varnames, Overall), y=Overall)) +  geom_point(color="blue",size=4)+
  ggtitle("Ranger")+ xlab("") + ylab("")+ coord_flip()+theme_tq() + 
  theme(plot.title = element_text(size=14, hjust = 0.5, color="black"),
        text = element_text(size = 14, face="bold",color = "black"))
dev.off()

##### partial dependence plots
library(pdp)
pdp.agg = topPredictors(models.agg$ranger.agg,n=3)
pd.agg.c1 <- NULL
for (i in pdp.agg) {
  tmp.c1 <- partial(models.agg$ranger.agg, pred.var = i, data = trainData.agg, type = "classification",which.class = 1L)
  names(tmp.c1) <- c("x", "y")
  pd.agg.c1 <- rbind(pd.agg.c1, cbind(tmp.c1, predictor = i))
}
pd.agg.c1$predictor <- factor(pd.agg.c1$predictor, levels = unique( pd.agg.c1$predictor))


pd.agg.c1.plt <- ggplot(pd.agg.c1, aes(x, y)) + geom_line(linewidth=0.5) + theme_classic() +
  theme(text = element_text(size = 14, face="bold",
                            color = "black"),axis.text.y = element_text(size=14, face="bold",color = "black"),
        plot.title = element_text(hjust = 0.5,color="navy"),axis.text = element_text(size = 14,color = "black"))+
  ggtitle("Ranger - low risk (<5ppb)")+ ylab("log-odds") +xlab("") + facet_wrap(~ predictor, scales = "free")

### Class 2
pd.agg.c2 <- NULL
for (i in pdp.agg) {
  tmp.c2 <- partial(models.agg$ranger.agg, pred.var = i, data = trainData.agg, type = "classification",which.class = 2L)
  names(tmp.c2) <- c("x", "y")
  pd.agg.c2 <- rbind(pd.agg.c2, cbind(tmp.c2, predictor = i))
}
pd.agg.c2$predictor <- factor(pd.agg.c2$predictor, levels = unique( pd.agg.c2$predictor))


pd.agg.c2.plt <- ggplot(pd.agg.c2, aes(x, y)) + geom_line(linewidth=0.5) + theme_classic() +
  theme(text = element_text(size = 14, face="bold",
                            color = "black"),axis.text.y = element_text(size=14, face="bold",color = "black"),
        plot.title = element_text(hjust = 0.5,color="navy"),axis.text = element_text(size = 14,color = "black"))+
  ggtitle("Ranger - medium risk (>5-20ppb)")+ ylab("log-odds") +xlab("") + facet_wrap(~ predictor, scales = "free")

### Class 3
pd.agg.c3 <- NULL
for (i in pdp.agg) {
  tmp.c3 <- partial(models.agg$ranger.agg, pred.var = i, data = trainData.agg, type = "classification",which.class = 3L)
  names(tmp.c3) <- c("x", "y")
  pd.agg.c3 <- rbind(pd.agg.c3, cbind(tmp.c3, predictor = i))
}
pd.agg.c3$predictor <- factor(pd.agg.c3$predictor, levels = unique( pd.agg.c3$predictor))

pd.agg.c3.plt <- ggplot(pd.agg.c3, aes(x, y)) + geom_line(linewidth=0.5) + theme_classic() +
  theme(text = element_text(size = 14, face="bold",
                            color = "black"),axis.text.y = element_text(size=14, face="bold",color = "black"),
        plot.title = element_text(hjust = 0.5,color="navy"),axis.text = element_text(size = 14,color = "black"))+
  ggtitle("Ranger - high risk(>20ppb)")+ ylab("log-odds") +xlab("") + facet_wrap(~ predictor, scales = "free")


png(file = "PartialPlots_Aflatoxin_STCV.png", width = 8000, height =8000, units = "px", res = 600, type = "cairo")
grid.arrange(pd.agg.c1.plt,pd.agg.c2.plt,pd.agg.c3.plt,nrow= 3)
dev.off()


#### Spatial predictions, future projections and novel conditions calculations.
ras<- rast("Raster_stack_All.tif")
cropmask <- rast("CropMask_DEA_prj.tif")
cropmask.prj <- project(cropmask,ras)

ret.ras18 <- c("mintemp2018Mar","dem","stemp2018May","prec2018Mar","stemp2018Jun")
ranger.ras.agg18 <- ras[[ret.ras18]]
ranger.ras.agg18 <- mask(ranger.ras.agg18,cropmask.prj)
names(ranger.ras.agg18) <- c("mintempMar","dem","stempMay","precMar","stempJun")
plot(ranger.ras.agg18)
pred.ranger.agg18 <- predict(object=ranger.ras.agg18,model=models.agg$gbm.agg,na.rm=T)
gc()
plot(pred.ranger.agg18)
writeRaster(pred.ranger.agg18,"prediction_afla_wet2018_gbm.tif",overwrite=TRUE)

names.novel.agg <- c("mintempMar","dem","stempMay","precMar","stempJun")
d.novel.agg.frame <- d.agg[,c(names.novel.agg)]
novel.test.agg <- ensemble.novel.object(d.novel.agg.frame, name="noveltest")

rastack.novel.agg18 <- stack(ranger.ras.agg18)
novel.raster.agg18 <- ensemble.novel(x=rastack.novel.agg18, novel.object=novel.test.agg)
writeRaster(novel.raster.agg18,"prediction_afla_wet2018_novel_gbm.tif",overwrite=TRUE)

### 2021
ret.ras21 <- c("mintemp2021Mar","dem","stemp2021May","prec2021Mar","stemp2021Jun")
ranger.ras.agg21 <- ras[[ret.ras21]]
names(ranger.ras.agg21) <- c("mintempMar","dem","stempMay","precMar","stempJun")
ranger.ras.agg21 <- mask(ranger.ras.agg21,cropmask.prj)
pred.ranger.agg21 <- predict(object=ranger.ras.agg21,model=models.agg$gbm.agg,na.rm=T)
gc()
#plot(pred.ranger.agg21)
writeRaster(pred.ranger.agg21,"prediction_afla_dry2021_gbm.tif",overwrite=TRUE)

rastack.novel.agg21 <- stack(ranger.ras.agg21)
novel.raster.agg21 <- ensemble.novel(x=rastack.novel.agg21, novel.object=novel.test.agg)
writeRaster(novel.raster.agg21,"prediction_afla_dry2021_novel_gbm.tif",overwrite=TRUE)

##### Future projections assuming increase 15% increase or decrease
## Wet year 2018 #### Assume rainfall increase by 15% and temp decrease by 15%
ranger.ras.agg18F = ranger.ras.agg18
temperature_vars <- c("mintempMar","stempMay","stempJun")
ranger.ras.agg18F[[temperature_vars]] <- ranger.ras.agg18F[[temperature_vars]] * 0.85

prec_vars <- c("precMar")
ranger.ras.agg18F[[prec_vars]] <- ranger.ras.agg18F[[prec_vars]] * 1.15

pred.ranger.agg18F <- predict(object=ranger.ras.agg18F,model=models.agg$gbm.agg,na.rm=T)
gc()
#plot(pred.ranger.agg18F)
writeRaster(pred.ranger.agg18F,"prediction_afla_wet2018F_gbm.tif",overwrite=TRUE)

rastack.novel.agg18F <- stack(ranger.ras.agg18F)
novel.raster.agg18F <- ensemble.novel(x=rastack.novel.agg18F, novel.object=novel.test.agg)
writeRaster(novel.raster.agg18F,"prediction_afla_wet2018F_novel_gbm.tif",overwrite=TRUE)

##Dry year 2021
#### temperature will increase by 15% and rainfall decrease by 15%
ranger.ras.agg21F = ranger.ras.agg21
ranger.ras.agg21F[[temperature_vars]] <- ranger.ras.agg21F[[temperature_vars]] * 1.15
ranger.ras.agg21F[[prec_vars]] <- ranger.ras.agg21F[[prec_vars]] * 0.85
pred.ranger.agg21F <- predict(object=ranger.ras.agg21F,model=models.agg$gbm.agg,na.rm=T)
gc()
plot(pred.ranger.agg21F)
writeRaster(pred.ranger.agg21F,"prediction_afla_dry2021F_gbm.tif",overwrite=TRUE)

rastack.novel.agg21F <- stack(ranger.ras.agg21F)
novel.raster.agg21F <- ensemble.novel(x=rastack.novel.agg21F, novel.object=novel.test.agg)
writeRaster(novel.raster.agg21F,"prediction_afla_dry2021F_novel_gbm.tif",overwrite=TRUE)
##############************************THE END**********************#################

