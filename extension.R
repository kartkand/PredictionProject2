######################################################################
########
#
# Replication Data Political Analysis Forum: Comparing Random Forest with
# Logistic Regression for Predicting Class-Imbalanced Civil War Onset Data

######################################################################
########
# Set the Working Directory
# setwd("/Documents/R")
# getwd()
# data for prediction
data <- read.csv(file = "SambanisImp.csv")
# data for looking at Variable Importance Plots
data2 <- read.csv(file = "Amelia.Imp3.csv")
library(randomForest)
library(caret) # for CV folds and data splitting
library(ROCR) # for diagnostics and ROC plots/stats
library(pROC) # same as ROCR
library(stepPlr) # Firth’s logit implemented thru caret library
library(doMC) # for using multiple processor cores
library(xtable) # for writing Table 1 in Latex
### Use only the 88 variables specified in Sambanis (2006) Appendix###

# Convert DV into Factor with names for Caret Library###

data.full <- data[, c(
    "warstds", "ager", "agexp", "anoc", "army85", "autch98", "auto4",
    "autonomy", "avgnabo", "centpol3", "coldwar", "decade1", "decade2",
    "decade3", "decade4", "dem", "dem4", "demch98", "dlang", "drel",
    "durable", "ef", "ef2", "ehet", "elfo", "elfo2", "etdo4590",
    "expgdp", "exrec", "fedpol3", "fuelexp", "gdpgrowth", "geo1", "geo2",
    "geo34", "geo57", "geo69", "geo8", "illiteracy", "incumb", "infant",
    "inst", "inst3", "life", "lmtnest", "ln_gdpen", "lpopns", "major", "manuexp", "milper",
    "mirps0", "mirps1", "mirps2", "mirps3", "nat_war", "ncontig",
    "nmgdp", "nmdp4_alt", "numlang", "nwstate", "oil", "p4mchg",
    "parcomp", "parreg", "part", "partfree", "plural", "plurrel",
    "pol4", "pol4m", "pol4sq", "polch98", "polcomp", "popdense",
    "presi", "pri", "proxregc", "ptime", "reg", "regd4_alt", "relfrac", "seceduc",
    "second", "semipol3", "sip2", "sxpnew", "sxpsq", "tnatwar", "trade",
    "warhist", "xconst"
)]

data.full$warstds <- factor(
    data.full$warstds,
    levels = c(0, 1),
    labels = c("peace", "war")
)
vals <- list(
    339, 339, 232, 371, 305, 305, 300, 373, 370, 211, 211, 346, 355, 344,
    352, 316, 315, 315, 390, 390, 366, 366, 375, 220, 220, 372, 265, 260,
    255, 255, 350, 350, 310, 395, 205, 325, 705, 347, 367, 367, 223, 368,
    368, 212, 212, 343, 338, 359, 221, 341, 210, 210, 385, 385, 290, 290,
    235, 360, 365, 331, 317, 349, 230, 380, 225, 369, 200, 345, 345, 692,
    651, 630, 645, 666, 663, 690, 660, 620, 698, 694, 670, 652, 652, 640,
    696, 679, 678, 680
)
data.subset <- data.full[data$cowcode %in% vals, ]
data.test <- data.full[!(data$cowcode %in% vals), ]
# distribute workload over multiple cores for faster computation
registerDoMC(cores = 7)
set.seed(666)

tc <- trainControl(
    method = "cv",
    number = 10,
    summaryFunction = twoClassSummary,
    classProb = T,
    savePredictions = T,
    allowParallel = FALSE,
)
# Fearon and Laitin Model (2003) Specification###
model.fl.1 <-
    train(as.factor(warstds) ~ warhist + ln_gdpen + lpopns + lmtnest + ncontig + oil + nwstate
        + inst3 + pol4 + ef + relfrac, # FL 2003 model spec
    metric = "ROC", method = "glm", family = "binomial",
    trControl = tc, data = data.subset
    )
summary(model.fl.1)
model.fl.1
### Fearon and Laitin (2003) penalized logistic regression
model.fl.2 <-
    train(as.factor(warstds) ~ warhist + ln_gdpen + lpopns + lmtnest + ncontig + oil + nwstate
        + inst3 + pol4 + ef + relfrac, # FL 2003 model spec
    metric = "ROC", method = "plr",
    trControl = tc, data = data.subset
    )
summary(model.fl.2)
model.fl.2
### Collier and Hoeffler (2004) Model specification###
model.ch.1 <-
    train(as.factor(warstds) ~ sxpnew + sxpsq + ln_gdpen + gdpgrowth + warhist + lmtnest + ef + popdense + lpopns + coldwar + seceduc + ptime,
        metric = "ROC", method = "glm", family = "binomial",
        trControl = tc, data = data.subset
    )
model.ch.1
### Collier and Hoeffler penalized logistic regression###
model.ch.2 <-
    train(as.factor(warstds) ~ sxpnew + sxpsq + ln_gdpen + gdpgrowth + warhist + lmtnest + ef + popdense
        + lpopns + coldwar + seceduc + ptime,
    metric = "ROC", method = "plr",
    trControl = tc, data = data.subset
    )
model.ch.2
### Hegre and Sambanis (2006) Model Specification###
model.hs.1 <-
    train(warstds ~ lpopns + ln_gdpen + inst3 + parreg + geo34 + proxregc + gdpgrowth + anoc +
        partfree + nat_war + lmtnest + decade1 + pol4sq + nwstate + regd4_alt + etdo4590 + milper +
        geo1 + tnatwar + presi,
    metric = "ROC", method = "glm", family = "binomial",
    trControl = tc, data = data.subset
    )
model.hs.1

model.hs.2 <-
    train(warstds ~ lpopns + ln_gdpen + inst3 + parreg + geo34 + proxregc + gdpgrowth + anoc +
        partfree + nat_war + lmtnest + decade1 + pol4sq + nwstate + regd4_alt + etdo4590 + milper +
        geo1 + tnatwar + presi,
    metric = "ROC", method = "plr",
    trControl = tc, data = data.subset
    )
model.hs.2

model.rf <- train(as.factor(warstds) ~ .,
    metric = "ROC", method = "rf",
    sampsize = c(30, 20),
    importance = T,
    proximity = F, ntree = 1000,
    trControl = tc, data = data.subset
)
model.rf

library(ROCR)
attach(data.subset)

# pred.FL.war <- model.fl.1$finalModel$fitted.values
# pred.CH.war <- model.ch.1$finalModel$fitted.values
# pred.HR.war <- model.hs.1$finalModel$fitted.values
### Notice the key difference between original code and revised code is the
### $finalModel$fitted.values this extracts the predicted probabilities
### from the best caret CV model, ensuring ROC curves drawn will match AUC scores.
### predicted probabilities for the Random Forest model
RF.1.pred <- predict(model.rf, type = "prob", newdata = data.test)
RF.1.pred <- as.data.frame(RF.1.pred)
pred.RF.1 <- prediction(RF.1.pred$war, data.test$warstds)
perf.RF.1 <- performance(pred.RF.1, "tpr", "fpr")
auc.tmp <- performance(pred.RF.1, "auc")
auc.RF.logit <- as.numeric(auc.tmp@y.values)
pred.CH.war <- predict(model.ch.1, newdata = data.test, type = "prob")
pred.CH.war <- as.data.frame(pred.CH.war)
pred.HS.war <- predict(model.hs.1, newdata = data.test, type = "prob")
pred.HS.war <- as.data.frame(pred.HS.war)
pred.FL.war <- predict(model.fl.1, newdata = data.test, type = "prob")
pred.FL.war <- as.data.frame(pred.FL.war)
nrow(pred.FL.war)
pred.FL <- prediction(pred.FL.war$war, data.test$warstds)
perf.FL <- performance(pred.FL, "tpr", "fpr")
auc.tmp <- performance(pred.FL, "auc")
auc.FL.logit <- as.numeric(auc.tmp@y.values)
pred.CH <- prediction(pred.CH.war$war, data.test$warstds)
perf.CH <- performance(pred.CH, "tpr", "fpr")
auc.tmp <- performance(pred.CH, "auc")
auc.CH.logit <- as.numeric(auc.tmp@y.values)
pred.HS <- prediction(pred.HS.war$war, data.test$warstds)
perf.HS <- performance(pred.HS, "tpr", "fpr")
auc.tmp <- performance(pred.HS, "auc")
auc.HS.logit <- as.numeric(auc.tmp@y.values)
### Code for plotting the corrected ROC Curves in Figure 1.
jpeg("plot1.jpg")
plot(perf.FL, main = "Uncorrected Logits")
plot(perf.CH, add = T, lty = 2)
plot(perf.HS, add = T, lty = 3)
plot(perf.RF.1, add = T, lty = 4)
legend(0.32, 0.25, c(
    paste("Fearon and Laitin (2003)", auc.FL.logit, sep = " "),
    paste("Collier and Hoeffler (2004)", auc.CH.logit, sep = " "),
    paste("Hegre and Sambanis (2006)", auc.HS.logit, sep = " "),
    paste("Random Forest", auc.RF.logit, sep = " ")
),
lty = c(1, 2, 3, 4), bty = "n",
cex = .75
)
dev.off()
### The code to correct the Penalized Logistic regression figure in Figure 2 is shown below.
### ROC Plots for Penalized Logits and RF###
FL.2.pred <- predict(model.fl.2, newdata = data.test, type = "prob")
FL.2.pred <- as.data.frame(FL.2.pred)
CH.2.pred <- predict(model.ch.2, newdata = data.test, type = "prob")
CH.2.pred <- as.data.frame(CH.2.pred)
HS.2.pred <- predict(model.hs.2, type = "prob", newdata = data.test)
HS.2.pred <- as.data.frame(HS.2.pred)
# FL.2.pred <- 1 - FL.2.pred$war
# CH.2.pred <- 1 - CH.2.pred$war
# HS.2.pred <- 1 - HS.2.pred$war
FL.2.pred <- FL.2.pred$war
CH.2.pred <- CH.2.pred$war
HS.2.pred <- HS.2.pred$war
pred.FL.2 <- prediction(FL.2.pred, data.test$warstds)
perf.FL.2 <- performance(pred.FL.2, "tpr", "fpr")
pred.CH.2 <- prediction(CH.2.pred, data.test$warstds)
perf.CH.2 <- performance(pred.CH.2, "tpr", "fpr")
pred.HS.2 <- prediction(HS.2.pred, data.test$warstds)
perf.HS.2 <- performance(pred.HS.2, "tpr", "fpr")
auc.tmp <- performance(pred.FL.2, "auc")
auc.FL.plogit <- as.numeric(auc.tmp@y.values)
auc.tmp <- performance(pred.CH.2, "auc")
auc.CH.plogit <- as.numeric(auc.tmp@y.values)
auc.tmp <- performance(pred.HS.2, "auc")
auc.HS.plogit <- as.numeric(auc.tmp@y.values)
#### Plot corrected ROC Curves in Figure 1 for penalized logistic regression models.
jpeg("plot2.jpg")
plot(perf.FL.2, main = "Penalized Logits and Random Forests (Corrected)")
plot(perf.CH.2, add = T, lty = 2)
plot(perf.HS.2, add = T, lty = 3)
plot(perf.RF.1, add = T, lty = 4)
legend(0.32, 0.25, c(
    paste("Fearon and Laitin (2003)", auc.FL.plogit, sep = " "),
    paste("Collier and Hoeffler (2004)", auc.CH.plogit, sep = " "),
    paste("Hegre and Sambanis (2006)", auc.HS.plogit, sep = " "),
    paste("Random Forest", auc.RF.logit, sep = " ")
),
lty = c(1, 2, 3, 4), bty = "n",
cex = .75
)
dev.off()
### Combine both ROC plots
# jpeg("plot3.jpg")
par(mfrow = c(1, 2))
plot(perf.FL, main = "Logits and Random Forests (Corrected)")
plot(perf.CH, add = T, lty = 2)
plot(perf.HS, add = T, lty = 3)
plot(perf.RF.1, add = T, lty = 4)
legend(0.32, 0.25, c(
    paste("Fearon and Laitin (2003)", format(round(auc.FL.logit, 2), nsmall = 2), sep = " "),
    paste("Collier and Hoeffler (2004)", format(round(auc.CH.logit, 2), nsmall = 2), sep = " "),
    paste("Hegre and Sambanis (2006)", format(round(auc.HS.logit, 2), nsmall = 2), sep = " "),
    paste("Random Forest", format(round(auc.RF.logit, 2), nsmall = 2), sep = " ")
),
lty = c(1, 2, 3, 4), bty = "n",
cex = .75
)
plot(perf.FL.2, main = "Penalized Logits and Random Forests (Corrected)")
plot(perf.CH.2, add = T, lty = 2)
plot(perf.HS.2, add = T, lty = 3)
plot(perf.RF.1, add = T, lty = 4)
legend(0.32, 0.25, c(
    paste("Fearon and Laitin (2003)", format(round(auc.FL.plogit, 2), nsmall = 2), sep = " "),
    paste("Collier and Hoeffler (2004)", format(round(auc.CH.plogit, 2), nsmall = 2), sep = " "),
    paste("Hegre and Sambanis (2006)", format(round(auc.HS.plogit, 2), nsmall = 2), sep = " "),
    paste("Random Forest", format(round(auc.RF.logit, 2), nsmall = 2), sep = " ")
),
lty = c(1, 2, 3, 4), bty = "n",
cex = .75
)
# dev.off()
### Corrected code to draw corrected Separation Plots in Figure 2 as per Wang.
### Separation Plots###
library(separationplot)
## Transform DV back to 0,1 values for separation plots.
data.full$warstds <- factor(
    data.full$warstds,
    levels = c("peace", "war"),
    labels = c(0, 1)
)

data.test$warstds <- factor(
    data.test$warstds,
    levels = c("peace", "war"),
    labels = c(0, 1)
)

data.subset$warstds <- factor(
    data.subset$warstds,
    levels = c("peace", "war"),
    labels = c(0, 1)
)
# transform actual observations into vector for separation plots.
Warstds <- as.vector(data.test$warstds)
### Corrected Separation Plots###
# The corrections are the extraction of the fitted values for the logistic regression models
# from caret CV procedure.
separationplot(RF.1.pred$war, Warstds,
    type = "line", line = T, lwd2 = 1,
    show.expected = T,
    heading = "Random Forests", height = 2.5, col0 = "white", col1 = "black"
)
separationplot(pred.FL.war[[1]], Warstds,
    type = "line", line = T, lwd2 = 1, show.expected = T,
    heading = "Fearon and Laitin (2003)", height = 2.5, col0 = "white", col1 = "black"
)
separationplot(pred.CH.war[[1]], Warstds,
    type = "line", line = T, lwd2 = 1, show.expected = T,
    heading = "Collier and Hoeffler (2004)", height = 2.5, col0 = "white", col1 = "black"
)
separationplot(pred.HS.war[[1]], Warstds,
    type = "line", line = T, lwd2 = 1, show.expected = T,
    heading = "Hegre and Sambanis (2006)", height = 2.5, col0 = "white", col1 = "black"
)
### Plot Partial Dependence Plots###
######################################################################
######################################
# par(mfrow=c(3,3))
# partialPlot(RF.out.am, data2, gdpgrowth, which.class="1", xlab="GDP Growth Rate", main="",
# ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, ln_gdpen, ylim=c(-0.15, 0.15), which.class="1", xlab="GDP per Capita (log)",
# main="", ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, life, ylim=c(-0.15, 0.15), which.class="1", xlab="Life Expectancy",
# main="", ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, infant, ylim=c(-0.15, 0.15), which.class="1", xlab="Infant Mortality Rate",
# main="", ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, lmtnest, ylim=c(-0.15, 0.15), which.class="1", xlab <- "Mountainous Terrain (log)"
# , main="", ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, pol4sq, ylim=c(-0.15, 0.15), which.class="1", xlab="Polity IV Sq",
# main="", ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, lpopns, ylim=c(-0.15, 0.15), which.class="1", xlab="Population", main="",
# ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, trade, ylim=c(-0.15, 0.15), which.class="1", xlab="Trade", main="",
# xlim=c(0,200), ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
# partialPlot(RF.out.am, data2, geo1, ylim=c(-0.15, 0.15), which.class="1", xlab="W. Europe and U.S.",
# main="",ylab=expression(paste(Delta, "Fraction of Votes p(Y=1)")))
######################################################################
#######################################
# This section provides the correct code for the imputing of the out-of-sample data
# to replicate the new and corrected Table 1.
# Originally this section of code was uploaded in error, along with an incorrect dataset,
# see our response in the forum.
# This corrected code provides the means to impute the out-of-sample data and generate
# the out-of-sample predictions as per our response.
# The substance of our results do not change, though we are unable to replicate Table 1
# exactly due to loss of original code and data.
# Random Forest predicts more CW onsets in out-of-sample data than logistic regression.
# Seed for Imputation of out-of-sample data.
set.seed(425)
### Dataset for imputation.
data_imp <- read.csv(file = "data_full.csv")
# Imputation procedure.
# This is the imputation procedure we originally used to impute this data.
rf.imp <- rfImpute(data_imp, as.factor(data_imp$warstds), iter = 5, ntree = 1000)
### Out of Sample Data ###
# Subsetting imputed data.
mena <- subset(rf.imp, !(rf.imp$X...cowcode %in% vals))
### Generate out of sample predictions for Table 1 (corrected)
fl.pred <- predict(model.fl.1, newdata = mena, type = "prob")
fl.pred <- as.data.frame(fl.pred)
pred.FL.1 <- prediction(fl.pred$war, mena$`as.factor(data_imp$warstds)`)
perf.FL.1 <- performance(pred.FL.1, "auc")
ch.pred <- predict(model.ch.1, newdata = mena, type = "prob")
ch.pred <- as.data.frame(ch.pred)
pred.CH.1 <- prediction(ch.pred$war, mena$`as.factor(data_imp$warstds)`)
perf.CH.1 <- performance(pred.CH.1, "auc")
hs.pred <- predict(model.hs.1, newdata = mena, type = "prob")
hs.pred <- as.data.frame(hs.pred)
pred.HS.1 <- prediction(hs.pred$war, mena$`as.factor(data_imp$warstds)`)
perf.HS.1 <- performance(pred.HS.1, "auc")
rf.pred <- predict(model.rf, newdata = mena, type = "prob")
rf.pred <- as.data.frame(rf.pred)
pred.RF.1 <- prediction(rf.pred$war, mena$`as.factor(data_imp$warstds)`)
perf.RF.1 <- performance(pred.RF.1, "tpr", "fpr")
perf.RF.1 <- performance(pred.RF.1, "auc")
### Save Imputed Data. ###
predictions <- cbind(
    mena$cowcode, mena$year, mena$warstds, fl.pred[, 2], ch.pred[, 2],
    hs.pred[, 2], rf.pred[, 2]
)
### Write column headings for the out of sample data. ###
colnames(predictions) <- c(
    "COWcode", "year", "CW_Onset", "Fearon and Latin (2003)",
    "Collier and Hoeffler (2004)", "Hegre and Sambanis (2006)",
    "Random Forest"
)
### Save predictions as data frame for ordering the columns.
predictions <- as.data.frame(predictions)
### Table 1 Results, ordered by Onset (decreasing), and year (increasing) in R rather than excel.
Onset_table <- predictions[order(-predictions$CW_Onset, predictions$year), ]
### Rows 1-19 of the above go in Table 1. ###
Onset_table_1thru19 <- head(Onset_table, n = 19)
### Here's the code for Table 1 in Latex. ###
xtable(Onset_table_1thru19)
### Write the .csv file for all predictions to check against the Latex code for Table 1.
### Sort the csv same way as the Latex table - CW_Onset (decreasing), then by year (increasing).
write.csv(predictions, file = "subset_test.csv")