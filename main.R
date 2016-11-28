## Nibu Jacob, MS in Data Science, Indiana University Bloomington
## Date: Nov 2016
## To prepare European Social Survey data for ML algorithms
## To run multiple algorithms on the data and make evaluations

# setwd('C:/Users/nibu/Google Drive/data science/Sriram/project/ESS_income_prediction')


# load SPSS format data file downloaded from ESS website
# --------------------------------------------------------
require(haven)
data_raw <- read_spss("ESS6MDWe02.2_F1.sav")  # original raw data


# take a working copy of the data and check the size
# --------------------------------------------------------
data <- data_raw         # working copy of the data
dim(data)                # 1847 obs. of 32 variables


# remove irrelevant variables
# --------------------------------------------------------
data$ESS6_id  <- NULL
data$cntry    <- NULL
data$ESS6_reg <- NULL
data$NUTS1    <- NULL
data$NUTS2    <- NULL
data$NUTS3    <- NULL
data$dweight  <- NULL
data$pspwght  <- NULL
data$pweight  <- NULL

dim(data)                # 1847 obs. of 23 variables


# rename the variables for clarity
# --------------------------------------------------------
names(data)
names(data)[names(data) == 'agea'] <- 'age'
names(data)[names(data) == 'hinctnta'] <- 'income'


# examine the variables for missing values and normalization
# --------------------------------------------------------
summary(data)            # all columns have NA values; age needs normalization
data_withNA <- data
data <- data[complete.cases(data),]  # 1613 obs.


# include only people of age 30 years or more
# --------------------------------------------------------
data <- data[(data$age >= 30),]
nrow(data)               # 1300 obs.


# normalize the age variable; 6 levels for most other variables
# --------------------------------------------------------
table(data$age)          # only few people above 90; therefore can cluster 80s and 90s into one
data$age <- replace(data$age, data$age < 40, 1) 
data$age <- replace(data$age, data$age >= 40 & data$age < 50, 2) 
data$age <- replace(data$age, data$age >= 50 & data$age < 60, 3) 
data$age <- replace(data$age, data$age >= 60 & data$age < 70, 4) 
data$age <- replace(data$age, data$age >= 70 & data$age < 80, 5) 
data$age <- replace(data$age, data$age >= 80, 6) 


# create a new binomial variable to represent income, based on median income level of 7
# --------------------------------------------------------
summary(data$income)
table(data$income)
data$income_bi[data$income < 7] <- 0
data$income_bi[data$income >= 7] <- 1
data$income <- NULL   # removing the multilevel factor income variable
names(data)[names(data) == 'income_bi'] <- 'income'
table(data$income)    # 605 obs. with 0, 695 obs. with 1; balanced representation of levels


# check the datatype of variables
# --------------------------------------------------------
sapply(data, class)
data[] <- lapply(data, factor)  # make all variables as factors


# check the class distribution
# --------------------------------------------------------
# split data inputs attributes as x and the output attribute (or class) as y
x <- data[,1:22]
y <- data[,23]

# check the class distribution
plot(y)  # the two levels are evenly distributed

# check interaction between attributes using a scatterplot matrix
# --------------------------------------------------------
# featurePlot(x=x, y=y, plot="ellipse")

# check and remove highly correlated features
# --------------------------------------------------------
findCorrelation(data, cutoff = 0.9, verbose = FALSE, names = FALSE)


# split the dataset into training and test sets
# --------------------------------------------------------
require(caTools)
set.seed(371)
split <- sample.split(data$income, SplitRatio = 0.75)
train <- subset(data, split == TRUE)
test <- subset(data, split == FALSE)



# ========================================================
# multiple ML algorithms using caret package
# ========================================================

install.packages("caret", dependencies = c("Depends", "Suggests"))
require(caret)

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# logistic regression
set.seed(371)
fit.glm <- train(income~., data=train, method="glm", metric=metric, trControl=control)

# CART
set.seed(371)
fit.cart <- train(income~., data=train, method="rpart", metric=metric, trControl=control)

# Random Forest
set.seed(371)
fit.rf <- train(income~., data=train, method="rf", metric=metric, trControl=control)

# kNN
set.seed(371)
fit.knn <- train(income~., data=train, method="knn", metric=metric, trControl=control)

# SVM
set.seed(371)
fit.svm <- train(income~., data=train, method="svmRadial", metric=metric, trControl=control)



# summarize accuracy of models
results <- resamples(list(glm=fit.glm, cart=fit.cart, rf=fit.rf, knn=fit.knn, svm=fit.svm))
summary(results)
dotplot(results)


# estimate performance of each model on the validation dataset
require(ROCR)

predictions <- predict(fit.glm, test)
confusionMatrix(predictions, test$income)
pred     <- as.numeric(predictions)
ROCRpred <- prediction(pred, test$income)
ROCRperf <- performance(ROCRpred, "tpr", "fpr") 
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
as.numeric(performance(ROCRpred, "auc")@y.values)

predictions <- predict(fit.cart, test)
confusionMatrix(predictions, test$income)
pred     <- as.numeric(predictions)
ROCRpred <- prediction(pred, test$income)
ROCRperf <- performance(ROCRpred, "tpr", "fpr") 
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
as.numeric(performance(ROCRpred, "auc")@y.values)


predictions <- predict(fit.rf, test)
confusionMatrix(predictions, test$income)
pred     <- as.numeric(predictions)
ROCRpred <- prediction(pred, test$income)
ROCRperf <- performance(ROCRpred, "tpr", "fpr") 
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
as.numeric(performance(ROCRpred, "auc")@y.values)


predictions <- predict(fit.knn, test)
confusionMatrix(predictions, test$income)
pred     <- as.numeric(predictions)
ROCRpred <- prediction(pred, test$income)
ROCRperf <- performance(ROCRpred, "tpr", "fpr") 
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
as.numeric(performance(ROCRpred, "auc")@y.values)


predictions <- predict(fit.svm, test)
confusionMatrix(predictions, test$income)
pred     <- as.numeric(predictions)
ROCRpred <- prediction(pred, test$income)
ROCRperf <- performance(ROCRpred, "tpr", "fpr") 
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
as.numeric(performance(ROCRpred, "auc")@y.values)


# plot accuracy on training and test sets for all 5 algorithms
require(ggplot2)
axis_values <- c(1,2,3,4,5)
algorithms <- c("glm","cart","rf","knn","svm")
acc_train <- c(0.6513, 0.6202, 0.6633, 0.6154, 0.6615)
acc_test <- c(0.6316, 0.6388, 0.6718, 0.6773, 0.7801)

plot(axis_values, acc_train, type="l")
line(axis_values, acc_test, col="blue", lty=2)

plotdata <- data.frame(algorithms=algorithms,acc_train=acc_train, acc_test=acc_test)

ggplot(plotdata, aes(x=algorithms))+
  geom_line(aes(y=acc_train, color="acc_train", group=1), size=2)+
  geom_line(aes(y=acc_test, color="acc_test", group=1), size=2)

