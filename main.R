## Nibu Jacob, MS in Data Science, Indiana University Bloomington
## Date: Nov 2016
## To prepare European Social Survey data for ML algorithms
## To run multiple algorithms on the data and make evaluations

setwd("Z:/R")


# load SPSS format data file downloaded from ESS website
# --------------------------------------------------------
library("haven")
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
table(data$income_bi)    # 605 obs. with 0, 695 obs. with 1; balanced representation of levels


# check the datatype of variables
# --------------------------------------------------------
summary(data)
data[] <- lapply(data, factor)  # make all variables as factors


# split the dataset into training and test sets
# --------------------------------------------------------
require(caTools)
set.seed(371)
split <- sample.split(data$income_bi, SplitRatio = 0.75)
train <- subset(data, split == TRUE)
test <- subset(data, split == FALSE)

train_bi <- subset(train, select = -income)
test_bi <- subset(test, select = -income)

train_fac <- subset(train, select = -income_bi)
test_fac <- subset(test, select = -income_bi)






