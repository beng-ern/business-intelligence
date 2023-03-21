library(data.table)
library(dplyr)
library(rpart)
library(grid)
#install.packages('partykit')
library(partykit) #draw decision tree
library(ggplot2)
library(caret) # one-hot encoding
library(ltm) #biserial correlation
library(MASS)
#install.packages('sampler')
library(sampler) #stratified sampling
#install.packages('e1071')
library(e1071) #CV, SVM
#install.packages('arules')
library(arules) #discretization

##########################################################
#################### (0) Loading Data ####################
##########################################################
#read the csv - strip leading and trailing white space - replace blank cells '' with NA
churn <- read.csv('Churn.csv', na.strings = c("", "NA"), strip.white = TRUE, stringsAsFactors = FALSE)

##########################################################
################# (1) Data Understanding #################
##########################################################
# dimension of train and test set
dim(churn)

# structure of train set
str(churn)

# summary of train set
summary(churn)

# number of NA values
sapply(churn, FUN = function(x) sum(is.na(x)))


#--------------------------------#
#====+ Categorical Variable +====#
#--------------------------------#

# cardinality
sapply(churn[, c('COLLEGE', 'REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL', 'CONSIDERING_CHANGE_OF_PLAN', 'LEAVE')], FUN = function(x) length(unique(x)))

# count of values
sort(table(churn$COLLEGE), decreasing = TRUE)
sort(table(churn$REPORTED_SATISFACTION), decreasing = TRUE)
sort(table(churn$REPORTED_USAGE_LEVEL), decreasing = TRUE)
sort(table(churn$CONSIDERING_CHANGE_OF_PLAN), decreasing = TRUE)
sort(table(churn$LEAVE), decreasing = TRUE)

# count of values based on churn status
table(churn$COLLEGE, churn$LEAVE)
table(churn$REPORTED_SATISFACTION, churn$LEAVE)
table(churn$REPORTED_USAGE_LEVEL, churn$LEAVE)
table(churn$CONSIDERING_CHANGE_OF_PLAN, churn$LEAVE)


#--------------------------------#
#=====+ Numerical Variable +=====#
#--------------------------------#
# five number summary
sapply(churn %>% dplyr::select(where(is.numeric)), FUN = function(x) summary(x, digits = 4))

# determine outliers
FindOutliers <- function(data) {
  lowerq = quantile(data, na.rm=TRUE)[2]
  upperq = quantile(data, na.rm=TRUE)[4]
  iqr = upperq - lowerq
  extreme.threshold.upper = (iqr * 1.5) + upperq
  extreme.threshold.lower = lowerq - (iqr * 1.5)
  result <- which(data > extreme.threshold.upper | data < extreme.threshold.lower)
}

sapply(churn %>% dplyr::select(where(is.numeric)), FUN = FindOutliers)

for(i in colnames(churn %>% dplyr::select(where(is.numeric)))){
  boxplot(churn[, i])
  title(paste("Boxplot of", i))
}

for(i in colnames(churn %>% dplyr::select(where(is.numeric)))){
  hist(churn[, i], main = paste('Histogram of', i), xlab = i)
}

##########################################################
################# (2) Data Preprocessing #################
##########################################################

# Variable COLLEGE: convert zero to 0, one to 1
churn$COLLEGE <- ifelse(tolower(churn$COLLEGE) == 'one',1,0)


# Variable REPORTED_SATISFACTION: convert to ordinal scale
churn[(churn$REPORTED_SATISFACTION) == 'very_sat', 'REPORTED_SATISFACTION'] <- 5
churn[churn$REPORTED_SATISFACTION == 'sat', 'REPORTED_SATISFACTION'] <- 4
churn[churn$REPORTED_SATISFACTION == 'avg', 'REPORTED_SATISFACTION'] <- 3
churn[churn$REPORTED_SATISFACTION == 'unsat', 'REPORTED_SATISFACTION'] <- 2
churn[churn$REPORTED_SATISFACTION == 'very_unsat', 'REPORTED_SATISFACTION'] <- 1


# Variable REPORTED_USAGE_LEVEL: convert to ordinal scale
churn[churn$REPORTED_USAGE_LEVEL == 'very_high', 'REPORTED_USAGE_LEVEL'] <- 5
churn[churn$REPORTED_USAGE_LEVEL == 'high', 'REPORTED_USAGE_LEVEL'] <- 4
churn[churn$REPORTED_USAGE_LEVEL == 'avg', 'REPORTED_USAGE_LEVEL'] <- 3
churn[churn$REPORTED_USAGE_LEVEL == 'little', 'REPORTED_USAGE_LEVEL'] <- 2
churn[churn$REPORTED_USAGE_LEVEL == 'very_little', 'REPORTED_USAGE_LEVEL'] <- 1


# Variable CONSIDERING_CHANGE_OF_PLAN: convert to ordinal scale
churn[churn$CONSIDERING_CHANGE_OF_PLAN == 'actively_looking_into_it', 'CONSIDERING_CHANGE_OF_PLAN'] <- 5
churn[churn$CONSIDERING_CHANGE_OF_PLAN == 'considering', 'CONSIDERING_CHANGE_OF_PLAN'] <- 4
churn[churn$CONSIDERING_CHANGE_OF_PLAN == 'perhaps', 'CONSIDERING_CHANGE_OF_PLAN'] <- 3
churn[churn$CONSIDERING_CHANGE_OF_PLAN == 'no', 'CONSIDERING_CHANGE_OF_PLAN'] <- 2
churn[churn$CONSIDERING_CHANGE_OF_PLAN == 'never_thought', 'CONSIDERING_CHANGE_OF_PLAN'] <- 1


# Variable LEAVE: convery to binary - LEAVE = 1, STAY = 0
churn$LEAVE <- ifelse(tolower(churn$LEAVE) == 'leave',1,0)


# Variable OVERAGE: the only case with -2 overcharge to be replaced by 0
churn[churn$OVERAGE < 0, 'OVERAGE'] <- as.integer(0)

# convert to integer for correlation checking
churn$CONSIDERING_CHANGE_OF_PLAN <- as.integer(churn$CONSIDERING_CHANGE_OF_PLAN)
churn$REPORTED_USAGE_LEVEL <- as.integer(churn$REPORTED_USAGE_LEVEL)
churn$REPORTED_SATISFACTION <- as.integer(churn$REPORTED_SATISFACTION)


# checking correlation
cor(churn)

# OVERAGE and OVER_15MINS_CALLS_PER_MONTH are highly correlated = 0.77
# LEFTOVER and AVERAGE_CALL_DURATION are moderately correlated = - 0.66
# HANDSET_PRICE and INCOME is highly correlated = 0.73


# convert back to factor
churn$CONSIDERING_CHANGE_OF_PLAN <- as.factor(churn$CONSIDERING_CHANGE_OF_PLAN)
churn$REPORTED_USAGE_LEVEL <- as.factor(churn$REPORTED_USAGE_LEVEL)
churn$REPORTED_SATISFACTION <- as.factor(churn$REPORTED_SATISFACTION)
churn$COLLEGE <- as.factor(churn$COLLEGE)
churn$LEAVE <- as.factor(churn$LEAVE)

str(churn)

# create customer ID for stratified sampling later
churn$CID <- 1:nrow(churn)


#============================#
##++ Train/Test Set Split ++##
#============================#
set.seed(4)
churn_train <- ssamp(churn, 20000*0.8, LEAVE, over = 0)
churn_test <- churn[!churn$CID %in% churn_train$CID, ]

# drop the CID column
churn_train$CID <- NULL
churn_test$CID <- NULL

str(churn_train)
str(churn_test)

################################################################
################# (3) Q1 - Decision Tree Model #################
################################################################
# using only default parameters
dcfit1 <- rpart(LEAVE~., data = churn_train, method = 'class')
plot(as.party(dcfit1), tp_args=list(id=FALSE))
dc_pred1 <- predict(dcfit1, churn_test, na.action = na.omit, type='class')
# accuracy
mean(dc_pred1 == churn_test[, 'LEAVE'])


################################################################
########### (4) Q2 - Alternative Decision Tree Model ###########
################################################################

dcfit2 <- rpart(LEAVE~., data = churn_train, method = 'class', control = rpart.control(cp = 0.05))
dc_pred2 <- predict(dcfit2, churn_test, na.action = na.omit, type='class')
# accuracy
mean(dc_pred2 == churn_test[, 'LEAVE'])

dcfit3 <- rpart(LEAVE~., data = churn_train, method = 'class', control = rpart.control(cp = 0.005))
dc_pred3 <- predict(dcfit3, churn_test, na.action = na.omit, type='class')
# accuracy
mean(dc_pred3 == churn_test[, 'LEAVE'])


dcfit3.1 <- rpart(LEAVE~., data = churn_train, method = 'class', control = rpart.control(cp = 0.005, minsplit=500))
dc_pred3.1 <- predict(dcfit3.1, churn_test, na.action = na.omit, type='class')
# accuracy
mean(dc_pred3.1 == churn_test[, 'LEAVE'])

dcfit3.2 <- rpart(LEAVE~., data = churn_train, method = 'class', control = rpart.control(cp = 0.005, minsplit=2000))
dc_pred3.2 <- predict(dcfit3.2, churn_test, na.action = na.omit, type='class')
# accuracy
mean(dc_pred3.2 == churn_test[, 'LEAVE'])

dcfit4 <- rpart(LEAVE~., data = churn_train, method = 'class', control = rpart.control(cp = 0.001))
dc_pred4 <- predict(dcfit4, churn_test, na.action = na.omit, type='class')
# accuracy
mean(dc_pred4 == churn_test[, 'LEAVE'])

dcfit5 <- rpart(LEAVE~., data = churn_train, method = 'class', control = rpart.control(cp = 0.001, maxdepth=5))
dc_pred5 <- predict(dcfit5, churn_test, na.action = na.omit, type='class')
# accuracy
mean(dc_pred5 == churn_test[, 'LEAVE'])


dcfit6 <- rpart(LEAVE~., data = churn_train, method = 'class', control = rpart.control(cp = 0.001, minsplit = 500))
dc_pred6 <- predict(dcfit6, churn_test, na.action = na.omit, type='class')
# accuracy
mean(dc_pred6 == churn_test[, 'LEAVE'])

dcfit6.1 <- rpart(LEAVE~., data = churn_train, method = 'class', control = rpart.control(cp = 0.001,minsplit = 1000))
dc_pred6.1 <- predict(dcfit6.1, churn_test, na.action = na.omit, type='class')
# accuracy
mean(dc_pred6.1 == churn_test[, 'LEAVE'])
plot(as.party(dcfit6.1), tp_args=list(id=FALSE))

dcfit7 <- rpart(LEAVE~., data = churn_train, method = 'class', control = rpart.control(cp = 0.0005))
dc_pred7 <- predict(dcfit7, churn_test, na.action = na.omit, type='class')
# accuracy
mean(dc_pred7 == churn_test[, 'LEAVE'])

###############################################################
############## (5) Q3 - Logistic Regresion Model ##############
###############################################################

# Fit the full model 
full.model <- glm(LEAVE ~ ., data = churn_train, family='binomial')
full_glm_pred <- predict(full.model, churn_test, na.action = na.omit, type = 'response')
full_glm_pred <- ifelse(full_glm_pred > 0.5,1,0)
misClasificError0 <- mean(full_glm_pred != churn_test[, 'LEAVE'])
print(paste('Accuracy =',1-misClasificError0))

# Stepwise regression model
step.model <- stepAIC(full.model, direction = "both", 
                      trace = FALSE)
summary(step.model)

glm_pred <- predict(step.model, churn_test, na.action = na.omit, type = 'response')
glm_pred <- ifelse(glm_pred > 0.5,1,0)
misClasificError <- mean(glm_pred != churn_test[, 'LEAVE'])
print(paste('Accuracy =',1-misClasificError))


# k-fold cross validation
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 20)

# WITHOUT ANY interaction effect
cv.model1 <- train(LEAVE ~ INCOME+OVERAGE+LEFTOVER+HOUSE+OVER_15MINS_CALLS_PER_MONTH+AVERAGE_CALL_DURATION+HANDSET_PRICE,data=churn_train, method="glm", family="binomial",trControl = ctrl)
cv_glm_pred1 <- predict(cv.model1, churn_test, na.action = na.omit)
# accuracy
mean(cv_glm_pred1 == churn_test[, 'LEAVE'])


# WITH interaction effect
cv.model2 <- train(LEAVE ~ INCOME+OVERAGE+LEFTOVER+HOUSE+OVER_15MINS_CALLS_PER_MONTH+AVERAGE_CALL_DURATION+HANDSET_PRICE+OVERAGE:OVER_15MINS_CALLS_PER_MONTH+LEFTOVER:AVERAGE_CALL_DURATION+HANDSET_PRICE:INCOME,
                   data=churn_train, method="glm", family="binomial",
                   trControl = ctrl)
cv_glm_pred2 <- predict(cv.model2, churn_test, na.action = na.omit)
# accuracy
mean(cv_glm_pred2 == churn_test[, 'LEAVE'])

# WITH interaction effect, WITHOUT INCOME AND HANDSET_PRICE as main effect
cv.model3 <- train(LEAVE ~ OVERAGE+LEFTOVER+HOUSE+OVER_15MINS_CALLS_PER_MONTH+AVERAGE_CALL_DURATION+OVERAGE:OVER_15MINS_CALLS_PER_MONTH+LEFTOVER:AVERAGE_CALL_DURATION+HANDSET_PRICE:INCOME,
                   data=churn_train, method="glm", family="binomial",
                   trControl = ctrl)
cv_glm_pred3 <- predict(cv.model3, churn_test, na.action = na.omit)
# accuracy
mean(cv_glm_pred3 == churn_test[, 'LEAVE'])

# WITH interaction effect, WITHOUT INCOME AND HANDSET_PRICE as main effect
# remove highly correlated variables
cv.model4 <- train(LEAVE ~ OVERAGE+LEFTOVER+HOUSE+OVERAGE:OVER_15MINS_CALLS_PER_MONTH+LEFTOVER:AVERAGE_CALL_DURATION+HANDSET_PRICE:INCOME,
                   data=churn_train, method="glm", family="binomial",
                   trControl = ctrl)
cv_glm_pred4 <- predict(cv.model4, churn_test, na.action = na.omit)
# accuracy
mean(cv_glm_pred4 == churn_test[, 'LEAVE'])

# WITHOUT interaction effect
# remove highly correlated variables
cv.model5 <- train(LEAVE ~ INCOME+OVERAGE+LEFTOVER+HOUSE,
                   data=churn_train, method="glm", family="binomial",
                   trControl = ctrl)
cv_glm_pred5 <- predict(cv.model5, churn_test, na.action = na.omit)
# accuracy
mean(cv_glm_pred5 == churn_test[, 'LEAVE'])



################################################
############## (6) Q4 - SVM Model ##############
################################################
# using ALL variables
svm.model1 <- svm(LEAVE~., data = churn_train)
svm_pred1 <- predict(svm.model1, churn_test)
# accuracy
mean(svm_pred1 == churn_test[, 'LEAVE'])


# using variables same as stepwise regression model
svm.model2 <- svm(LEAVE~INCOME+OVERAGE+LEFTOVER+HOUSE+OVER_15MINS_CALLS_PER_MONTH+AVERAGE_CALL_DURATION+HANDSET_PRICE, data = churn_train)
svm_pred2 <- predict(svm.model2, churn_test)
# accuracy
mean(svm_pred2 == churn_test[, 'LEAVE'])


# using variables same as stepwise regression model
# remove redundant variables (moderately and highly correlated)
svm.model3 <- svm(LEAVE~INCOME+OVERAGE+LEFTOVER+HOUSE, data = churn_train)
svm_pred3 <- predict(svm.model3, churn_test)
# accuracy
mean(svm_pred3 == churn_test[, 'LEAVE'])

# remove only handset price
svm.model4 <- svm(LEAVE~INCOME+OVERAGE+LEFTOVER+HOUSE+OVER_15MINS_CALLS_PER_MONTH+AVERAGE_CALL_DURATION, data = churn_train)
svm_pred4 <- predict(svm.model4, churn_test)
# accuracy
mean(svm_pred4 == churn_test[, 'LEAVE'])

# remove only Average call duration and handset price
svm.model4 <- svm(LEAVE~INCOME+OVERAGE+LEFTOVER+HOUSE+OVER_15MINS_CALLS_PER_MONTH, data = churn_train)
svm_pred4 <- predict(svm.model4, churn_test)
# accuracy
mean(svm_pred4 == churn_test[, 'LEAVE'])


#for tuning the models with best performance and least variables above
#x <- data.frame(subset(churn_train, select=c('OVERAGE', 'LEFTOVER', 'HOUSE', 'INCOME')))
#y <- churn_train$LEAVE

#svm_tune <- tune(svm, train.x=x, train.y=y, 
#                kernel="radial", ranges=list(cost=8^(-1:2), gamma=c(.5,1,2)))

#summary(svm_tune)
#print(svm_tune)

# model with tuned parameters
svm.model.tuned <- svm(LEAVE~INCOME+OVERAGE+LEFTOVER+HOUSE, data = churn_train, kernel='radial', cost=1, gamma=2)
svm_pred_tuned <- predict(svm.model.tuned, churn_test)
# accuracy
mean(svm_pred_tuned == churn_test[, 'LEAVE'])



#____________________________________________________________________________________________#

###################################################################################
############ (etc.) Q3 - Logistic Regresion Model with DISCRETIZATION #############
###################################################################################
# Discretization , method = 'interval', breaks = 4
churn$OVERAGE <- discretize(churn$OVERAGE)
churn$INCOME <- discretize(churn$INCOME)
churn$HOUSE <- discretize(churn$HOUSE)
churn$HANDSET_PRICE <- discretize(churn$HANDSET_PRICE)
churn$OVER_15MINS_CALLS_PER_MONTH <- discretize(churn$OVER_15MINS_CALLS_PER_MONTH)
churn$AVERAGE_CALL_DURATION <- discretize(churn$AVERAGE_CALL_DURATION)
churn$LEFTOVER <- discretize(churn$LEFTOVER)

#============================#
##++ Train/Test Set Split ++##
#============================#
set.seed(4)
churn_train <- ssamp(churn, 20000*0.8, LEAVE, over = 0)
churn_test <- churn[!churn$CID %in% churn_train$CID, ]

# drop the CID column
churn_train$CID <- NULL
churn_test$CID <- NULL

# rerun with only 4 variables
full.model2 <- glm(LEAVE ~ INCOME+OVERAGE+LEFTOVER+HOUSE, data = churn_train, family='binomial')

# Stepwise regression model
step.model2 <- stepAIC(full.model2, direction = "both", 
                       trace = FALSE)
summary(step.model2)
glm_pred2 <- predict(step.model2, churn_test, na.action = na.omit, type = 'response')
glm_pred2 <- ifelse(glm_pred2 > 0.5,1,0)
misClasificError2 <- mean(glm_pred2 != churn_test[, 'LEAVE'])
print(paste('Accuracy =',1-misClasificError2))
