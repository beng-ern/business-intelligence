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
library(C50)
library(randomForest)
library(caTools)
library(leaps) # forward regression


########################################################################
################# (0) Data Preprocessing following HW7 #################
########################################################################
churn <- read.csv('Churn.csv', na.strings = c("", "NA"), strip.white = TRUE, stringsAsFactors = FALSE)

# Variable COLLEGE: convert zero to 0, one to 1
churn$COLLEGE <- ifelse(tolower(churn$COLLEGE) == 'one',1,0)

# Variable REPORTED_SATISFACTION: convert to ordinal scale
churn[(churn$REPORTED_SATISFACTION) == 'very_sat', 'REPORTED_SATISFACTION'] <- 5
churn[churn$REPORTED_SATISFACTION == 'sat', 'REPORTED_SATISFACTION'] <- 4
churn[churn$REPORTED_SATISFACTION == 'avg', 'REPORTED_SATISFACTION'] <- 3
churn[churn$REPORTED_SATISFACTION == 'unsat', 'REPORTED_SATISFACTION'] <- 2
churn[churn$REPORTED_SATISFACTION == 'very_unsat', 'REPORTED_SATISFACTION'] <- 1
churn$REPORTED_SATISFACTION <- as.integer(churn$REPORTED_SATISFACTION)

# Variable REPORTED_USAGE_LEVEL: convert to ordinal scale
churn[churn$REPORTED_USAGE_LEVEL == 'very_high', 'REPORTED_USAGE_LEVEL'] <- 5
churn[churn$REPORTED_USAGE_LEVEL == 'high', 'REPORTED_USAGE_LEVEL'] <- 4
churn[churn$REPORTED_USAGE_LEVEL == 'avg', 'REPORTED_USAGE_LEVEL'] <- 3
churn[churn$REPORTED_USAGE_LEVEL == 'little', 'REPORTED_USAGE_LEVEL'] <- 2
churn[churn$REPORTED_USAGE_LEVEL == 'very_little', 'REPORTED_USAGE_LEVEL'] <- 1
churn$REPORTED_USAGE_LEVEL <- as.integer(churn$REPORTED_USAGE_LEVEL)

# Variable CONSIDERING_CHANGE_OF_PLAN: convert to ordinal scale
churn[churn$CONSIDERING_CHANGE_OF_PLAN == 'actively_looking_into_it', 'CONSIDERING_CHANGE_OF_PLAN'] <- 5
churn[churn$CONSIDERING_CHANGE_OF_PLAN == 'considering', 'CONSIDERING_CHANGE_OF_PLAN'] <- 4
churn[churn$CONSIDERING_CHANGE_OF_PLAN == 'perhaps', 'CONSIDERING_CHANGE_OF_PLAN'] <- 3
churn[churn$CONSIDERING_CHANGE_OF_PLAN == 'no', 'CONSIDERING_CHANGE_OF_PLAN'] <- 2
churn[churn$CONSIDERING_CHANGE_OF_PLAN == 'never_thought', 'CONSIDERING_CHANGE_OF_PLAN'] <- 1
churn$CONSIDERING_CHANGE_OF_PLAN <- as.integer(churn$CONSIDERING_CHANGE_OF_PLAN)

# Variable LEAVE: convery to binary - LEAVE = 1, STAY = 0
churn$LEAVE <- ifelse(tolower(churn$LEAVE) == 'leave',1,0)
churn$LEAVE <- as.factor(churn$LEAVE)


# Variable OVERAGE: the only case with -2 overcharge to be replaced by 0
churn[churn$OVERAGE < 0, 'OVERAGE'] <- as.integer(0)


# create customer ID for stratified sampling later
churn$CID <- 1:nrow(churn)

#----- Train / Test Split -----#
set.seed(4)
churn_train <- ssamp(churn, 20000*0.8, LEAVE, over = 0)
churn_train <- data.frame(churn_train)
churn_test <- churn[!churn$CID %in% churn_train$CID, ]

# drop the CID column
churn_train$CID <- NULL
churn_test$CID <- NULL
churn$CID <- NULL

##################################################
################# (1) Question 1 #################
##################################################

#---------------------- Decision Tree Model ----------------------#

# choosing majority counts as baseline model
table(churn_train$LEAVE)
sort(table(churn_train$LEAVE) ,decreasing = TRUE)[1] / dim(churn_train)[1]
sort(table(churn_test$LEAVE) ,decreasing = TRUE) / dim(churn_test)[1]

# using single node to build baseline model
# using only HOUSE variable
dcfit_base <- C5.0(LEAVE~HOUSE, data = churn_train, 
                   control = C5.0Control(noGlobalPruning = FALSE, minCases=30))
base_pred <- predict(dcfit_base, newdata = churn_test) # prediction
print(paste('Accuracy =', mean(base_pred == churn_test$LEAVE))) # accuracy
# plot(as.party(dcfit_base), tp_args=list(id=FALSE))

# using ALL variables for build the tree
dcfit1 <- C5.0(LEAVE~., data = churn_train,  
              control = C5.0Control(noGlobalPruning = FALSE, minCases=30))
summary(dcfit1)
# plot(as.party(dcfit1), tp_args=list(id=FALSE))
dc1_pred <- predict(dcfit1, newdata = churn_test) # prediction
print(paste('Accuracy =', mean(dc1_pred == churn_test$LEAVE))) # accuracy



#-------------------- Logistic Regression Model --------------------#

# build logistic regression model using ALL variables 
full.glm <- glm(LEAVE ~ ., data = churn_train, family='binomial')
summary(full.glm)
full.glm_pred <- predict(full.glm, churn_test, na.action = na.omit, type = 'response')
full.glm_pred <- ifelse(full.glm_pred > 0.5,1,0)
print(paste('Accuracy =', mean(full.glm_pred == churn_test$LEAVE)))


#####------- Learning Curve Plotting -------#####
i = 16000
num_instance <- array()
acc_arr_dc <- array()
acc_arr_lr <- array()
j <- 1
while(i > 1) {
  set.seed(4)
  dc.fit <- C5.0(LEAVE~., data = slice_sample(churn_train, n = i),
              control = C5.0Control(noGlobalPruning = FALSE, minCases=30))
  dc.pred <- predict(dc.fit, newdata = churn_test)
  dc.acc <- mean(dc.pred == churn_test$LEAVE)
  
  glm_fit <- glm(LEAVE ~ ., data = slice_sample(churn_train, n = i), family='binomial')
  glm_pred <- predict(glm_fit, churn_test, na.action = na.omit, type = 'response')
  glm_pred <- ifelse(glm_pred > 0.5,1,0)
  glm.acc <- mean(glm_pred == churn_test$LEAVE)
  
  num_instance[j] <- i
  acc_arr_dc[j] <- dc.acc
  acc_arr_lr[j] <- glm.acc
  
  i = ceiling(i / 2)
  j = j + 1
}


learning_curve <- data.frame(num_instance, acc_arr_dc, acc_arr_lr)
learning_curve[nrow(learning_curve) + 1,] = c(0,0,0)
setorder(learning_curve)
rownames(learning_curve) <- NULL

ggplot(data=learning_curve) + 
  geom_line(aes(y=acc_arr_dc, x=num_instance, colour = "Decision Tree"), size=0.8) +
  geom_line(aes(y=acc_arr_lr, x=num_instance, colour = "Logistic Regression"), size=0.8) +
  scale_color_discrete(name = "Model") +
  xlab('Number of Training Instances') + 
  ylab('Accuracy')

# zoom in
ggplot(data=learning_curve[1:11,]) + 
  geom_line(aes(y=acc_arr_dc, x=num_instance, colour = "Decision Tree"), size=1) +
  geom_line(aes(y=acc_arr_lr, x=num_instance, colour = "Logistic Regression"), size=1) +
  scale_color_discrete(name = "Model") +
  xlab('Number of Training Instances') + 
  ylab('Accuracy') + geom_vline(xintercept = 250, colour = 'green')

#######################################################################
################# (3) Question 3 - Varying Complexity #################
#######################################################################
# decision tree model
# decrease minCases to increase complexity
train.dc.acc <- array()
test.dc.acc <- array()
node.size <- array()

k = 1
for(i in c(1:50, seq(60,100,10), seq(100,1000,100))){
  dc.fit <- C5.0(LEAVE~., data = churn_train,  
                 control = C5.0Control(noGlobalPruning=FALSE, minCases = i))
  node.size[k] <- dc.fit$size
  train_pred <- predict(dc.fit, newdata = churn_train)
  train.dc.acc[k] <- mean(train_pred == churn_train$LEAVE)
  test_pred <- predict(dc.fit, newdata = churn_test)
  test.dc.acc[k] <- mean(test_pred == churn_test$LEAVE)
  k = k + 1
}

fit_graph <- data.frame(c(1:50, seq(60,100,10), seq(100,1000,100)), node.size, train.dc.acc, test.dc.acc)
colnames(fit_graph) <- c("MinCases", "Node", 'Train', 'Test')

# num of min cases vs num of nodes
ggplot(data=fit_graph) +
  geom_line(aes(y=Node, x=MinCases), size=0.8, colour='darkblue') +
  xlab('Number of Min. Cases') +
  ylab('Number of Nodes')

fit_graph_group <- fit_graph %>% group_by(Node) %>% summarise_at(vars(Train, Test), list(name = mean))
fit_graph_group <- data.frame(fit_graph_group)
colnames(fit_graph_group) <- c("Node", 'Train', 'Test')

ggplot(data=fit_graph_group) + 
  geom_line(aes(y=Train, x=Node, colour = "Train Set"), size=0.8) +
  geom_line(aes(y=Test, x=Node, colour = "Test Set"), size=0.8) +
  scale_color_discrete(name = "") +
  xlab('Number of Nodes') + 
  ylab('Accuracy')


# logistic regression model
# SINGLE variable to ALL variables
# using regsubsets stepwise regression function to select the best set of attributes
models <- regsubsets(LEAVE~., data = churn_train, nvmax = 11,
                     method = "seqrep")
summary(models)

cols_reg <- c("OVERAGE", "HOUSE", "INCOME", "LEFTOVER", "AVERAGE_CALL_DURATION", "OVER_15MINS_CALLS_PER_MONTH",
              "HANDSET_PRICE", "CONSIDERING_CHANGE_OF_PLAN", "REPORTED_SATISFACTION", "REPORTED_USAGE_LEVEL", 
              "COLLEGE")
train.lr.acc <- array()
test.lr.acc <- array()

m = 1
for(i in cols_reg){
  if(i=="REPORTED_USAGE_LEVEL"){
    lr.formula <- as.formula(paste("LEAVE ~ ", paste(cols_reg[-8], collapse= "+")))
  } else {
    lr.formula <- as.formula(paste("LEAVE ~ ", paste(cols_reg[1:m], collapse= "+")))}
  glm.step <- glm(lr.formula, data = churn_train, family='binomial')
  
  train.pred <- predict(glm.step, churn_train, na.action = na.omit, type = 'response')
  train.pred <- ifelse(train.pred > 0.5,1,0)
  train.lr.acc[m] <- mean(train.pred == churn_train$LEAVE)
  
  test.pred <- predict(glm.step, churn_test, na.action = na.omit, type = 'response')
  test.pred <- ifelse(test.pred > 0.5,1,0)
  test.lr.acc[m] <- mean(test.pred == churn_test$LEAVE)
  m = m + 1
}

# WITH 3 interaction effects added
glm.step <- glm(LEAVE ~ OVERAGE + HOUSE + INCOME + LEFTOVER + AVERAGE_CALL_DURATION + 
                  OVER_15MINS_CALLS_PER_MONTH + HANDSET_PRICE + CONSIDERING_CHANGE_OF_PLAN + 
                  REPORTED_SATISFACTION + REPORTED_USAGE_LEVEL + COLLEGE + HANDSET_PRICE:INCOME +
                  LEFTOVER:AVERAGE_CALL_DURATION + OVERAGE:OVER_15MINS_CALLS_PER_MONTH,
                data = churn_train, family='binomial')

train.pred <- predict(glm.step, churn_train, na.action = na.omit, type = 'response')
train.pred <- ifelse(train.pred > 0.5,1,0)
train.lr.acc[m] <- mean(train.pred == churn_train$LEAVE)

test.pred <- predict(glm.step, churn_test, na.action = na.omit, type = 'response')
test.pred <- ifelse(test.pred > 0.5,1,0)
test.lr.acc[m] <- mean(test.pred == churn_test$LEAVE)

lr.acc.df <- data.frame(c(1:11, 14),test.lr.acc, train.lr.acc)
colnames(lr.acc.df) <- c("Predictors", "Test", "Train")


ggplot(data=lr.acc.df) + 
  geom_line(aes(y=Train, x=Predictors, colour = "Train Set"), size=0.8) +
  geom_line(aes(y=Test, x=Predictors, colour = "Test Set"), size=0.8) +
  scale_color_discrete(name = "") +
  xlab('Number of Predictors') + 
  ylab('Accuracy')

#####################################################################
################# (4) Question 4 - Cross Validation #################
#####################################################################
set.seed(4)
# 10-fold cross validation
ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE)

# decision tree
set.seed(4)
cv.dc.model <- train(LEAVE ~ .,data=churn, method="C5.0Tree", trControl = ctrl)
summary(cv.dc.model)
print(paste("Accuracy =", cv.dc.model$results[2]))


# check each fold accuracy
cv.dc.pred <- cv.dc.model$pred
cv.dc.pred$equal <- ifelse(cv.dc.pred$pred == cv.dc.pred$obs, 1,0)
eachfold_dc <- cv.dc.pred %>%
  group_by(Resample) %>%
  summarise_at(vars(equal),
               list(Accuracy = mean))
eachfold_dc <- data.frame(eachfold_dc)
colnames(eachfold_dc) <- c("Fold", "Accuracy")

ggplot(data = eachfold_dc) + 
  geom_bar(aes(x=Fold, y=Accuracy, fill=Fold), stat = "identity") + theme(legend.position = "none")

set.seed(4)
# logistic regression model
# WITHOUT ANY interaction effect
cv.glm.model <- train(LEAVE ~ .,data=churn, method="glm", family="binomial", trControl = ctrl)
print(paste("Accuracy =", cv.glm.model$results[2]))

# check each fold accuracy
cv.glm.pred <- cv.glm.model$pred
cv.glm.pred$equal <- ifelse(cv.glm.pred$pred == cv.glm.pred$obs, 1,0)
eachfold_lr <- cv.glm.pred %>%
  group_by(Resample) %>%
  summarise_at(vars(equal),
               list(Accuracy = mean))
eachfold_lr <- data.frame(eachfold_lr)
colnames(eachfold_lr) <- c("Fold", "Accuracy")

ggplot(data = eachfold_lr) + 
  geom_bar(aes(x=Fold, y=Accuracy, fill=Fold), stat = "identity") + theme(legend.position = "none") +
  scale_fill_brewer(palette = "Spectral")


######################################################################
################# (5) Question 5 - Ensemble Modeling #################
######################################################################
### RANDOM FOREST ###
# holdout sampling approach
set.seed(4)
rf <- randomForest(LEAVE ~ ., data=churn_train)
rf_pred <- predict(rf, newdata=churn_test)
print(paste("Accuracy =", mean(rf_pred == churn_test$LEAVE)))


# cross validation
ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE)
set.seed(4)
cv.rf <- train(LEAVE ~ .,data=churn, method="rf", trControl = ctrl)
print(paste("Accuracy =", cv.rf$results[2]))
cv.rf$results

### boosting C5.0 algorithm ###
# holdout sampling approach
set.seed(4)
c50.boost <- C5.0(LEAVE~., data = churn_train,  trials = 3,
               control = C5.0Control(noGlobalPruning=FALSE, minCases = 30))
boost_pred <- predict(c50.boost, newdata=churn_test)
print(paste("Accuracy =", mean(boost_pred == churn_test$LEAVE)))


###  simple ensemble based on majority votes ###
major_ens <- data.frame(boost_pred, rf_pred, dc1_pred)
major_ens$boost_pred <- as.integer(as.character(major_ens$boost_pred))
major_ens$rf_pred <- as.integer(as.character(major_ens$rf_pred))
major_ens$dc1_pred <- as.integer(as.character(major_ens$dc1_pred))
major_ens <- major_ens %>% mutate(maj = ifelse(rowSums(across(where(is.numeric))) > 1, 1, 0))
print(paste("Accuracy =", mean(major_ens$maj == churn_test$LEAVE)))