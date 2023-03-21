# import the dataset without doing data preprocessing
AT_df <- read.csv('AT.csv')

#### (Q2) Data Quality Report for Age, Income, Occupation, and Churn ####
### (a) Age, Income
# count of elements
count_cont <- sapply(AT_df[, c('age', 'income')], FUN = length)
# % of missing values
miss_cont <- sapply(AT_df[, c('age', 'income')], FUN = function(x) mean(is.na(x))*100)
# cardinality
card_cont <- sapply(AT_df[, c('age', 'income')], FUN = function(x) length(unique(x)))
# min, max, 1st quartile, median, 3rd quartile, mean
fivenum_cont <- sapply(AT_df[, c('age', 'income')], FUN = function(x) summary(x, digits = 4))
# standard deviation
sd_cont <- sapply(AT_df[, c('age', 'income')], FUN = function(x) signif(sd(x),digits=4))

# put into a dataframa
data.frame(Count=count_cont, `% Miss.` = miss_cont, `Card.` = card_cont, t(fivenum_cont),
           `Std.Dev.`= sd_cont, check.names = FALSE)


### (a) categorical variables
# count of elements
count_cat <- sapply(AT_df[, c('occupation', 'churn')], FUN = length)
# percentage of missing values
miss_cat <- sapply(AT_df[, c('occupation', 'churn')], FUN = function(x) mean(is.na(x))*100)
# cardinality
card_cat <- sapply(AT_df[, c('occupation', 'churn')], FUN = function(x) length(unique(x)))
# sort count of values to get mode and 2nd mode
sort_occ <- sort(table(AT_df$occupation), decreasing = TRUE)
sort_churn <- sort(table(AT_df$churn), decreasing = TRUE)

# put into a dataframe
data.frame(Count=count_cat, `% Miss.` = miss_cat, `Card.` = card_cat, 
          Mode = c(names(sort_occ[1]), names(sort_churn[1])),
          'Mode Freq' = c(as.numeric(sort_occ[1]), as.numeric(sort_churn[1])),
          `Mode %` = c(as.numeric(sort_occ[1])/sum(sort_occ)*100, as.numeric(sort_churn[1])/sum(sort_churn)*100),
          '2nd Mode' = c(names(sort_occ[2]), names(sort_churn[2])),
          '2nd Mode Freq' = c(as.numeric(sort_occ[2]), as.numeric(sort_churn[2])),
          `2nd Mode %` = c(as.numeric(sort_occ[2])/sum(sort_occ)*100, as.numeric(sort_churn[2])/sum(sort_churn)*100),
          check.names = FALSE)

#### (Q3) Finding out specific problems in the dataset####

### Handling empty/blank/whitespace values during import###
# re-import the dataset
# strip leading and trailing white spaces
# replace blank cells '' with NA
AT_df <- read.csv('AT.csv', na.strings = c("", "NA"), strip.white = TRUE)
str(AT_df)

# find out variables with missing values
sapply(AT_df, FUN = function(x) mean(is.na(x)))

#######################################################
####### THE SECTION BELOW IS FOR 'AGE' VARIABLE #######
# take a deeper look into 'Age'
summary(AT_df$age)
ggplot(data=AT_df, aes(x=age)) + geom_bar() + theme_light()
table(AT_df$age)

# determining the type of missing values
library(data.table)
AT_df_miss<- copy(AT_df)
AT_df_miss$age_bool <- sapply(AT_df$age, FUN = function(x) if(x==0) x='no' else x='yes')
AT_df_miss <- subset(AT_df_miss, select=-c(age, churn, customer))

library(rpart)
fit <- rpart(age_bool~., data = AT_df_miss, method = 'class')

library(grid)
library(partykit)
plot(as.party(fit), tp_args=list(id=FALSE))

# converting Income level
AT_df_miss$income <- sapply(AT_df_miss$income, FUN = function(x) if(x==0) x='no' else x='yes')
fit <- rpart(age_bool~., data = AT_df_miss, method = 'class')
plot(as.party(fit), tp_args=list(id=FALSE))

# checking co-existence relationship of Age, Income and Marriage Status variables
AT_df_age<- subset(AT_df, select = c(age, income, marriageStatus))
AT_df_age$age <- sapply(AT_df_age$age, FUN = function(x) if(x==0) x=0 else x=1)
AT_df_age$income <- sapply(AT_df_age$income, FUN = function(x) if(x==0) x=0 else x=1)
AT_df_age$marriageStatus <- sapply(AT_df_age$marriageStatus, FUN = function(x) if(x=='unknown') x=0 else x=1)
sapply(AT_df_age[,c('income', 'marriageStatus')], FUN = function(x) cor(x, AT_df_age$age, use='pairwise.complete.obs'))
####### THE SECTION ABOVE IS FOR 'AGE' VARIABLE #######
#######################################################

# select only categorical variables from the dataset
AT_df_cat <- AT_df[,sapply(AT_df, is.factor)]
summary(AT_df_cat)
unique(AT_df$occupation)
unique(AT_df$regionType)
ggplot(data=AT_df, aes(x=occupation,fill=churn)) + geom_bar()

# convert credit card variable
AT_df$creditCard <- sapply(AT_df$creditCard,
                           FUN = function(x) if(x %in% c('t', 'yes', 'true')) x=1
                           else x=0)

# convert RegionType variable
AT_df$regionType <- sapply(AT_df$regionType,
                           FUN = function(x) if(x %in% c('r', 'rural')) x='rural'
                           else if(x %in% c('s', 'suburban')) x='suburban'
                           else if(x %in% c('t', 'town')) x='town' else x='unknown')
dim(AT_df_cat[AT_df_cat$regionType %in% c('unknown', NA),])[1]

library(ggplot2)
ggplot(data=AT_df, aes(x=regionType,fill=churn)) + geom_bar()


# remove Occupation and Region Type
AT_df <- subset(AT_df, select=-c(occupation, regionType))


# select only numerical variables
AT_df_cont <- AT_df[,sapply(AT_df, is.numeric)]
AT_df_cont <- subset(AT_df_cont, select=-c(income, creditCard, customer))

# check if any column contains negative values
names(AT_df_cont[,sapply(AT_df_cont, function(x) any(x < 0))])
dim(AT_df_cont[AT_df_cont$handsetAge < 0,])[1]
summary(AT_df[AT_df$handsetAge > 0, 'handsetAge'])

# remove records with negative handset age
AT_df <- AT_df[AT_df$handsetAge >= 0,]

#barplot for categorical variables
ggplot(data=AT_df, aes(x=marriageStatus,fill=churn)) + geom_bar()
ggplot(data=AT_df, aes(x=creditCard,fill=churn)) + geom_bar()
ggplot(data=AT_df, aes(x=smartPhone,fill=churn)) + geom_bar()
ggplot(data=AT_df, aes(x=homeOwner,fill=churn)) + geom_bar()
ggplot(data=AT_df, aes(x=creditRating,fill=churn)) + geom_bar()
ggplot(data=AT_df, aes(x=children,fill=churn)) + geom_bar()


# convert categorical variables to numerical variable
AT_df$marriageStatus <- sapply(AT_df$marriageStatus, FUN = function(x) if(x=='yes') x=1 else if(x=='no') x=0 else x=NA)
AT_df$children <- sapply(AT_df$children, FUN = function(x) if(x=='true') x=1 else x=0)
AT_df$smartPhone <- sapply(AT_df$smartPhone, FUN = function(x) if(x=='true') x=1 else x=0)
AT_df$homeOwner <- sapply(AT_df$homeOwner, FUN = function(x) if(x=='true') x=1 else x=0)


for(i in 1:7){
  AT_df$creditRating <- sapply(AT_df$creditRating, FUN = function(x) if(x==LETTERS[i]) x=i else x=x)
}

AT_df[AT_df$income==0, 'income'] <- NA
AT_df[AT_df$age==0, 'age'] <- NA

ggplot(data=AT_df, aes(x=income,fill=churn)) + geom_bar()

# observe outliers using boxplot
for(i in names(AT_df_cont)){
  print(ggplot(AT_df, aes_string(y=i, x="churn")) +
    geom_boxplot() + ggtitle(paste('Boxplot of', i))
    )
}


FindOutliers <- function(data) {
  lowerq = quantile(data, na.rm=TRUE)[2]
  upperq = quantile(data, na.rm=TRUE)[4]
  iqr = upperq - lowerq
  extreme.threshold.upper = (iqr * 1.5) + upperq
  extreme.threshold.lower = lowerq - (iqr * 1.5)
  result <- which(data > extreme.threshold.upper | data < extreme.threshold.lower)
}


for(i in names(AT_df_cont)){
  print(paste(i, 'variable :', length(FindOutliers(AT_df[,i])), 'outliers'))
}

# correlation between each numerical variable
cor(AT_df_cont, use = 'pairwise.complete.obs')


ggplot(data=AT_df, aes(y=billAmountChangePct, x=callMinutesChangePct)) +
  geom_point(aes(color=churn))




# write.csv(AT_df, file='AT2_20204887.csv', row.names = FALSE)


##### Q4 #####
# correlation
library(ltm)
as.matrix(tail(sort(abs((sapply(subset(AT_df,select=-c(churn)), FUN = function(x) biserial.cor(x, AT_df$churn, use = 'complete.obs')))))))


plot(AT_df[, c('handsetAge','smartPhone', 'avgrecurringCharge', 'avgMins', 'numRetentionCalls')])


for(i in c('handsetAge','smartPhone', 'avgrecurringCharge', 'avgMins', 'numRetentionCalls')){
  print(ggplot(AT_df, aes_string(y=i, x="churn")) +
          geom_point() + ggtitle(paste('Scatterplot of', i))
  )
}