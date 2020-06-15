#install packages
if(!require(knitr)) install.packages(knitr) 
if(!require(tree)) install.packages(tree)
if(!require(dplyr)) install.packages(dplyr)
if(!require(stringr)) install.packages(stringr)
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(randomForest)) install.packages(randomForest)
if(!require(gbm)) install.packages(gbm)
if(!require(tree)) install.packages(tree)
if(!require(devtools)) install.packages("devtools")
if(!require(ggcorrplot)) install.packages(ggcorrplot)
#load libraries
library(knitr)
library(dplyr)
library(tidyr)
library(stringr)
library(tree)
library(randomForest)
library(ggplot2)
library(gbm)
library(devtools)
library(ggcorrplot)

#Get the data from the UCI website to download into R
temp <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00445/Absenteeism_at_work_AAA.zip",temp)
df <- read.csv(unz(temp, "Absenteeism_at_work.csv"))

Column Headers

#To address the first issue we notice that all of the field names start with a capital letter. After separating 
#every value based on where it is capitalized, we see that the length of the vector is 26. We want the length 
#value to be 21 since that is how many variables there. The reason why there are 26 elements is because some 
#fields use multiple capital letters. For example, “Distance from Residence to Work” got split up into three 
#fields since it is capitalized twice. So we have “Distance from” and “Residence to” and “work”.  We use the 
#paste function to concatenate those values. So all 3 of those values get replaced with “Distance from Residence
#to work”. So we still have 26 elements in the list, but we have the correct fields name this time. Using the 
#“unique” function will get rid of any duplicates headers and we have 21 columns. A similar approach is used for
#“ID” and “Workload Average day”. So that is how the first issue is addressed above. 

#Create column headers
col_headers<-names(df)
col_headers<-gsub('([[:upper:]])', ' \\1', col_headers)
col_headers<-strsplit(col_headers," ")[[1]]
col_headers[1:3]<-"ID"
col_headers[9:11]<-paste(col_headers[9],col_headers[10],col_headers[11],sep = "")
col_headers[14:15]<-paste(col_headers[14],col_headers[15],sep = "")
col_headers<-unique(col_headers)

#Rename column headers
obsv_list<-strsplit(df[,1],";")
df<-data.frame(t(sapply(obsv_list,c)))
as.data.frame(obsv_list)
names(df)<-col_headers

#Convert variables to factors and numeric
factor_var_col<-c(2:5,12,13,15,16)
df_factor<-data.frame(lapply(df[factor_var_col],as.factor))
df_numeric<-data.frame(lapply(df[-factor_var_col],as.numeric))
final_df<-cbind(df_factor,df_numeric)

#Plot histogram of absenteeism
hist(final_df$Absenteeism.time.in.hours, col="red", xlab="Absenteeism Time In Hours",ylab="Frequency",main="Distribution of Absenteeism")
#Plot Correlation Matrix
corr <- round(cor(df_numeric), 1)
png(height=1200, width=1500, pointsize=15, file="overlap.png")
ggcorrplot(corr, hc.order = TRUE, type = "lower",lab = TRUE, lab_size = 2,pch = 2, tl.cex = 7,tl.srt = 90)

#Model: Random Forest and gradient boosting

#Assign training/test split
set.seed(40, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(40)` instead
train_percent<-0.8
train <- sample(1:nrow(final_df),round(nrow(final_df)*train_percent))


#Create an empty vector that will contain the MSE for both models
number_of_trees<-seq(0,5000,100)
RF.MSE<-c()
Boost.MSE<-c()

#Run random forest and boost model. The number of trees wil increase by 100 each time and the mse will be calculated for each
#We can then find where the minimum MSE occurs in which model
for(i in 2:(length(number_of_trees)-1)){
  
  #Run both models using the training data
  RF.train<-randomForest(Absenteeism.time.in.hours~.,data=final_df[train,],mtry=5, ntree=number_of_trees[i])
  boost.train<-gbm(Absenteeism.time.in.hours~.,data=final_df[train,],distribution = "gaussian",n.trees =number_of_trees[i], interaction.depth = 4)
  
  #Calculate the prediction on both models using the test set
  RF.predict<-predict(RF.train,newdata =final_df[-train,],n.trees = number_of_trees[i] )
  Boost.predict<-predict(boost.train,newdata =final_df[-train,],n.trees = number_of_trees[i] )
  
  #calculate MSE AND assign it to the MSE vector
  actual_values<-final_df[-train,21]
  RF.MSE[i]<-round(mean((RF.predict-actual_values)^2))
  Boost.MSE[i]<-round(mean((Boost.predict-actual_values)^2))
}

#graph of trees and MSE value
ggplot() + geom_line(aes(x=number_of_trees[-1],y=RF.MSE, group =1),color='red') + 
  geom_line(aes(x=number_of_trees[-1],y=Boost.MSE, group = 2),color='blue') + 
  ylab('MSE')+xlab('Number of Trees')

#Determine where the minimum MSE occurs in each model
RF.min.mse.index<-which.min(RF.MSE)
Boost.min.mse.index<-which.min(Boost.MSE)

#Determine the actual min MSE for each algorithm
RF.min.mse<-RF.MSE[RF.min.mse.index]
Boost.min.mse<-Boost.MSE[Boost.min.mse.index]

#Optimal number of trees
opt.number.of.trees<-number_of_trees[RF.min.mse]

Based off of the results we see that the optimal algorithm is a Random Forest  with 3800 trees which has a MSE of 39.Let's see what the most important variables are for this model.


RF.train.optimal<-randomForest(Absenteeism.time.in.hours~.,data=final_df[train,],mtry=5, ntree=3800)
imp_variables<-importance(RF.train.optimal)
importance(RF.train.optimal)





Dataset owner:


Martiniano, A., Ferreira, R. P., Sassi, R. J., & Affonso, C. (2012). Application of a neuro fuzzy network in prediction of absenteeism at work. In Information Systems and Technologies (CISTI), 7th Iberian Conference on (pp. 1-4). IEEE.

Acknowledgements:
Professor Gary Johns for contributing to the selection of relevant research attributes.
Professor Emeritus of Management
Honorary Concordia University Research Chair in Management
John Molson School of Business
Concordia University
Montreal, Quebec, Canada
Adjunct Professor, OB/HR Division
Sauder School of Business,
University of British Columbia
Vancouver, British Columbia, Canada

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

Here is a BiBTeX citation as well:

@misc{Dua:2019 ,
author = "Dua, Dheeru and Graff, Casey",
year = "2017",
title = "{UCI} Machine Learning Repository",
url = "http://archive.ics.uci.edu/ml",
institution = "University of California, Irvine, School of Information and Computer Sciences" }




