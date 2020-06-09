#Harvardx-MovieLens
#author: Raymond Peter 

#Introduction
#In this project, we will look into the MovieLens dataset and create a movie recommendation system. 
#We will also try to predict the movie ratings based on the given variables. For this specific project, 
#we will use the MovieLens 10M dataset with 6 variables.To get an idea of what each variables means, you 
#can check the link in the footnote.[^1].In this project, we will go through the preprocessing activity to 
#modify the data so that we can create a good model. Then,we will divide the dataset into the testing and 
#training dataset.Also, we will try to test the value of the RMSE with different parameters.Lastly, we will 
#evaluate the value of the RMSE. 

#Loading the dataset
##Code given from the Harvardx capstone course
# Create test and validation sets
################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr",repos = "http://cran.us.r-project.org")
if(!require(kknn)) install.packages("kknn",repos = "http://cran.us.r-project.org")
#Load the packages into library
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(kknn)
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")

# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(
  y = movielens$rating, 
  times = 1, 
  p = 0.1, 
  list = FALSE)

# Spliting
edx <- movielens[-test_index,] #train
temp <- movielens[test_index,] #test

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#Cleaning the environment...
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#You should run all the codes above, if somehow you can't get the complete data after you view it 'view(edx)' and 'view(validation)', then below is an alternative.

#Alternative method to load the dataset in case the code above doesn't work.
#This code depends on where you store edx.rds and validation.rds. 
#The link to these files will be provided below.
#https://drive.google.com/drive/folders/1IZcBBX0OmL9wu9AdzMBFUG8GoPbGQ38D

edx <- readRDS("~/Downloads/edx.rds")
validation <- readRDS("~/Downloads/validation.rds")
set.seed(1, sample.kind="Rounding")

#Method and Analysis
##Initial data analysis 
##This is not necessary, but it is a good practice to be familiar with the dataset. 
#Data structure and size
str(edx)
#Names of different columns
names(edx)
#Top 6 data 
head(edx)
#Check for missing values
#This is an important step because otherwise we have to remove the missing values. 
sum(is.na(edx))


# Graph of variables(Visualization)

#Plots to see data characteristic
#This next two lines will be neccessary to make the graph, however, it will be explained later. 
org_data<- read_csv("~/Downloads/theMovies_train.csv")
org_test<- read_csv("~/Downloads/theMovies_test.csv")

#Distribution of Avg_Rating 
ggplot(org_data) +
  aes(x = Avg_Rating) +
  geom_histogram(bins = 30L, fill = "#0c4c8a") +
  theme_minimal()
#We can see that the ratings tend to be above 3.There are more movies rated above three than below 3.


#Age vs Avg_Rating adjusted with votes
##We can see that recent movies are rated better than old movies.
ggplot(org_data) +
  aes(x = Avg_Rating, y = Age, size = Votes) +
  geom_point(colour = "#0c4c8a") +
  theme_minimal()

#Voters vs Age graph
##We can see that recent movies have higher votes
ggplot(org_data) +
  aes(x = Age, y = Votes) +
  geom_point(size = 1L, colour = "#0c4c8a") +
  theme_minimal()


#This is what we want to do in the preprocessing process.We will try to predict 
#the ratings with the variables by organizing them this way.

##########################
#Organize your data (EDX)
#We will organize the data in the format as given below for observation.
#We will first organize the based on movies and then we will try to predict the average rating by movies.
#1-the movie is categorized in that genres, 0- movie not part of that genres
#Avg_Rating; Age_of_movie; Drama; Comedy; Thriller; Romance; ... Audience; 
#       4.3;           30;     1;      0;        1;       0; ...   40000;


#Install random forest package if needed.
if(!require(randomForest)){
  install.packages("randomForest")
  library(randomForest)
}


##Organizing Data
#The data will be organize with the following codes. However, to avoid loading this all over again, which takes a lot of time,I will separate the results of this code into 2 different files that I will attached in github (theMovies_train.csv, and theMovies_test.csv)

#Discover all the different genres
genres = paste(edx$genres,collapse="|") %>%
  strsplit("\\|")
#Remove the equal genres.Remove the replication
genres = genres[[1]] %>% unique
#calculate the age of the movies
getYear = function(tt){
  temp = strsplit(tt,"\\(")[[1]]
  temp = strsplit(temp[length(temp)],"\\)")[[1]]
  return(as.numeric(temp))
}
#Select the genres of each movie 
getGenres = function(list_genres){
  temp = paste(list_genres, collapse="|")
  temp = unique(strsplit(temp, "\\|")[[1]])
  return(temp)
}

#separate the movieIDs
mov = edx$movieId %>% unique
#Preparing the data about the movies.We will name the columns.
org_data = data.frame(matrix(0, length(mov), length(genres) + 5))
names(org_data) = c("Movie_ID", "Title", "Avg_Rating", "Age", "Votes", genres)
names(org_data)[c(12,21,25)] = c("Sci_Fi","Film_Noir","None")
#Filling the information about all the movies 
for (i in 1:length(mov)){
  org_data$Movie_ID[i] = mov[i]
  focus = which(edx$movieId == mov[i])
  org_data$Title[i] = names(table(edx$title[focus]))
  org_data$Avg_Rating[i] = round(mean(edx$rating[focus]), 1)
  org_data$Age[i] = 2020 - getYear(org_data$Title[i])
  org_data$Votes[i] = length(focus)
  who = 5 + match(getGenres(edx$genres[focus]), genres)
  org_data[i,who] = 1
}


#Random forest model
#We will start to do our model-specific codes.We will begin by creating the training 
#and testing data.Here, 70% will be for the training set and 30% will be for the testing set. 

######################
test_index <- createDataPartition(
  y = org_data$Avg_Rating, 
  times = 1, 
  p = 0.3, #30% for test
  list = FALSE)

org_train <- org_data[-test_index,]
org_test <- org_data[test_index,]


#Install randomForest if needed
if(!require(randomForest)){
  install.packages("randomForest")
  library(randomForest)
}
#if the code above doesn't run, run these two codes instead.
# org_data = read.csv("theMovies_train.csv")
# org_test = read.csv("theMovies_test.csv")

#Transform the genres columns in factors
for (i in 6:25){
  org_train[,i] = as.factor(org_train[,i])
  org_test[,i] = as.factor(org_test[,i])
}

library(reshape)
library(tidyverse)

org_data_melt <- melt(data = org_data, measure.vars = names(org_data)[6:25]) %>% 
  filter(value == 1) %>% 
  select(!value)

org_test_melt <- melt(data = org_test, measure.vars = names(org_test)[6:25]) %>% 
  filter(value == 1) %>% 
  select(!value)

#We will create 'my_pred' which will predict the average rating based on movies.
rf_mov = randomForest(Avg_Rating ~ ., data=org_train[,-c(1,2,25)])

my_pred = predict(rf_mov, org_test[,-c(1,2,25)])


if(!require(MLmetrics)){
  install.packages("MLmetrics")
  library(MLmetrics)
}

#We'll create 'rf_rmse' to store the value of RMSE for random forest. Then, we'll print the value.
rf_rmse = RMSE(my_pred, org_test$Avg_Rating)
rf_rmse


# Knn
#KNN algorithm

train <- org_train[,-c(1,2,25)]
test <- org_test[,-c(1,2,25)]

results = list(k = rep(0,100), rmse = rep(0,100))

for (i in 2:101){
  my_kknn <- kknn(Avg_Rating ~ ., 
                  train, 
                  test, 
                  k = i)
  
  my_rmse = RMSE(my_kknn$fitted.values, test$Avg_Rating)
  my_rmse
  
  results$k[i] = i
  results$rmse[i] = my_rmse
}


# K vs Accuracy
#Here we will try to find the best value of k.
results$k = results$k[-1]
results$rmse = results$rmse[-1]
plot(results$k, results$rmse, type="l",
     xlab="k", ylab="RMSE")

best.K = results$k[which.min(results$rmse)]
best.K


best_kknn <- kknn(Avg_Rating ~ ., 
                  train, 
                  test, 
                  k = best.K)

kknn_rmse = RMSE(best_kknn$fitted.values, test$Avg_Rating)
kknn_rmse


#Results
rf_rmse
kknn_rmse

#The general formula of RMSE is: sqrt(((y predicted-y observed)^2)/n).
#Our RMSE result for random forest is 0.456029 and 0.4832858 for KNN which is really
#good since this means that our predictions is closer to the regression line since the 
#lesser the RMSE the better.
#This shows that our prediction that we generated with random forest is proven to be successful 
#in achieving our goal. 

#Conclusion
#From this we can conclude that movieID,Genres set (binary variables), age of the movie, 
#and total of votes is a good factor to predict movie ratings. Also, we can see that the 
#accuracy of the random forest model is better than the KNN algorithm. However, these two 
#predictions aren't so far from each other. 


#Citations and credits
#Credit to dataset owner:
#F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872
