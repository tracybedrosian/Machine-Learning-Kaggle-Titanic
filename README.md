# Kaggle Titanic Data
## Predicting survival on the Titanic using machine learning
In this project I analyze passenger data from the Titanic and build a random forest model to predict survival. 

### 1. Packages and data
First, load the required packages and read in data from Kaggle
```library(ggplot2)
library(rpart)
library(randomForest)
train <- read.csv("train.csv")
test <- read.csv("test.csv")
```

### 2. Combine train and test datasets

```test$Survived <- NA  
full_data <- rbind(train, test)
```

### 3. Create new features
For my first attempt I take the lead from great examples on Kaggle Kernels
#### A. Family size
```full_data$familysize <- NA  
full_data$familysize <- full_data$SibSp + full_data$Parch + 1
```

#### B. Child
```full_data$Child <- NA  
full_data$Child[full_data$Age >= 18] <- 0  
full_data$Child[full_data$Age < 18] <- 1
```

#### C. Title
```full_data$Title <- gsub('(.*, )|(\\..*)', '', full_data$Name)  
table(full_data$Sex, full_data$Title)  
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')  
full_data$Title[full_data$Title == 'Mlle']        <- 'Miss'  
full_data$Title[full_data$Title == 'Ms']          <- 'Miss'  
full_data$Title[full_data$Title == 'Mme']         <- 'Mrs'  
full_data$Title[full_data$Title %in% rare_title]  <- 'Rare Title'  
full_data$Title <- factor(full_data$Title)
```

### 4. Identify missing values
```summary(full_data)  
levels(full_data$Embarked)[1] <- NA  
which(is.na(full_data$Embarked))
```

### 5. Impute missing values
#### A. Embarkment: Two passengers are missing data for place of embarkment, but we can determine that info based on their fare and class
```full_data[c(62, 830), 'Fare']  
ggplot(full_data, aes(x = Embarked, y = Fare, Fill = factor(Pclass))) + geom_boxplot()  
full_data$Embarked[c(62, 830)] <- 'C'
```

#### B. Fare: Another passenger is missing fare data, but we can estimate it from their class and point of embarkment
```which(is.na(full_data$Fare))  
full_data[c(1044), 'Fare']  
full_data[1044, ]  
ggplot(full_data[full_data$Pclass == '3' & full_data$Embarked == 'S', ], aes(x=Fare)) + geom_density()  
median(full_data[full_data$Pclass == '3' & full_data$Embarked == 'S', ]$Fare, na.rm = TRUE)  
full_data$Fare[1044] <- 8.05
```

#### C. Age: Many passengers are missing data for age, but we can use the ```rpart``` package to determine the most important 
predictors and then impute an age value for each passenger
```age_tree <- rpart(Age ~ Title + Pclass + Sex + Fare + Embarked, data = full_data[!is.na(full_data$Age),], method = "anova")  
full_data$Age[is.na(full_data$Age)] <- predict(age_tree, full_data[is.na(full_data$Age),])  
full_data$Child[full_data$Age >= 18] <- 0  
full_data$Child[full_data$Age < 18] <- 1
```

### 6. Split up the train and test datasets
```train <- full_data[1:891,]  
test <- full_data[892:1309,]
```

### 7. Make survival predictions
Use our newly created features along with the given variables to run a random forest model that predicts which passengers survived
```rf_model <- randomForest(factor(Survived) ~ Pclass + Title + Sex + SibSp + Age  
                                  + Parch + Fare + Embarked + Child + familysize, data=train)  
plot(rf_model)  
prediction <- predict(rf_model, test)
```

### 8. Export results as a csv
```solution <- data.frame(PassengerId = test$PassengerId, Survived = prediction)  
write.csv(solution, file = 'rf_mod_solution.csv', row.names = FALSE)```



