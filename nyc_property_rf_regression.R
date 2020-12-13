# Install required libraries
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(randomForest)
library(tidyverse)
library(caret)
library(corrplot)
library(e1071)
library(lubridate)

## Options for running the code
# sol_opt  1: Random Forest Algorithm  2: Linear Regression Algorithm 
sol_opt = 1       
# is_tune    1: Tune parameters in the Random Forest model
#            0: Don't tune. Train the Random Forest model using mtry_input
is_tune = 0       
# mtry_input sets the mtry parameter for the random forest algorithm: 
# the user can select a value from 2 to 6. The optimal value is 6.
# The user can set mtry to lower values for a faster run but the 
# model will not be optimal
mtry_input = 6

# Start time of run
start_time <- Sys.time()

# Download dataset from Github and read the csv file
download.file("https://github.com/ghanag/HarvardX-Capstone-Project-NYC-House-Price-Forecast/raw/main/nyc-rolling-sales.csv",
              "./nyc-rolling-sales.csv")
nyc_house_data <- read.csv("nyc-rolling-sales.csv")

# Dimension, Column Names and first few rows of the dataset
dim(nyc_house_data)
names(nyc_house_data)
head(nyc_house_data)

## Data Cleaning

# Delete Easement Column since all Eastment values are NA and delete column X
# Apartment number and address don't appear to be useful for predicting the price
nyc_house_data <- subset(nyc_house_data, select = -c(EASE.MENT, X, 
                                                     APARTMENT.NUMBER, ADDRESS))

# Check for duplicate rows and remove them
sum(duplicated(nyc_house_data))
nyc_house_data <- unique(nyc_house_data)


## CLEANING SALE PRICE COLUMN
# Many rows don't have a sale price
sum(nyc_house_data$SALE.PRICE == " -  ")
# Removing Rows that don't have a Sale Price
nyc_house_data <- nyc_house_data[!nyc_house_data$SALE.PRICE == " -  ",]
# Change the Sale Price Variable class from Character to Numeric
nyc_house_data$SALE.PRICE <- as.numeric(nyc_house_data$SALE.PRICE)

## Check Sale Price Values
# Summary of sale prices
summary(nyc_house_data$SALE.PRICE)
# Removing houses with Sale Price less than 250,000$
nyc_house_data <- subset(nyc_house_data,SALE.PRICE > 250000)
# Check Properties with highest Sale Prices
head(nyc_house_data[order(-nyc_house_data$SALE.PRICE),])

# Check the entries with Gross square feet equal to - 
sum(nyc_house_data$GROSS.SQUARE.FEET == " -  ")
head(nyc_house_data[nyc_house_data$GROSS.SQUARE.FEET == " -  ",])
# Check the entries with Gross square feet and Land Square Feet equal to - 
sum(nyc_house_data$GROSS.SQUARE.FEET == " -  " & 
      nyc_house_data$LAND.SQUARE.FEET == " -  ")
head(nyc_house_data[nyc_house_data$GROSS.SQUARE.FEET == " -  " & 
                      nyc_house_data$LAND.SQUARE.FEET != " -  " ,])

# Convert blank ("-") values of Gross Square Feet and Land Square Feet to 0
nyc_house_data[nyc_house_data$GROSS.SQUARE.FEET == " -  ",
               names(nyc_house_data) == "GROSS.SQUARE.FEET"] = "0"
nyc_house_data[nyc_house_data$LAND.SQUARE.FEET == " -  ",
               names(nyc_house_data) == "LAND.SQUARE.FEET"] = "0"
# Converting Gross Square Feet and Land Square Feet to Numerical variables
nyc_house_data$GROSS.SQUARE.FEET <- as.numeric(nyc_house_data$GROSS.SQUARE.FEET)
nyc_house_data$LAND.SQUARE.FEET  <- as.numeric(nyc_house_data$LAND.SQUARE.FEET)

# Remove rows that have a year built equal to 0
nyc_house_data <- nyc_house_data[!nyc_house_data$YEAR.BUILT == 0, ]
# Remove rows that have a zip code equal to 0
nyc_house_data <- nyc_house_data[!nyc_house_data$ZIP.CODE == 0, ]

# Remove entries with both Residential and Commerical units equal to 0
nyc_house_data <- filter(nyc_house_data, RESIDENTIAL.UNITS != 0 |
                           COMMERCIAL.UNITS != 0)
# Number of entries with sum of Residential and Commerical units not equal to Total units
sum(nyc_house_data$RESIDENTIAL.UNITS + nyc_house_data$COMMERCIAL.UNITS != nyc_house_data$TOTAL.UNITS)
# Remove entries with sum of Residential and Commerical units not equal to Total units
nyc_house_data <- filter(nyc_house_data, RESIDENTIAL.UNITS+COMMERCIAL.UNITS == TOTAL.UNITS)

# Time is 00:00:00 for all entries in SALE.DATE. Remove time. Convert dates to numeric values.
nyc_house_data <- nyc_house_data %>% 
  separate(col = "SALE.DATE", into = c("SALE.DATE", "SALE.TIME"), sep = " ")
nyc_house_data <- subset(nyc_house_data, select = -c(SALE.TIME))
nyc_house_data$SALE.DATE <- as.numeric(ymd(nyc_house_data$SALE.DATE))

# Check if there is any NA left in the dataset. RandomForest algorithm cannot handle NA.
colSums(is.na(nyc_house_data))

# Converting Borough, neighborhood, building & tax class category variables type to factor
nyc_house_data$BOROUGH <- as.factor(nyc_house_data$BOROUGH)
nyc_house_data$BUILDING.CLASS.CATEGORY <- as.factor(nyc_house_data$BUILDING.CLASS.CATEGORY)
nyc_house_data$TAX.CLASS.AT.PRESENT <- as.factor(nyc_house_data$TAX.CLASS.AT.PRESENT)
nyc_house_data$TAX.CLASS.AT.TIME.OF.SALE <- as.factor(nyc_house_data$TAX.CLASS.AT.TIME.OF.SALE)
nyc_house_data$NEIGHBORHOOD <- as.factor(nyc_house_data$NEIGHBORHOOD)  
nyc_house_data$BUILDING.CLASS.AT.TIME.OF.SALE <- as.factor(nyc_house_data$BUILDING.CLASS.AT.TIME.OF.SALE)
nyc_house_data$BUILDING.CLASS.AT.PRESENT <- as.factor(nyc_house_data$BUILDING.CLASS.AT.PRESENT)

# Dimension of the dataset after data cleaning
dim(nyc_house_data)


## Data Exploration and Visualization

# Summary Statistics of Sale Prices 
summary(nyc_house_data$SALE.PRICE)
# Plot Histogram of Sale Prices
nyc_house_data %>% 
  ggplot(aes(SALE.PRICE)) + 
  geom_histogram(binwidth = 1e6) + 
  scale_x_continuous(limit = c(0, 15e6), name = "SALE PRICE") +
  scale_y_continuous(name = "COUNT")
# Calculate Skewness of Sale Price
skewness(nyc_house_data$SALE.PRICE,3)

# Summary Statistics of Gross Square Feet
summary(nyc_house_data$GROSS.SQUARE.FEET)
# Plot Histogram of Gross Square Feet
nyc_house_data %>% 
  ggplot(aes(GROSS.SQUARE.FEET)) + 
  geom_histogram(binwidth = 500) + 
  scale_x_continuous(limit = c(0, 1e4), name = "GROSS SQUARE FEET") +
  scale_y_continuous(limits = c(0,8e3), name = "COUNT")
# Calculate Skewness of Gross Square Feet 
skewness(nyc_house_data$GROSS.SQUARE.FEET)

# Plot Sale Price vs Gross Square Feet
nyc_house_data %>%
  ggplot(aes(x=GROSS.SQUARE.FEET,y=SALE.PRICE))+
  geom_point()+
  scale_x_continuous(name = "GROSS SQUARE FEET") +
  scale_y_continuous(name = "SALE PRICE")
# Plot Sale Price vs Gross Square Feet with different axes limits
nyc_house_data %>%
  ggplot(aes(x=GROSS.SQUARE.FEET,y=SALE.PRICE))+
  geom_point()+
  scale_x_continuous(limits = c(0,1e6), name = "GROSS SQUARE FEET") +
  scale_y_continuous(limits = c(0,1e8), name = "SALE PRICE")

# Plot Sale Price vs Building Class Category
nyc_house_data %>%
  ggplot(aes(x=BUILDING.CLASS.CATEGORY,y= SALE.PRICE))+
  geom_boxplot()+
  scale_x_discrete(name = "BUILDING CLASS CATEGORY",
                   labels=seq(1,43,1)) +
  scale_y_continuous(name = "SALE PRICE")

# Plot Sale Price vs Borough
nyc_house_data %>%
  ggplot(aes(x=as.factor(BOROUGH),y=SALE.PRICE))+
  geom_boxplot()+
  scale_y_continuous(name = "SALE PRICE")+
  scale_x_discrete(name= "BOROUGH",
                   labels=c("Manhattan","Bronx","Brooklyn","Queens","Staten Island"))

# Plot Sale Price vs BOROUGH (lower Sale price limit)
nyc_house_data %>%
  ggplot(aes(x=as.factor(BOROUGH),y=SALE.PRICE))+
  geom_boxplot()+
  scale_y_continuous(limit = c(0, 5e6), name = "SALE PRICE")+
  scale_x_discrete(name= "BOROUGH",
                   labels=c("Manhattan","Bronx","Brooklyn","Queens","Staten Island"))

# Tax Class at Present levels
levels(nyc_house_data$TAX.CLASS.AT.PRESENT)
# Plot Sale Price vs Tax Class at Present
nyc_house_data %>%
  ggplot(aes(x=TAX.CLASS.AT.PRESENT,y=SALE.PRICE))+
  geom_boxplot()+
  scale_y_continuous(name = "SALE PRICE")+
  scale_x_discrete(name= "TAX CLASS AT PRESENT")

# Tax Class at Time of Sale levels
levels(nyc_house_data$TAX.CLASS.AT.TIME.OF.SALE)
# Plot Sale Price vs Tax Class at Time of Sale
nyc_house_data %>%
  ggplot(aes(x=as.factor(TAX.CLASS.AT.TIME.OF.SALE),y=SALE.PRICE))+
  geom_boxplot()+
  scale_y_continuous(breaks= seq(0,5e6,0.2e6))+
  scale_y_continuous(name = "SALE PRICE")+
  scale_x_discrete(name= "TAX CLASS AT TIME OF SALE")
  
# Plot Sale Price vs Year Built
nyc_house_data %>%
  ggplot(aes(x=YEAR.BUILT,y=SALE.PRICE))+
  geom_point()+
  scale_x_continuous(name = "YEAR BUILT")+
  scale_y_continuous(name = "SALE PRICE")
# Plot Sale Price vs Year Built with lower sale price limit
nyc_house_data %>%
  ggplot(aes(x=YEAR.BUILT,y=SALE.PRICE))+
  geom_point()+
  scale_x_continuous(name = "YEAR BUILT")+
  scale_y_continuous(name = "SALE PRICE", limit = c(0, 2e7))

# Plot Correlations of numerical variables 
num_col <- sapply(nyc_house_data, is.numeric) # find numeric columns
corrs   <- cor(nyc_house_data[,num_col])
corrplot(corrs, method="square")

# Apply log transformation to Numerical Variable with skewness > 0.75
for(x in seq(1,ncol(nyc_house_data)))
{
  if(is.numeric(nyc_house_data[,x]))
  {
    skew <- skewness(nyc_house_data[,x],na.rm = T)
    if (skew > 0.75 &
        names(nyc_house_data[x]) != "RESIDENTIAL.UNITS" &
        names(nyc_house_data[x]) != "COMMERCIAL.UNITS" &
        names(nyc_house_data[x]) != "LAND.SQUARE.FEET" &
        names(nyc_house_data[x]) != "GROSS.SQUARE.FEET"
        )
    {
      print(skew)
      print(names(nyc_house_data[x]))
      nyc_house_data[,x] <- log(nyc_house_data[,x])+1
    }
  }
}

# Plot Histogram of Sale Prices after log transformation
nyc_house_data %>% 
  ggplot(aes(SALE.PRICE)) + 
  geom_histogram(binwidth = 1) + 
  scale_x_continuous(limit = c(10, 20), name = "SALE PRICE") +
  scale_y_continuous(name = "COUNT")

# Calculate Skewness of Sale Price after log transformation
skewness(nyc_house_data$SALE.PRICE)

# Plot Correlations of numerical variables after log transformation
num_col <- sapply(nyc_house_data, is.numeric) # find numeric columns
corrs   <- cor(nyc_house_data[,num_col])
corrplot(corrs, method="square")

# Create training and test sets
set.seed(1)
test_index <- createDataPartition(y = nyc_house_data$SALE.PRICE, times = 1,
                                  p = 0.1, list = FALSE)
train_set <- nyc_house_data[-test_index,]
test_set  <- nyc_house_data[test_index,]

# Print categorical predictors that have more than 53 levels/categories
for(x in seq(1,ncol(nyc_house_data)))
{
  if(is.factor(nyc_house_data[,x]) & length(unique(nyc_house_data[,x])) > 53)
  {
    print(names(nyc_house_data[x]))
    print(length(unique(nyc_house_data[,x])))
  }
}


# Algorithm 1: RandomForest
# Remove Categorical Variables with more than 53 levels since RandomForest cannot handle that
if(sol_opt == 1){
  nyc_house_data <- subset(nyc_house_data, select = -c(NEIGHBORHOOD,BUILDING.CLASS.AT.PRESENT,
                                                  BUILDING.CLASS.AT.TIME.OF.SALE
  ))
  test_set <- subset(test_set, select = -c(NEIGHBORHOOD,BUILDING.CLASS.AT.PRESENT,
                                                 BUILDING.CLASS.AT.TIME.OF.SALE
  ))
  train_set <- subset(train_set, select = -c(NEIGHBORHOOD,BUILDING.CLASS.AT.PRESENT,
                                                       BUILDING.CLASS.AT.TIME.OF.SALE
                                                       ))
}
# If is_tune = 1, the code tunes the mtry parameter, which is the number of variables
# randomly sampled at each split
if(sol_opt == 1) {
  if(is_tune == 1) {
    nyc_house_forest1 <- train(
      SALE.PRICE ~ .,
      data = train_set,
      method = "rf",
      tuneGrid = data.frame(mtry = seq(2, 6, 1))
    )
    print(nyc_house_forest1$finalModel)
    print(nyc_house_forest1$bestTune)        
    print(nyc_house_forest1$results)
    print(importance(nyc_house_forest1))           # Variable Importance measure   
    varImpPlot(nyc_house_forest1, cex = 0.5)       # Chart of Variable Importance
    fit1 <- predict(nyc_house_forest1, newdata = test_set)
    data1 = data.frame(obs=test_set$SALE.PRICE, pred=fit1)
    defaultSummary(exp(data1-1))     # Compare model prediction with test_set 
  } else{
    # Random Forest Training with the mtry_input value
    nyc_house_forest2 <- randomForest(SALE.PRICE ~ . ,
                                      data = train_set, mtry = mtry_input, importance = T)
    print(nyc_house_forest2)
    print(importance(nyc_house_forest2))           # Variable Importance measure   
    varImpPlot(nyc_house_forest2, cex = 0.5)       # Chart of Variable Importance
    
    fit2 <- predict(nyc_house_forest2, newdata = test_set)
    data2 = data.frame(obs=test_set$SALE.PRICE, pred=fit2)
    defaultSummary(exp(data2-1))     # Compare model prediction with test_set        
  }
}
# Linear Regression
# Remove the same parameters as the ones for Random Forest for comparison purpose

# Convert character factors to numeric factors
if(sol_opt == 2) {nyc_house_data$BUILDING.CLASS.CATEGORY <- as.factor(as.numeric(
  nyc_house_data$BUILDING.CLASS.CATEGORY))

nyc_house_data$TAX.CLASS.AT.PRESENT <- as.factor(as.numeric(
  nyc_house_data$TAX.CLASS.AT.PRESENT))

nyc_house_data$NEIGHBORHOOD <- as.factor(as.numeric(
  nyc_house_data$NEIGHBORHOOD))

nyc_house_data$BUILDING.CLASS.AT.PRESENT <- as.factor(as.numeric(
  nyc_house_data$BUILDING.CLASS.AT.PRESENT))

nyc_house_data$BUILDING.CLASS.AT.TIME.OF.SALE <- as.factor(as.numeric(
  nyc_house_data$BUILDING.CLASS.AT.TIME.OF.SALE))
}
# Use Linear Regression to Predict Property Sale Prices
if(sol_opt == 2) {
  nyc_house_data <- subset(nyc_house_data, select = -c(NEIGHBORHOOD, 
                                                       BUILDING.CLASS.AT.PRESENT,
                                                       BUILDING.CLASS.AT.TIME.OF.SALE))

  model_lm=train(SALE.PRICE~., 
             data=train_set,
             method="lm"
  )

  fit_lm <- predict(model_lm, newdata = test_set)
  data = data.frame(obs=test_set$SALE.PRICE, pred=fit_lm)
  defaultSummary(exp(data-1))           # Compare the model prediction with test_set sale prices
}

end_time <- Sys.time()
run_time = end_time - start_time
print(c("Run time is" ,run_time))