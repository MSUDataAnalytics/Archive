# Lab 2: Linear Regression and Data Analysis

This question will require you to analyze a new dataset. Specifically, this lab utilizes the “ames.csv” data [(found here)](/assets/ames.csv). This data is a famous dataset which concerns housing sales from Ames, Iowa.

### Backstory
You have been recently hired to Zillow’s Zestimate$^{\text{TM}}$ product team as a junior analyst. As a part of their regular hazing, they have given you access to a small subset of their historic sales data. Your job is to present some basic predictions for housing values in a small geographic area (Ames, IA) using this historical pricing.

First, let's load the data.

```{r}
df <- read.table("https://msudataanalytics.github.io/SSC442/assets/ames.csv",
                 header = FALSE,
                 sep = ",")
```


(a) Produce a scatterplot matrix which includes all of the variables in the data set.
(b) Compute the matrix of correlations between the variables using the function cor(). You will need to exclude the name variable, which is qualitative.
(c) Use the lm() function to perform a multiple linear regression with mpg as the response and all other variables except name as the predictors. Use the summary() function to print the results. Comment on the output. For instance:
i. Is there a relationship between the predictors and the re- sponse?
ii. Which predictors appear to have a statistically significant relationship to the response?
iii. What does the coefficient for the year variable suggest?
(d) Use the plot() function to produce diagnostic plots of the linear regression fit. Comment on any problems you see with the fit. Do the residual plots suggest any unusually large outliers? Does the leverage plot identify any observations with unusually high leverage?
(e) Use the * and : symbols to fit linear regression models with interaction effects. Do any interactions appear to be statistically significant?
(f) Try a few different transformations of the variables, such as
2
cor()
√
log(X), X, X . Comment on your findings
