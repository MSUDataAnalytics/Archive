
### `SAheart` Example

To illustrate the use of logistic regression, we will use the `SAheart` dataset from the `ElemStatLearn` package.

```{r}
# install.packages("ElemStatLearn")
library(ElemStatLearn)
data("SAheart")
```

```{r}
knitr::kable(head(SAheart))
```

This data comes from a retrospective sample of males in a heart-disease high-risk region of the Western Cape, South Africa. The `chd` variable, which we will use as a response, indicates whether or not coronary heart disease is present in an individual. Note that this is coded as a numeric `0` / `1` variable. Using this as a response with `glm()` it is important to indicate `family = binomial`, otherwise ordinary linear regression will be fit. Later, we will see the use of a factor variable response, which is actually preferred, as you cannot accidentally fit ordinary linear regression.

The predictors are various measurements for each individual, many related to heart health. For example `sbp`, systolic blood pressure, and `ldl`, low density lipoprotein cholesterol. For full details, use `?SAheart`.

We'll begin by attempting to model the probability of coronary heart disease based on low density lipoprotein cholesterol. That is, we will fit the model

$$
\log\left(\frac{P[\texttt{chd} = 1]}{1 - P[\texttt{chd} = 1]}\right) = \beta_0 + \beta_{\texttt{ldl}} x_{\texttt{ldl}}
$$

```{r}
chd_mod_ldl = glm(chd ~ ldl, data = SAheart, family = binomial)
plot(jitter(chd, factor = 0.1) ~ ldl, data = SAheart, pch = 20,
     ylab = "Probability of CHD", xlab = "Low Density Lipoprotein Cholesterol")
grid()
curve(predict(chd_mod_ldl, data.frame(ldl = x), type = "response"),
      add = TRUE, col = "dodgerblue", lty = 2)
```

As before, we plot the data in addition to the estimated probabilities. Note that we have "jittered" the data to make it easier to visualize, but the data do only take values `0` and `1`.

As we would expect, this plot indicates that as `ldl` increases, so does the probability of `chd`.

```{r}
coef(summary(chd_mod_ldl))
```

To perform the test

$$
H_0: \beta_{\texttt{ldl}} = 0
$$

we use the `summary()` function as we have done so many times before. Like the $t$-test for ordinary linear regression, this returns the estimate of the parameter, its standard error, the relevant test statistic ($z$), and its p-value. Here we have an incredibly low p-value, so we reject the null hypothesis. The `ldl` variable appears to be a significant predictor.

When fitting logistic regression, we can use the same formula syntax as ordinary linear regression. So, to fit an additive model using all available predictors, we use:

```{r}
chd_mod_additive = glm(chd ~ ., data = SAheart, family = binomial)
```

We can then use the likelihood-ratio test to compare the two model. Specifically, we are testing

$$
H_0: \beta_{\texttt{sbp}} = \beta_{\texttt{tobacco}} = \beta_{\texttt{adiposity}} = \beta_{\texttt{famhist}} = \beta_{\texttt{typea}} = \beta_{\texttt{obesity}} = \beta_{\texttt{alcohol}} = \beta_{\texttt{age}} = 0
$$

We could manually calculate the test statistic,

```{r}
-2 * as.numeric(logLik(chd_mod_ldl) - logLik(chd_mod_additive))
```

Or we could utilize the `anova()` function. By specifying `test = "LRT"`, `R` will use the likelihood-ratio test to compare the two models.

```{r}
anova(chd_mod_ldl, chd_mod_additive, test = "LRT")
```

We see that the test statistic that we had just calculated appears in the output. The very small p-value suggests that we prefer the larger model.

While we prefer the additive model compared to the model with only a single predictor, do we actually need all of the predictors in the additive model? To select a subset of predictors, we can use a stepwise procedure as we did with ordinary linear regression. Recall that AIC and BIC were defined in terms of likelihoods. Here we demonstrate using AIC with a backwards selection procedure.

```{r}
chd_mod_selected = step(chd_mod_additive, trace = 0)
coef(chd_mod_selected)
```

We could again compare this model to the additive models.

$$
H_0: \beta_{\texttt{sbp}} = \beta_{\texttt{adiposity}} = \beta_{\texttt{obesity}} = \beta_{\texttt{alcohol}} = 0
$$

```{r}
anova(chd_mod_selected, chd_mod_additive, test = "LRT")
```

Here it seems that we would prefer the selected model.

### Confidence Intervals

We can create confidence intervals for the $\beta$ parameters using the `confint()` function as we did with ordinary linear regression.

```{r}
confint(chd_mod_selected, level = 0.99)
```

Note that we could create intervals by rearranging the results of the Wald test to obtain the Wald confidence interval. This would be given by

$$
\hat{\beta}_j \pm z_{\alpha/2} \cdot \text{SE}[\hat{\beta}_j].
$$

However, `R` is using a slightly different approach based on a concept called the profile likelihood. (The details of which we will omit.) Ultimately the intervals reported will be similar, but the method used by `R` is more common in practice, probably at least partially because it is the default approach in `R`. Check to see how intervals using the formula above compare to those from the output of `confint()`. (Or, note that using `confint.default()` will return the results of calculating the Wald confidence interval.)

### Confidence Intervals for Mean Response

Confidence intervals for the mean response require some additional thought. With a "large enough" sample, we have

$$
\frac{\hat{\eta}({\bf x}) - \eta({\bf x})}{\text{SE}[\hat{\eta}({\bf x})]} \overset{\text{approx}}{\sim} N(0, 1)
$$

Then we can create an approximate $(1 - \alpha)\%$ confidence intervals for $\eta({\bf x})$ using

$$
\hat{\eta}({\bf x}) \pm z_{\alpha/2} \cdot \text{SE}[\hat{\eta}({\bf x})]
$$

where $z_{\alpha/2}$ is the critical value such that $P(Z > z_{\alpha/2}) = \alpha/2$.

This isn't a particularly interesting interval. Instead, what we really want is an interval for the mean response, $p({\bf x})$. To obtain an interval for $p({\bf x})$, we simply apply the inverse logit transform to the endpoints of the interval for $\eta.$

$$
\left(\text{logit}^{-1}(\hat{\eta}({\bf x}) - z_{\alpha/2} \cdot \text{SE}[\hat{\eta}({\bf x})] ), \ \text{logit}^{-1}(\hat{\eta}({\bf x}) + z_{\alpha/2} \cdot \text{SE}[\hat{\eta}({\bf x})])\right)
$$

To demonstrate creating these intervals, we'll consider a new observation.

```{r}
new_obs = data.frame(
  sbp = 148.0,
  tobacco = 5,
  ldl = 12,
  adiposity = 31.23,
  famhist = "Present",
  typea = 47,
  obesity = 28.50,
  alcohol = 23.89,
  age = 60
)
```

Fist, we'll use the `predict()` function to obtain $\hat{\eta}({\bf x})$ for this observation.

```{r}
eta_hat = predict(chd_mod_selected, new_obs, se.fit = TRUE, type = "link")
eta_hat
```

By setting `se.fit = TRUE`, `R` also computes $\text{SE}[\hat{\eta}({\bf x})]$. Note that we used `type = "link"`, but this is actually a default value. We added it here to stress that the output from `predict()` will be the value of the link function.

```{r}
z_crit = round(qnorm(0.975), 2)
round(z_crit, 2)
```

After obtaining the correct critical value, we can easily create a $95\%$ confidence interval for $\eta({\bf x})$.

```{r}
eta_hat$fit + c(-1, 1) * z_crit * eta_hat$se.fit
```

Now we simply need to apply the correct transformation to make this a confidence interval for $p({\bf x})$, the probability of coronary heart disease for this observation. Note that the `boot` package contains functions `logit()` and `inv.logit()` which are the logit and inverse logit transformations, respectively.

```{r}
boot::inv.logit(eta_hat$fit + c(-1, 1) * z_crit * eta_hat$se.fit)
```

Notice, as we would expect, the bounds of this interval are both between 0 and 1. Also, since both bounds of the interval for $\eta({\bf x})$ are positive, both bounds of the interval for $p({\bf x})$ are greater than 0.5.

### Formula Syntax

Without really thinking about it, we've been using our previous knowledge of `R`'s model formula syntax to fit logistic regression.

#### Interactions

Let's add an interaction between LDL and family history for the model we selected.

```{r}
chd_mod_interaction = glm(chd ~ alcohol + ldl + famhist + typea + age + ldl:famhist,
                          data = SAheart, family = binomial)
summary(chd_mod_interaction)
```

Based on the $z$-test seen in the above summary, this interaction is significant. The effect of LDL on the probability of CHD is different depending on family history.

#### Polynomial Terms

Let's take the previous model, and now add a polynomial term.

```{r}
chd_mod_int_quad = glm(chd ~ alcohol + ldl + famhist + typea + age + ldl:famhist + I(ldl^2),
                       data = SAheart, family = binomial)
summary(chd_mod_int_quad)
```

Unsurprisingly, since this additional transformed variable wasn't intelligently chosen, it is not significant. However, this does allow us to stress the fact that the syntax notation that we had been using with `lm()` works basically exactly the same for `glm()`, however now we understand that this is specifying the linear combination of predictions, $\eta({\bf x})$.

That is, the above fits the model

$$
\log\left(\frac{p({\bf x})}{1 - p({\bf x})}\right) =
\beta_0 +
\beta_{1}x_{\texttt{alcohol}} +
\beta_{2}x_{\texttt{ldl}} +
\beta_{3}x_{\texttt{famhist}} +
\beta_{4}x_{\texttt{typea}} +
\beta_{5}x_{\texttt{age}} +
\beta_{6}x_{\texttt{ldl}}x_{\texttt{famhist}} +
\beta_{7}x_{\texttt{ldl}}^2
$$

You may have realized this before we actually explicitly wrote it down!

### Deviance

You have probably noticed that the output from `summary()` is also very similar to that of ordinary linear regression. One difference, is the "deviance" being reported. The `Null deviance` is the deviance for the null model, that is, a model with no predictors. The `Residual deviance` is the deviance for the mode that was fit.

[**Deviance**](https://en.wikipedia.org/wiki/Deviance_(statistics)){target="_blank"} compares the model to a saturated model. (Without repeated observations, a saturated model is a model that fits perfectly, using a parameter for each observation.) Essentially, deviance is a generalized *residual sum of squared* for GLMs. Like RSS, deviance decreased as the model complexity increases.

```{r}
deviance(chd_mod_ldl)
deviance(chd_mod_selected)
deviance(chd_mod_additive)
```

Note that these are nested, and we see that deviance does decrease as the model size becomes larger. So while a lower deviance is better, if the model becomes too big, it may be overfitting. Note that `R` also outputs AIC in the summary, which will penalize according to model size, to prevent overfitting.

## Classification

So far we've mostly used logistic regression to estimate class probabilities. The somewhat obvious next step is to use these probabilities to make "predictions," which in this context, we would call **classifications**. Based on the values of the predictors, should an observation be classified as $Y = 1$ or as $Y = 0$?

Suppose we didn't need to estimate probabilities from data, and instead, we actually knew both

$$
p({\bf x}) = P[Y = 1 \mid {\bf X} = {\bf x}]
$$

and

$$
1 - p({\bf x}) = P[Y = 0 \mid {\bf X} = {\bf x}].
$$

With this information, classifying observations based on the values of the predictors is actually extremely easy. Simply classify an observation to the class ($0$ or $1$) with the larger probability. In general, this result is called the **Bayes Classifier**,

$$
C^B({\bf x}) = \underset{k}{\mathrm{argmax}} \ P[Y = k \mid {\bf X = x}].
$$

For a binary response, that is,

$$
\hat{C}(\bf x) =
\begin{cases}
      1 & p({\bf x}) > 0.5 \\
      0 & p({\bf x}) \leq 0.5
\end{cases}
$$

Simply put, the Bayes classifier (not to be confused with the Naive Bayes Classifier) minimizes the probability of misclassification by classifying each observation to the class with the highest probability. Unfortunately, in practice, we won't know the necessary probabilities to directly use the Bayes classifier. Instead we'll have to use estimated probabilities. So to create a classifier that seeks to minimize misclassifications, we would use,

$$
\hat{C}({\bf x}) = \underset{k}{\mathrm{argmax}} \ \hat{P}[Y = k \mid {\bf X = x}].
$$

In the case of a binary response since $\hat{p}({\bf x}) = 1 - \hat{p}({\bf x})$, this becomes

$$
\hat{C}(\bf x) =
\begin{cases}
      1 & \hat{p}({\bf x}) > 0.5 \\
      0 & \hat{p}({\bf x}) \leq 0.5
\end{cases}
$$

Using this simple classification rule, we can turn logistic regression into a classifier. To use logistic regression for classification, we first use logistic regression to obtain estimated probabilities, $\hat{p}({\bf x})$, then use these in conjunction with the above classification rule.

Logistic regression is just one of many ways that these probabilities could be estimated. In a course completely focused on machine learning, you'll learn many additional ways to do this, as well as methods to directly make classifications without needing to first estimate probabilities. But since we had already introduced logistic regression, it makes sense to discuss it in the context of classification.

### `spam` Example

To illustrate the use of logistic regression as a classifier, we will use the `spam` dataset from the `kernlab` package.

```{r}
# install.packages("kernlab")
library(kernlab)
data("spam")
tibble::as.tibble(spam)
```

This dataset, created in the late 1990s at Hewlett-Packard Labs, contains 4601 emails, of which 1813 are considered spam. The remaining are not spam. (Which for simplicity, we might call, ham.) Additional details can be obtained by using `?spam` of by visiting the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/spambase){target="_blank"}.

The response variable, `type`, is a **factor** with levels that label each email as `spam` or `nonspam`. When fitting models, `nonspam` will be the reference level, $Y = 0$, as it comes first alphabetically.

```{r}
is.factor(spam$type)
levels(spam$type)
```

Many of the predictors (often called features in machine learning) are engineered based on the emails. For example, `charDollar` is the number of times an email contains the `$` character. Some variables are highly specific to this dataset, for example `george` and `num650`. (The name and area code for one of the researchers whose emails were used.) We should keep in mind that this dataset was created based on emails send to academic type researcher in the 1990s. Any results we derive probably won't generalize to modern emails for the general public.

To get started, we'll first test-train split the data.

```{r}
set.seed(42)
# spam_idx = sample(nrow(spam), round(nrow(spam) / 2))
spam_idx = sample(nrow(spam), 1000)
spam_trn = spam[spam_idx, ]
spam_tst = spam[-spam_idx, ]
```

We've used a somewhat small train set relative to the total size of the dataset. In practice it should likely be larger, but this is simply to keep training time low for illustration and rendering of this document.

```{r, message = FALSE, warning = FALSE}
fit_caps = glm(type ~ capitalTotal,
               data = spam_trn, family = binomial)
fit_selected = glm(type ~ edu + money + capitalTotal + charDollar,
                   data = spam_trn, family = binomial)
fit_additive = glm(type ~ .,
                   data = spam_trn, family = binomial)
fit_over = glm(type ~ capitalTotal * (.),
               data = spam_trn, family = binomial, maxit = 50)
```

We'll fit four logistic regressions, each more complex than the previous. Note that we're suppressing two warnings. The first we briefly mentioned previously.

```{r, echo = FALSE}
message("Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred")
```

Note that, when we receive this warning, we should be highly suspicious of the parameter estimates.

```{r}
coef(fit_selected)
```

However, the model can still be used to create a classifier, and we will evaluate that classifier on its own merits.

We also, "suppressed" the warning:

```{r, echo = FALSE}
message("Warning: glm.fit: algorithm did not converge")
```

In reality, we didn't actually suppress it, but instead changed `maxit` to `50`, when fitting the model `fit_over`. This was enough additional iterations to allow the iteratively reweighted least squares algorithm to converge when fitting the model.

### Evaluating Classifiers

The metric we'll be most interested in for evaluating the overall performance of a classifier is the **misclassification rate**. (Sometimes, instead accuracy is reported, which is instead the proportion of correction classifications, so both metrics serve the same purpose.)

$$
\text{Misclass}(\hat{C}, \text{Data}) = \frac{1}{n}\sum_{i = 1}^{n}I(y_i \neq \hat{C}({\bf x_i}))
$$

$$
I(y_i \neq \hat{C}({\bf x_i})) =
\begin{cases}
  0 & y_i = \hat{C}({\bf x_i}) \\
  1 & y_i \neq \hat{C}({\bf x_i}) \\
\end{cases}
$$

When using this metric on the training data, it will have the same issues as RSS did for ordinary linear regression, that is, it will only go down.

```{r}
# training misclassification rate
mean(ifelse(predict(fit_caps) > 0, "spam", "nonspam") != spam_trn$type)
mean(ifelse(predict(fit_selected) > 0, "spam", "nonspam") != spam_trn$type)
mean(ifelse(predict(fit_additive) > 0, "spam", "nonspam") != spam_trn$type)
mean(ifelse(predict(fit_over) > 0, "spam", "nonspam") != spam_trn$type)
```

Because of this, training data isn't useful for evaluating, as it would suggest that we should always use the largest possible model, when in reality, that model is likely overfitting. Recall, a model that is too complex will overfit. A model that is too simple will underfit. (We're looking for something in the middle.)

To overcome this, we'll use cross-validation as we did with ordinary linear regression, but this time we'll cross-validate the misclassification rate. To do so, we'll use the `cv.glm()` function from the `boot` library. It takes arguments for the data (in this case training), a model fit via `glm()`, and `K`, the number of folds. See `?cv.glm` for details.

Previously, for cross-validating RMSE in ordinary linear regression, we used LOOCV. We certainly could do that here. However, with logistic regression, we no longer have the clever trick that would allow use to obtain a LOOCV metric without needing to fit the model $n$ times. So instead, we'll use 5-fold cross-validation. (5 and 10 fold are the most common in practice.) Instead of leaving a single observation out repeatedly, we'll leave out a fifth of the data.

Essentially we'll repeat the following process 5 times:

- Randomly set aside a fifth of the data (each observation will only be held-out once)
- Train model on remaining data
- Evaluate misclassification rate on held-out data

The 5-fold cross-validated misclassification rate will be the average of these misclassification rates. By only needing to refit the model 5 times, instead of $n$ times, we will save a lot of computation time.

```{r, message=FALSE, warning=FALSE}
library(boot)
set.seed(1)
cv.glm(spam_trn, fit_caps, K = 5)$delta[1]
cv.glm(spam_trn, fit_selected, K = 5)$delta[1]
cv.glm(spam_trn, fit_additive, K = 5)$delta[1]
cv.glm(spam_trn, fit_over, K = 5)$delta[1]
```

Note that we're suppressing warnings again here. (Now there would be a lot more, since were fitting a total of 20 models.)

Based on these results, `fit_caps` and `fit_selected` are underfitting relative to `fit_additive`. Similarly, `fit_over` is overfitting relative to `fit_additive`. Thus, based on these results, we prefer the classifier created based on the logistic regression fit and stored in `fit_additive`.

Going forward, to evaluate and report on the efficacy of this classifier, we'll use the test dataset. We're going to take the position that the test data set should **never** be used in training, which is why we used cross-validation within the training dataset to select a model. Even though cross-validation uses hold-out sets to generate metrics, at some point all of the data is used for training.

To quickly summarize how well this classifier works, we'll create a confusion matrix.

![Confusion Matrix](images/confusion.png)

It further breaks down the classification errors into false positives and false negatives.

```{r}
make_conf_mat = function(predicted, actual) {
  table(predicted = predicted, actual = actual)
}
```

Let's explicitly store the predicted values of our classifier on the test dataset.

```{r}
spam_tst_pred = ifelse(predict(fit_additive, spam_tst) > 0,
                       "spam",
                       "nonspam")
spam_tst_pred = ifelse(predict(fit_additive, spam_tst, type = "response") > 0.5,
                       "spam",
                       "nonspam")
```

The previous two lines of code produce the same output, that is the same predictions, since

$$
\eta({\bf x}) = 0 \iff p({\bf x}) = 0.5
$$
Now we'll use these predictions to create a confusion matrix.

```{r}
(conf_mat_50 = make_conf_mat(predicted = spam_tst_pred, actual = spam_tst$type))
```

$$
\text{Prev} = \frac{\text{P}}{\text{Total Obs}}= \frac{\text{TP + FN}}{\text{Total Obs}}
$$

```{r}
table(spam_tst$type) / nrow(spam_tst)
```

First, note that to be a reasonable classifier, it needs to outperform the obvious classifier of simply classifying all observations to the majority class. In this case, classifying everything as non-spam for a test misclassification rate of `r as.numeric((table(spam_tst$type) / nrow(spam_tst))[2])`

Next, we can see that using the classifier create from `fit_additive`, only a total of $137 + 161 = 298$ from the total of 3601 email in the test set are misclassified. Overall, the accuracy in the test set it

```{r}
mean(spam_tst_pred == spam_tst$type)
```

In other words, the test misclassification is

```{r}
mean(spam_tst_pred != spam_tst$type)
```

This seems like a decent classifier...

However, are all errors created equal? In this case, absolutely note. The 137 non-spam emails that were marked as spam (false positives) are a problem. We can't allow important information, say, a job offer, miss our inbox and get sent to the spam folder. On the other hand, the 161 spam email that would make it to an inbox (false negatives) are easily dealt with, just delete them.

Instead of simply evaluating a classifier based on its misclassification rate (or accuracy), we'll define two additional metrics, sensitivity and specificity. Note that this are simply two of many more metrics that can be considered. The [Wikipedia page for sensitivity and specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity){target="_blank"} details a large number of metrics that can be derived form a confusion matrix.

**Sensitivity** is essentially the true positive rate. So when sensitivity is high, the number of false negatives is low.

$$
\text{Sens} = \text{True Positive Rate} = \frac{\text{TP}}{\text{P}} = \frac{\text{TP}}{\text{TP + FN}}
$$

Here we have an `R` function to calculate the sensitivity based on the confusion matrix. Note that this function is good for illustrative purposes, but is easily broken. (Think about what happens if there are no "positives" predicted.)

```{r}
get_sens = function(conf_mat) {
  conf_mat[2, 2] / sum(conf_mat[, 2])
}
```

**Specificity** is essentially the true negative rate. So when specificity is high, the number of false positives is low.

$$
\text{Spec} = \text{True Negative Rate} = \frac{\text{TN}}{\text{N}} = \frac{\text{TN}}{\text{TN + FP}}
$$

```{r}
get_spec =  function(conf_mat) {
  conf_mat[1, 1] / sum(conf_mat[, 1])
}
```

We calculate both based on the confusion matrix we had created for our classifier.

```{r}
get_sens(conf_mat_50)
get_spec(conf_mat_50)
```

Recall that we had created this classifier using a probability of $0.5$ as a "cutoff" for how observations should be classified. Now we'll modify this cutoff. We'll see that by modifying the cutoff, $c$, we can improve sensitivity or specificity at the expense of the overall accuracy (misclassification rate).

$$
\hat{C}(\bf x) =
\begin{cases}
      1 & \hat{p}({\bf x}) > c \\
      0 & \hat{p}({\bf x}) \leq c
\end{cases}
$$

Additionally, if we change the cutoff to improve sensitivity, we'll decrease specificity, and vice versa.

First let's see what happens when we lower the cutoff from $0.5$ to $0.1$ to create a new classifier, and thus new predictions.

```{r}
spam_tst_pred_10 = ifelse(predict(fit_additive, spam_tst, type = "response") > 0.1,
                          "spam",
                          "nonspam")
```

This is essentially *decreasing* the threshold for an email to be labeled as spam, so far *more* emails will be labeled as spam. We see that in the following confusion matrix.

```{r}
(conf_mat_10 = make_conf_mat(predicted = spam_tst_pred_10, actual = spam_tst$type))
```

Unfortunately, while this does greatly reduce false negatives, false positives have almost quadrupled. We see this reflected in the sensitivity and specificity.

```{r}
get_sens(conf_mat_10)
get_spec(conf_mat_10)
```

This classifier, using $0.1$ instead of $0.5$ has a higher sensitivity, but a much lower specificity. Clearly, we should have moved the cutoff in the other direction. Let's try $0.9$.

```{r}
spam_tst_pred_90 = ifelse(predict(fit_additive, spam_tst, type = "response") > 0.9,
                          "spam",
                          "nonspam")
```

This is essentially *increasing* the threshold for an email to be labeled as spam, so far *fewer* emails will be labeled as spam. Again, we see that in the following confusion matrix.

```{r}
(conf_mat_90 = make_conf_mat(predicted = spam_tst_pred_90, actual = spam_tst$type))
```

This is the result we're looking for. We have far fewer false positives. While sensitivity is greatly reduced, specificity has gone up.

```{r}
get_sens(conf_mat_90)
get_spec(conf_mat_90)
```

While this is far fewer false positives, is it acceptable though? Still probably not. Also, don't forget, this would actually be a terrible spam detector today since this is based on data from a very different era of the internet, for a very specific set of people. Spam has changed a lot since 90s! (Ironically, machine learning is probably partially to blame.)

This chapter has provided a rather quick introduction to classification, and thus, machine learning. For a more complete coverage of machine learning, [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/){target="_blank"} is a highly recommended resource. Additionally, [`R` for Statistical Learning](https://daviddalpiaz.github.io/r4sl/){target="_blank"} has been written as a supplement which provides additional detail on how to perform these methods using `R`. The [classification](https://daviddalpiaz.github.io/r4sl/classification-overview.html){target="_blank"} and [logistic regression](https://daviddalpiaz.github.io/r4sl/logistic-regression.html){target="_blank"} chapters might be useful.

We should note that the code to perform classification using logistic regression is presented in a way that illustrates the concepts to the reader. In practice, you may to prefer to use a more general machine learning pipeline such as [`caret`](http://topepo.github.io/caret/index.html){target="_blank"} in `R`. This will streamline processes for creating predictions and generating evaluation metrics.
