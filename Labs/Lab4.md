# Lab 4: Classification and Logistic Regression

In this lab, you will explore how to use logistic regression in to make binary predictions. Along the way, we will explore making predictions that use **cross validation**, a central idea in statistical learning that helps us get an idea of how well our model performs.

This lab is broken into two sections. The first section is instructive and walks the reader through some new techniques. The second section requires the reader to apply the techniques to the bank data we previously explored (see Lab 1, exercise 2).

The data you will need can be found [here](/data/bank.csv).

---


This section introduces using logistic regession to make "predictions." In this setting, we would call these predictions **classifications**. Put simply, the idea is as follows: based on the values of the predictors, should an observation be classified as $Y = 1$ or as $Y = 0$?

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

### `spam` Dataset

To illustrate the use of logistic regression as a classifier, we will use the `spam` dataset from the `kernlab` package. This dataset was introduced in the lecture online.

```{r}
# install.packages("kernlab")
library(kernlab)
data("spam")
tibble::as.tibble(spam)
```

This dataset, created in the late 1990s at Hewlett-Packard Labs, contains 4601 emails, of which 1813 are considered spam. The remaining are not spam. (Which for simplicity, we might call, ham.) Additional details can be obtained by using `?spam` of by visiting the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/spambase).

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

We'll fit four logistic regressions, each more complex than the previous.


### Evaluating Classifiers

The metric we'll be most interested in for evaluating the overall performance of a classifier is the **misclassification rate**. Sometimes, instead accuracy is reported, which is instead the proportion of correction classifications, so both metrics serve the same purpose. We've previously explored the classification rate during in-class exercises, so this should be a somewhat familiar exercise.

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

To overcome this, we'll use **cross-validation**. To do so, we'll use the `cv.glm()` function from the `boot` library. It takes arguments for the data (in this case, the *training* data), a model fit via `glm()`, and `K`, the number of folds. See `?cv.glm` for details.

To start, you'll use 5-fold cross-validation. (5 and 10 fold are the most common in practice.) Instead of leaving a single observation out repeatedly, we'll leave out a fifth of the data.

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

---
**Exercise 1 (first part)**

1. Execute the code above. Based on the results, rank the models from "most underfit" to "most overfit".

2. Re-run the code above with 100 folds and a different seed. Does your conclusion change?
---


To evaluate and report on the efficacy of this classifier, we'll use the test dataset.

### Critically, the test data set should **never** be used in training. NEVER. Never.

This is why we used cross-validation within the training dataset to select a model. Even though cross-validation uses hold-out sets to generate metrics, at some point all of the data is used for training.

To quickly summarize how well this classifier performs, we'll create a confusion matrix. We've previously done this. The code we used previously is below.

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

Note that to be a reasonable classifier, it needs to outperform the obvious classifier of simply classifying all observations to the majority class. In this case, classifying everything as non-spam is relatively effective.

Recall two definitions:

**Sensitivity** is essentially the true positive rate. So when sensitivity is high, the number of false negatives is low.

$$
\text{Sens} = \text{True Positive Rate} = \frac{\text{TP}}{\text{P}} = \frac{\text{TP}}{\text{TP + FN}}
$$


**Specificity** is essentially the true negative rate. So when specificity is high, the number of false positives is low.

$$
\text{Spec} = \text{True Negative Rate} = \frac{\text{TN}}{\text{N}} = \frac{\text{TN}}{\text{TN + FP}}
$$

---

**Exercise 1 (part 2)**

3. Generate four confusion matrices for each of the four models fit in Part 1.

4. Which is the best model? Write 2 paragraphs justifying your decision. You must mention (a) the overall accuracy of each model; and (b) whether some errors are better or worse than others, and you must use the terms *specificity* and *sensitivity*. For (b) think carefully... misclassified email is a pain in the butt for users!

---


---

**Exercise 2**

1. Use the [bank data](/data/bank.csv) and create a train / test split.

2. Run any logistic regression you like with 10-fold cross-validation in order to predict the `yes/no` variable (`y`).

3. Discuss the interpretation of the coefficients in your model. That is, you must write at least one sentence for each of the coefficients which describes how it is related to the response. You may use transformations of variables if you like. FAKE EXAMPLE: `age` has a positive coefficient, which means that older individuals are more likely to have `y = yes`.

4. Create a confusion matrix of your preferred model, evaluated against your *test* data. 

---
