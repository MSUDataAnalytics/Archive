<!-- slide -->

# Model Building

> "Statisticians, like artists, have the bad habit of falling in love with their models."
>
> --- **George Box**

In the next two lectures we will close our discussion of the linear regression. These lectures describe the following:

- How to think about the model selection process.
- Exploration versus Prediction.
- Common issues when practically implementing linear regression.

<!-- slide -->

Let's take a step back and consider the process of finding a model for data at a higher level. We are attempting to find a model for a response variable $y$ based on a number of predictors $x_1, x_2, x_3, \ldots, x_{p-1}$.

Essentially, we are trying to discover the functional relationship between $y$ and the predictors. In the previous lectures we were fitting models for a car's fuel efficiency (`mpg`) as a function of its attributes (`wt`, `year`, `cyl`, `disp`, `hp`, `acc`). We also consider $y$ to be a function of some noise. Rarely if ever do we expect there to be such an *exact* functional relationship between the predictors and the response. Nearly any function could occur:

\[
y = f(x_1, x_2, x_3, \ldots, x_{p-1}) + \epsilon
\]

We can think of this as

\[
\text{response} = \text{signal} + \text{noise}.
\]

<!-- slide -->

We *could* consider all sorts of complicated functions for $f$. We will encounter several ways of doing this under the general umbrella of **machine learning**. But so far in this course we have focused on (multiple) linear regression. That is

\[
\begin{aligned}
y &= f(x_1, x_2, x_3, \ldots, x_{p-1}) + \epsilon \\
  &= \beta_0 + \beta_1 x_{1} + \beta_2 x_{2} + \cdots + \beta_{p-1} x_{p-1} + \epsilon
\end{aligned}
\]

In the big picture of possible models that we could fit to this data, this is a rather restrictive model. What do we mean by a restrictive model?

<!-- slide -->


## Family, Form, and Fit

When modeling data, there are a number of choices that need to be made.

- What **family** of models will be considered?
- What **form** of the model will be used?
- How will the model be **fit**?

Let's work backwards and discuss each of these.

<!-- slide -->

### Fit

Consider one of the simplest models we could fit to data, simple linear regression.

\[
y = f(x_1, x_2, x_3, \ldots, x_{p-1}) + \epsilon = \beta_0 + \beta_1 x_{1} + \epsilon
\]

So here, despite having multiple predictors, we chose to use only one. How is this model **fit**? We will almost exclusively use the method of least squares, but recall, we had seen alternative methods of fitting this model.

\[
\underset{\beta_0, \beta_1}{\mathrm{argmin}} \max|y_i - (\beta_0 + \beta_1 x_i)|
\]

\[
\underset{\beta_0, \beta_1}{\mathrm{argmin}} \sum_{i = 1}^{n}|y_i - (\beta_0 + \beta_1 x_i)|
\]

\[
\underset{\beta_0, \beta_1}{\mathrm{argmin}} \sum_{i = 1}^{n}(y_i - (\beta_0 + \beta_1 x_i))^2
\]

<!-- slide -->


Any of these methods (we will always use the last, least squares) will obtain estimates of the unknown parameters $\beta_0$ and $\beta_1$. Since those are the only unknowns of the specified model, we have then *fit* the model. The fitted model is then

\[
\hat{y} = \hat{f}(x_1, x_2, x_3, \ldots, x_{p-1}) = \hat{\beta}_0 + \hat{\beta}_1 x_{1}
\]

Note that we have dropped the term for the noise. We didn't make any effort to model the noise, only the signal. We construct estimates such that the noise averages out to zero and we politely ignore it.

<!-- slide -->


### Form

What are the different **forms** a model could take? Currently, for the linear models we have considered, the only method for altering the form of the model is to control the predictors used. For example, one form of the multiple linear regression model is simple linear regression.

\[
y = f(x_1, x_2, x_3, \ldots, x_{p-1}) + \epsilon = \beta_0 + \beta_1 x_{1} + \epsilon
\]

We could also consider a SLR model with a different predictor, thus altering the form of the model.

\[
y = f(x_1, x_2, x_3, \ldots, x_{p-1}) + \epsilon = \beta_0 + \beta_2 x_{2} + \epsilon
\]

Often, we'll use multiple predictors in our model. Very often, we will at least try a model with all possible predictors.

\[
\begin{aligned}
y &= f(x_1, x_2, x_3, \ldots, x_{p-1}) + \epsilon \\
  &= \beta_0 + \beta_1 x_{1} + \beta_2 x_{2} + \cdots + \beta_{p-1} x_{p-1} + \epsilon
\end{aligned}
\]

We could also use some, but not all of the predictors.

\[
\begin{aligned}
y &= f(x_1, x_2, x_3, \ldots, x_{p-1}) + \epsilon \\
  &= \beta_0 + \beta_1 x_{1} + \beta_3 x_{3} + \beta_5 x_{5} + \epsilon
\end{aligned}
\]

<!-- slide -->

These forms are **restrictive** in two senses. First, they only allow for linear relationships between the response and the predictors. This seems like an obvious restriction of linear models, but in fact, we will soon see how to use linear models for *non-linear* relationships. (It will involve transforming variables.) Second, how one variable affects the response is the same for **any** values of the other predictors. Soon we will see how to create models where the effect of $x_{1}$ can be different for different values of $x_{2}$. We will discuss the concept of *interaction*.

<!-- slide -->


### Family

A **family** of models is a broader grouping of many possible *forms* of a model. For example, above we saw several forms of models from the family of linear models. In social science, we often only concern ourselves with linear models, which model a response as a linear combination of predictors. There are certainly other families of models. Broadly, the **family** defines our hypothesis set $\mathcal{H}$, which we will discuss more coming up shortly.

For examples of alternative familes, there are several families of *non-parametric* regression. *Smoothing* is a broad family of models. As are  *trees*.

In linear regression, we specified models with parameters, $\beta_j$ and fit the model by finding the best values of these parameters. This is a *parametric* approach. A non-parametric approach skips the step of specifying a model with parameters, and are often described by the algorithm they employ. Non-parametric models are used in statistical learning.

<!-- slide -->

### Example Code (Simulation)
```{r}
set.seed(42)
n = 30
beta_0 = 1
beta_1 = 0.5
sigma = 3
x = runif(n, 0, 10)
xplot = seq(0, 10, by = 0.1)

sim_slr = function(n, x, beta_0 = 10, beta_1 = 5, sigma = 1) {
  epsilon = rnorm(n, mean = 0, sd = sigma)
  y       = beta_0 + beta_1 * x + epsilon
  data.frame(predictor = x, response = y)
}

sim_data = sim_slr(n = n, x = x, beta_0 = beta_0, beta_1 = beta_1, sigma = sigma)

sim_fit_1 = lm(response ~ predictor, data = sim_data)
sim_fit_2 = loess(response ~ predictor, data = sim_data)

par(mfrow = c(1, 2))

plot(response ~ predictor, data = sim_data,
     xlab = "Predictor", ylab = "Response",
     main = "Linear Regression",
     pch  = 20, cex  = 3, col = "darkgrey",
     xlim = c(0, 10), ylim = c(-6, 11))
abline(sim_fit_1, lwd = 3, col = "dodgerblue")

plot(response ~ predictor, data = sim_data,
     xlab = "Predictor", ylab = "Response",
     main = "Smoothing",
     pch  = 20, cex  = 3, col = "darkgrey",
     xlim = c(0, 10), ylim = c(-6, 11))
lines(xplot, predict(sim_fit_2, newdata = data.frame(predictor = xplot)),
      lwd = 3, lty = 2, col = "darkorange")
```

<!-- slide -->


Here, linear regression (parametric) is used on the left, while smoothing (non-parametric) is used on the right. This simple linear regression finds the "best" slope and intercept. Smoothing produces the fitted $y$ value at a particular $x$ value by considering the $y$ values of the data in a neighborhood of the $x$ value considered. (Local smoothing.) Which is "better"?

And why the focus on **linear models**? Two big reasons:

- Linear models are **the** go-to model. Linear models have been around for a long time, and are computationally easy. A linear model may not be the final model you use, but often, it should be the first model you try.
- The ideas behind linear models can be easily transferred to other modeling techniques.

<!-- slide -->

### Assumed Model, Fitted Model

When searching for a model, we often need to make assumptions. These assumptions are codified in the **family** and **form** of the model. For example

\[
y = \beta_0 + \beta_1 x_{1} + \beta_3 x_{3} + \beta_5 x_{5} + \epsilon
\]

assumes that $y$ is a linear combination of $x_{1}$, $x_{3}$, and $x_{5}$ as well as some noise. This assumes that the effect of $x_{1}$ on $y$ is $\beta_1$, which is the same for all values of $x_{3}$ and $x_{5}$. That is, we are using the *family* of linear models with a particular *form*.


Suppose we then *fit* this model to some data and obtain the **fitted model**. For example, in `R` we would use

```{r}
fit = lm(y ~ x1 + x3 + x5, data = some_data)
```

<!-- slide -->
This is `R`'s way of saying the *family* is *linear* and specifying the *form* from above. An additive model with the specified predictors as well as an intercept. We then obtain

\[
\hat{y} = 1.5 + 0.9 x_{1} + 1.1 x_{3} + 2.3 x_{5}.
\]

This is our best guess for the function $f$ in

\[
y = f(x_1, x_2, x_3, \ldots, x_{p-1}) + \epsilon
\]

for the assumed **family** and **form**. Fitting a model only gives us the best fit for the family and form that we specify. So the natural question is; how do we choose the correct family and form? We'll focus on *form* since we are focusing on the *family* of linear models.

<!-- slide -->

## "Explanation" versus Prediction

What is the purpose of fitting a model to data? Usually it is to accomplish one of two goals. Social scientist often uses models to *explain* the relationship between the response and the predictors. Models can also be used to **predict** the response based on the predictors.

For reasons we've previously discussed, I discourage pretending like you're *explaning*. It's very hard to show causality. Instead, I will use the term **explore**. Thus we'll consider **exploring versus predicting**. We'll discuss both goals below, and note how the process of finding models for explaining and predicting have some differences.

For the moment---since we are only considering linear models---searching for a good model is essentially searching for a good **form** of a model.

<!-- slide -->

### Exploration

If the goal of a model is to explore the relationship between the response and the predictors, we are looking for a model that is **small** and **interpretable**, but still fits the data well. When discussing linear models, the **size** of a model is essentially the number of $\beta$ parameters used.

Suppose we would like to find a model that explains fuel efficiency (`mpg`) based on a car's attributes (`wt`, `year`, `cyl`, `disp`, `hp`, `acc`). Perhaps we are a car manufacturer trying to engineer a fuel efficient vehicle. If this is the case, we are interested in both which predictor variables are useful for explaining the car's fuel efficiency, as well as how those variables effect fuel efficiency. By getting a better understanding of this relationship (at least locally), we can use this knowledge to our advantage when designing a car.

To explain a relationship, we are interested in keeping models as small as possible, since smaller models are easy to interpret. The fewer predictors the less considerations we need to make in our design process.[^1]

[^1]: Note that *linear* models of any size are rather interpretable to begin with. Later in your data analysis careers, you will see more complicated models that may fit data better, but are much harder, if not impossible to interpret. These models aren't nearly as useful for explaining a relationship. This is another reason to always attempt a linear model. If it fits as well as more complicated methods, it will be the easiest to understand.

<!-- slide -->



To find small and interpretable models, we will eventually use selection procedures, which search among many possible forms of a model. For now we will do this in a more ad-hoc manner using **inference** techniques we have already encountered. To use inference as we have seen it, we need an additional assumption in addition to the family and form of the model.

\[
y = \beta_0 + \beta_1 x_{1} + \beta_3 x_{3} + \beta_5 x_{5} + \epsilon
\]

Our additional assumption is about the error term.

\[
\epsilon \sim N(0, \sigma^2)
\]

This assumption, that the errors are normally distributed with some common variance is the key to all of the inference we have done so far. We will discuss this in great detail later.

So with our inference tools (ANOVA and $t$-test) we have two potential strategies. Start with a very small model (no predictors) and attempt to add predictors. Or, start with a big model (all predictors) and attempt to remove predictors.

<!-- slide -->

#### Correlation and Causation

```{r}
autompg = read.table(
  "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
  quote = "\"",
  comment.char = "",
  stringsAsFactors = FALSE)
colnames(autompg) = c("mpg", "cyl", "disp", "hp", "wt", "acc", "year", "origin", "name")
autompg = subset(autompg, autompg$hp != "?")
autompg = subset(autompg, autompg$name != "plymouth reliant")
rownames(autompg) = paste(autompg$cyl, "cylinder", autompg$year, autompg$name)
autompg = subset(autompg, select = c("mpg", "cyl", "disp", "hp", "wt", "acc", "year"))
autompg$hp = as.numeric(autompg$hp)
```

A word of caution when using a model to **explore** a relationship. There are two terms often used to describe a relationship between two variables: *causation* and *correlation*. [Correlation](https://xkcd.com/552/) is often also referred to as association.

Just because two variables are correlated does not necessarily mean that one causes the other. For example, consider modeling `mpg` as only a function of `hp`.

```{r}
plot(mpg ~ hp, data = autompg, col = "dodgerblue", pch = 20, cex = 1.5)
```

<!-- slide -->

Does an increase in horsepower cause a drop in fuel efficiency? Or, perhaps the causality is reversed and an increase in fuel efficiency cause a decrease in horsepower. Or, perhaps there is a third variable that explains both!

The issue here is that we have **observational** data. With observational data, we can only detect *associations*. To speak with confidence about *causality*, we would need to run **experiments**. Often, this decision is made for us, before we ever see data, so we can only modify our interpretation.

This is a concept that you will encounter often in your professional lives. (Especially for you economists---does demand cause supply or vice versa?) For some further reading, and some related fallacies, see: [Wikipedia: Correlation does not imply causation](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation).

<!-- slide -->

### Prediction

If the goal of a model is to predict the response, then the **only** consideration is how well the model fits the data.

Correlation and causation are *not* an issue here. If a predictor is correlated with the response, it is useful for prediction. For example, in elementary school aged children their shoe size certainly doesn't *cause* them to read at a higher level, however we could very easily use `shoe size` to make a prediction about `reading ability`. The larger their shoe size, the better they read. There's a lurking variable here though: `age`.

Also, since we are not performing inference, assumptions about the errors are not needed. The only thing we care about is how close the fitted model is to the data. Least squares is least squares. For a specified model, it will find the values of the parameters which will minimize the squared error loss. Your results might be largely uninterpretable and useless for inference, but for prediction none of that matters.

We'll return to **prediction** in earnest in a few days.

<!-- slide -->

# Assessing Our Toolkit

The follow is a reasonably set of questions to ask about a model:
1. Is **at least one** of the predictors $X_1, X_2, . . . , X_p$ useful in predicting the response?
2. Do **all** the predictors help to explain $Y$ , or is **only a subset** of the predictors useful?
3. **How well** does the model fit the data?
4. Given a set of predictor values, what response value should we **predict**, and **how accurate** is our prediction?

<!-- slide -->

## Is $X$ useful in predicting $Y$?

A natural question is whether at least one of the predictors is useful in capturing the variation in $Y$.

As we previously explored, perhaps the most straightforward way to address this question is with the $F$ test.

\[
F = \frac{\frac{(TSS − RSS)}{p}}{\frac{RSS}{(n−p−1)}} \sim F_{p,n−p−1}
\]

Written this way, we can loosely think of this test as the explained variation (numerator) normalized to the by the total variation (denominator). But this isn't exactly right, so let's be careful with such interpretations.

<!-- slide -->

## Could we merely use a subset of predictors?

This is a complicated question and a very common one in social science.

- The most direct approach is called **all subsets** or **best subsets regression**: we compute the least squares fit for all possible subsets and then choose between them based on some criterion that balances training error with model size.
- However we often can’t examine all possible models, since they are $2^p$ of them; for example when $p = 40$ there are over a billion potential models we could explore.
- Instead we need an automated approach that searches through a subset of them. We discuss two commonly use approaches next.

<!-- slide -->

### Forward Selection
1. Begin with the null model — a model that contains an intercept but no predictors.

2. Fit $p$ simple linear regressions and add to the null model the variable that results in the lowest RSS. You now have a simple linear regression.

3. Add to that model the variable that results in the lowest RSS amongst all two-variable models. (So now your model is given by $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \epsilon$.)

4. Continue adding variables one at a time until some stopping rule is satisfied.
  - This part is pretty ad-hoc. For example, people often adopt the rule that they will stop when all remaining variables have a $p$-value above some threshold. But anything is valid.

<!-- slide -->

### Backward Selection

- Start with all variables in the model.
- Remove the variable with the largest p-value — that is, the
variable that is the least statistically significant.
- The new $(p − 1)$-variable model is fit, and the variable with the **largest** $p$-value is removed.
- Continue until a stopping rule is reached. As before, this is an ad-hoc rule. For instance, we may stop when all remaining variables have a significant $p$-value defined by some significance threshold.

Later we will discuss more systematic criteria for choosing an "optimal" member in the subset of models produced by forward or backward stepwise selection. Moreover, we'll introduce formal mathematical models that automate this process.

<!-- slide -->

## How Well Does Our model Fit?

The **coefficient of determination**, $R^2$, is defined as

$$
\begin{aligned}
R^2 &= \frac{\text{SSReg}}{\text{SST}} = \frac{\sum_{i=1}^{n}(\hat{y}_i - \bar{y})^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \\[2.5ex]
    &= \frac{\text{SST} - \text{SSE}}{\text{SST}} = 1 - \frac{\text{SSE}}{\text{SST}} \\[2.5ex]
    &= 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} =
1 - \frac{\sum_{i = 1}^{n}e_i^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
\end{aligned}
$$

The coefficient of determination is interpreted as the proportion of observed variation in $y$ that can be explained by the simple linear regression model.

```{r}
R2 = SSReg / SST
R2
```

<!-- slide -->

For the `cars` example, we'd say that $R^2*100$ percent of the observed variability in stopping distance is explained by the linear relationship with speed. The following code visually demonstrates the three "sums of squares" for a simulated dataset which has a somewhat high $R^2$ value. It also demonstrates defining functions (a notion that is hopefully familiar but likely not).

```{r}
generate_data = function(int = 1,
                         slope = 2,
                         sigma = 5,
                         n_obs = 15,
                         x_min = 0,
                         x_max = 10) {
  x = seq(x_min, x_max, length.out = n_obs)
  y = int + slope * x + rnorm(n_obs, 0, sigma)
  fit = lm(y ~ x)
  y_hat = fitted(fit)
  y_bar = rep(mean(y), n_obs)
  data.frame(x, y, y_hat, y_bar)
}

plot_total_dev = function(reg_data) {
  plot(reg_data$x, reg_data$y, main = "SST (Sum of Squares Total)",
       xlab = "x", ylab = "y", pch = 20, cex = 2, col = "grey")
  arrows(reg_data$x, reg_data$y_bar,
         reg_data$x, reg_data$y,
         col = 'grey', lwd = 1, lty = 3, length = 0.2, angle = 20)
  abline(h = mean(reg_data$y), lwd = 2,col = "grey")
  # abline(lm(y ~ x, data = reg_data), lwd = 2, col = "grey")
}

plot_total_dev_prop = function(reg_data) {
  plot(reg_data$x, reg_data$y, main = "SST (Sum of Squares Total)",
       xlab = "x", ylab = "y", pch = 20, cex = 2, col = "grey")
  arrows(reg_data$x, reg_data$y_bar,
         reg_data$x, reg_data$y_hat,
         col = 'darkorange', lwd = 1, length = 0.2, angle = 20)
  arrows(reg_data$x, reg_data$y_hat,
         reg_data$x, reg_data$y,
         col = 'dodgerblue', lwd = 1, lty = 2, length = 0.2, angle = 20)
  abline(h = mean(reg_data$y), lwd = 2,col = "grey")
  abline(lm(y ~ x, data = reg_data), lwd = 2, col = "grey")
}
```
<!-- slide -->

### Code Example (Cont.)
```{r}
plot_unexp_dev = function(reg_data) {
  plot(reg_data$x, reg_data$y, main = "SSE (Sum of Squares Error)",
       xlab = "x", ylab = "y", pch = 20, cex = 2, col = "grey")
  arrows(reg_data$x, reg_data$y_hat,
         reg_data$x, reg_data$y,
         col = 'dodgerblue', lwd = 1, lty = 2, length = 0.2, angle = 20)
  abline(lm(y ~ x, data = reg_data), lwd = 2, col = "grey")
}

plot_exp_dev = function(reg_data) {
  plot(reg_data$x, reg_data$y, main = "SSReg (Sum of Squares Regression)",
  xlab = "x", ylab = "y", pch = 20, cex = 2, col = "grey")
  arrows(reg_data$x, reg_data$y_bar,
         reg_data$x, reg_data$y_hat,
         col = 'darkorange', lwd = 1, length = 0.2, angle = 20)
  abline(lm(y ~ x, data = reg_data), lwd = 2, col = "grey")
  abline(h = mean(reg_data$y), col = "grey")
}
```

<!-- slide -->

```{r}
set.seed(2)
plot_data = generate_data(sigma = 2)
par(mfrow = c(2, 2))
plot_exp_dev(plot_data)
plot_unexp_dev(plot_data)
plot_total_dev(plot_data)
plot_total_dev_prop(plot_data)
```

```{r}
set.seed(1)
plot_data = generate_data(slope = -1.5, sigma = 10)
```

The next plots again visually demonstrate the three "sums of squares" with a different $R^2$. Notice in the final plot, that now the blue arrows account for a larger proportion of the total arrow.

```{r}
par(mfrow = c(2, 2))
plot_exp_dev(plot_data)
plot_unexp_dev(plot_data)
plot_total_dev(plot_data)
plot_total_dev_prop(plot_data)
```




<!-- slide -->

## Data Manipulation for Linear Regressions

**Qualitative Predictors**
- Some predictors are not *quantitative* but are *qualitative*. That is, they take on a discrete set of values, which may include strings or data types.
- These are also called **categorical predictors** or **factor** variables.
- You have explored these a bit already. See, for instance `marital`, `education`, or `job` in the bank data from the previous lecture and Lab 1.
- Such variables need to be handled very carefully. The next few slides will give you some guidance, but this problem is general and will crop up for many different types of models that we will explore in this course.

<!-- slide -->

Consider the following exercise. Suppose we are endowed with data from a major national bank who is curious about credit card utilization amongst different generations. Specifically, we want to consider how credit card debt differs between Boomers and Millenials. They've (helpfully) generated a qualitative variable with these two words. That is, the data itself is a `string` which takes only two values.

We'll create a new variable:
\[\mathbb{D}_i = \begin{cases} 1 &\text{~ if the $i$th person is a boomer} \\ 0 &\text{~if the $i$th person is not a boomer} \end{cases}    \]

Now we can run the regression:
\[
y_i = \beta_0 + \beta_1 \mathbb{D_i} + \epsilon
\]

<!-- slide -->

The resulting model is given by
\[
y_i = \beta_0 + \beta_1 \mathbb{D_i} + \epsilon = \begin{cases} \beta_0 + \epsilon &\text{~ if the $i$th person is not a boomer} \\ \beta_0 + \beta_1 + \epsilon &\text{~ if the $i$th person is a boomer} \end{cases}
\]

How do we interpret this output? **Simple!**

- This simple model gives us the *average* credit card debt $y$ for each of the two groups. However, it's not quite as simple as just staring at the numbers.
  1. The average credit card debt of non-boomers is given by $\beta_0$. (**Try it:** Verify this is true by construction and using the regression equations.)
  2. The average credit card debt of boomers is given by $\beta_0 + \beta_1$.

This example helps show why thinking about things carefully matters a great deal. For example, this helps illustrate a natural statistical test is $H_0: \beta_1 = 0$.

<!-- slide -->

### Qualitative Variables Cont.

If our original data only had boomers and millenials, we could use the $\beta_0 + \beta_1$ as the mean debt of millenials. Of course, qualitative variables often take on more than two levels, and our data likely would have multiple levels. We can manually create additional "dummy" variables to account for each of these. For example, let's imagine that our credit card data has only births after 1950. Thus, for the variable we have been describing, a natural encoding would also include "Gen X" and "Gen Z". We can create two **more** dummy variables to account for this:
\[
\mathbb{D}_{i2} = \begin{cases} 1 & \text{if the $i$th person is Gen X} \\0 & \text{if the $i$th person is not Gen X} \end{cases}
\]
and
\[\mathbb{D}_{i3} = \begin{cases} 1 &\text{~ if the $i$th person is a millenial} \\ 0 &\text{~if the $i$th person is not a millenial} \end{cases}    \]

<!-- slide -->

Our final regression would include all of these dummies:
\[
y_i = \beta_0 + \beta_1 \mathbb{D}_{i1} + \beta_2 \mathbb{D}_{i2}+ \beta_3 \mathbb{D}_{i3} + \epsilon = \begin{cases} \beta_0 + \epsilon &\text{~ if the $i$th person is Gen Z} \\ \beta_0 + \beta_1 + \epsilon &\text{~ if the $i$th person is a boomer} \\ \beta_0 + \beta_1 + \beta_2 + \epsilon &\text{~ if the $i$th person is Gen Z} \\ \beta_0 + \beta_1 + \beta_2 + \beta_3 + \epsilon &\text{~ if the $i$th person is a millenial}\end{cases}
\]

This highlights two important facts:
1. The number of dummies is always one fewer than the number of values that the factor variable can take. The intercept captures the mean of this *baseline* group. In this example, `GenZ` is the baseline.
2. The order that you encode these variables can either aid or harm your interpretation; you should do this carefully.

<!-- slide -->

# Common Problems in Linear Regression
When we fit a linear regression model to a particular data set, many problems can occur. Some violations of the underlying assumptions are benign. Others are a bigger issue. (These are discussed in the text as Chapter 3.3.3). A few common issues are the following:
1. Non-linearity of the response-predictor relationships.
2. Correlation of error terms.
3. Non-constant variance of error terms.
4. Outliers.
5. High-leverage points.
6. Collinearity.

<!-- slide -->

## Non-Linearity

This is a deep issue and one that linear regression is limited in its ability to handle. As previously discussed, linear regression restricts the **family** that we utilize. We will describe a practical solution to this issue, but it introduces conceptual problems. To highlight the conceptual difficulties, let's remind ourselves that the underlying (unknown) relationship is given by:
\[
Y = f(X) + \epsilon
\]

Accordingly, it's unreasonable to expect a linear relationship. However, the Taylor's Theorem---and its associated series approximation---suggests that we might be able to locally represent $f$ by some linear function. (See [here](https://en.wikipedia.org/wiki/Taylor%27s_theorem) for the calculus.)

But we should be very wary of interpreting much from this exercise. If we find that the regression
\[
y = \beta_0 + \beta_1 x^4 + \epsilon
\]
yields a significant positive $\beta_1$, that does **not** mean that this is the true relationship. Merely, this polynomial does a reasonable job of approximating the underlying truth (at least locally).

<!-- slide -->
