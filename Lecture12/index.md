<!-- slide -->

# Logistic Regression

> "The line between disorder and order lies in logistics."
>
> - **Sun Tzu** (speaking about a decidedly different type of logistics)

After reading this chapter you will be able to:

- Understand how generalized linear models are a generalization of ordinary linear models.
- Use logistic regression to model a binary response.
- Apply concepts learned for ordinary linear models to logistic regression.
- Use logistic regression to perform classification.

<!-- slide -->

**Simple motivating question** Can we use linear models to deal with categorical variables?

Yes!

The model that we previously covered, which we now call *ordinary linear regression*, is actually a specific case of the more general, *generalized linear model*. (Aren't statisticians great at naming things?)

<!-- slide -->

## Generalized Linear Models

So far, we've had response variables that, conditioned on the predictors, were modeled using a normal distribution with a mean that is some linear combination of the predictors. This linear combination is what made a linear model "linear."

$$
Y \mid {\bf X} = {\bf x} \sim N(\beta_0 + \beta_1x_1 + \ldots + \beta_{p - 1}x_{p - 1}, \ \sigma^2)
$$

Now we'll allow for two modifications of this situation, which will let us use linear models in many more situations. Instead of using a normal distribution for the response conditioned on the predictors, we'll allow for other distributions. Also, instead of the conditional mean being a linear combination of the predictors, it can be some function of a linear combination of the predictors.

<!-- slide -->

In *general*, a generalized linear model has three parts:

- A **distribution** of the response conditioned on the predictors. (Technically this distribution needs to be from the [exponential family](https://en.wikipedia.org/wiki/Exponential_family) of distributions.)
- A **linear combination** of the $p - 1$ predictors, $\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots  + \beta_{p - 1} x_{p - 1}$, which we write as $\eta({\bf x})$. That is,

$$\eta({\bf x}) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots  + \beta_{p - 1} x_{p - 1}$$

- A **link** function, $g()$, that defines how $\eta({\bf x})$, the linear combination of the predictors, is related to the mean of the response conditioned on the predictors, $\text{E}[Y \mid {\bf X} = {\bf x}]$.

$$
\eta({\bf x}) = g\left(\text{E}[Y \mid {\bf X} = {\bf x}]\right).
$$

<!-- slide -->

The following table summarizes three examples of a generalized linear model:

|                 |Linear Regression | Poisson Regression | Logistic Regression |
|-----------------|------------------|--------------------|---------------------|
| $Y \mid {\bf X} = {\bf x}$ | $N(\mu({\bf x}), \sigma^2)$    | $\text{Pois}(\lambda({\bf x}))$          | $\text{Bern}(p({\bf x}))$                                              |
| **Distribution Name**                           | Normal                         | Poisson                                  | Bernoulli (Binomial)                                                   |
| $\text{E}[Y \mid {\bf X} = {\bf x}]$            | $\mu({\bf x})$                 | $\lambda({\bf x})$                       | $p({\bf x})$                                                           |
| **Support**                                     | Real: $(-\infty, \infty)$      | Integer: $0, 1, 2, \ldots$               | Integer: $0, 1$                                                        |
| **Usage**                                       | Numeric Data                   | Count (Integer) Data                     | Binary (Class ) Data                                            |
| **Link Name**                                   | Identity                       | Log                                      | Logit                                                                  |
| **Link Function**                               | $\eta({\bf x}) = \mu({\bf x})$ | $\eta({\bf x}) = \log(\lambda({\bf x}))$ | $\eta({\bf x}) = \log \left(\frac{p({\bf x})}{1 - p({\bf x})} \right)$          |
| **Mean Function**                               | $\mu({\bf x}) = \eta({\bf x})$ | $\lambda({\bf x}) = e^{\eta({\bf x})}$   | $p({\bf x}) = \frac{e^{\eta({\bf x})}}{1 + e^{\eta({\bf x})}} = \frac{1}{1 + e^{-\eta({\bf x})}}$ |

<!-- slide -->

Like ordinary linear regression, we will seek to "fit" the model by estimating the $\beta$ parameters. To do so, we will use the method of maximum likelihood.

Note that a Bernoulli distribution is a specific case of a binomial distribution where the $n$ parameter of a binomial is $1$. Binomial regression is also possible, but we'll focus on the much more popular Bernoulli case.

So, in general, GLMs relate the mean of the response to a linear combination of the predictors, $\eta({\bf x})$, through the use of a link function, $g()$. That is,

$$
\eta({\bf x}) = g\left(\text{E}[Y \mid {\bf X} = {\bf x}]\right).
$$

The mean is then

$$
\text{E}[Y \mid {\bf X} = {\bf x}] = g^{-1}(\eta({\bf x})).
$$

<!-- slide -->

## Binary Response

To illustrate the use of a GLM we'll focus on the case of binary responses variable coded using $0$ and $1$. In practice, these $0$ and $1$s will code for two classes such as yes/no, cat/dog, sick/healthy, etc.

$$
Y =
\begin{cases}
      1 & \text{yes} \\
      0 & \text{no}
\end{cases}
$$

First, we define some notation that we will use throughout.

$$
p({\bf x}) = P[Y = 1 \mid {\bf X} = {\bf x}]
$$

With a binary (Bernoulli) response, we'll mostly focus on the case when $Y = 1$, since with only two possibilities, it is trivial to obtain probabilities when $Y = 0$.

$$
P[Y = 0 \mid {\bf X} = {\bf x}] + P[Y = 1 \mid {\bf X} = {\bf x}] = 1
$$

$$
P[Y = 0 \mid {\bf X} = {\bf x}] = 1 - p({\bf x})
$$

<!-- slide -->

We now define the **logistic regression** model.

$$
\log\left(\frac{p({\bf x})}{1 - p({\bf x})}\right) = \beta_0 + \beta_1 x_1 + \ldots  + \beta_{p - 1} x_{p - 1}
$$

Immediately we notice some similarities to ordinary linear regression, in particular, the right hand side. This is our usual linear combination of the predictors. We have our usual $p - 1$ predictors for a total of $p$ $\beta$ parameters. (Note, many more machine learning focused texts will use $p$ as the number of parameters. This is an arbitrary choice, but you should be aware of it.)

The left hand side is called the **log odds**, which is the log of the odds. The odds are the probability for a positive event $(Y = 1)$ divided by the probability of a negative event $(Y = 0)$. So when the odds are $1$, the two events have equal probability. Odds greater than $1$ favor a positive event. The opposite is true when the odds are less than $1$.

$$
\frac{p({\bf x})}{1 - p({\bf x})} = \frac{P[Y = 1 \mid {\bf X} = {\bf x}]}{P[Y = 0 \mid {\bf X} = {\bf x}]}
$$

Essentially, the log odds are the [logit](https://en.wikipedia.org/wiki/Logit) transform applied to $p({\bf x})$.

$$
\text{logit}(\xi) = \log\left(\frac{\xi}{1 - \xi}\right)
$$

<!-- slide -->

It will also be useful to define the inverse logit, otherwise known as the "logistic" or [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function.

$$
\text{logit}^{-1}(\xi) = \frac{e^\xi}{1 + e^{\xi}} = \frac{1}{1 + e^{-\xi}}
$$

Note that for $x \in (-\infty, \infty))$, this function outputs values between 0 and 1.

Students often ask, where is the error term? The answer is that its something that is specific to the normal model. First notice that the model with the error term,

$$
Y = \beta_0 + \beta_1x_1 + \ldots + \beta_qx_q + \epsilon, \ \ \epsilon \sim N(0, \sigma^2)
$$
can instead be written as

$$
Y \mid {\bf X} = {\bf x} \sim N(\beta_0 + \beta_1x_1 + \ldots + \beta_qx_q, \ \sigma^2).
$$

<!-- slide -->

While our main focus is on estimating the mean, $\beta_0 + \beta_1x_1 + \ldots + \beta_qx_q$, there is also another parameter, $\sigma^2$ which needs to be estimated. This is the result of the normal distribution having two parameters.

With logistic regression, which uses the Bernoulli distribution, we only need to estimate the Bernoulli distribution's single parameter $p({\bf x})$, which happens to be its mean.

$$
\log\left(\frac{p({\bf x})}{1 - p({\bf x})}\right) = \beta_0 + \beta_1 x_1 + \ldots  + \beta_{q} x_{q}
$$

So even though we introduced ordinary linear regression first, in some ways, logistic regression is actually simpler.

Note that applying the inverse logit transformation allow us to obtain an expression for $p({\bf x})$.

$$
p({\bf x}) = P[Y = 1 \mid {\bf X} = {\bf x}] = \frac{e^{\beta_0 + \beta_1 x_{1} + \cdots + \beta_{p-1} x_{(p-1)}}}{1 + e^{\beta_0 + \beta_1 x_{1} + \cdots + \beta_{p-1} x_{(p-1)}}}
$$

<!-- slide -->

### Fitting Logistic Regression

With $n$ observations, we write the model indexed with $i$ to note that it is being applied to each observation.

$$
\log\left(\frac{p({\bf x_i})}{1 - p({\bf x_i)})}\right) = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_{p-1} x_{i(p-1)}
$$

We can apply the inverse logit transformation to obtain $P[Y_i = 1 \mid {\bf X_i} = {\bf x_i}]$ for each observation. Since these are probabilities, it's good that we used a function that returns values between $0$ and $1$.

$$
p({\bf x_i}) = P[Y_i = 1 \mid {\bf X_i} = {\bf x_i}] = \frac{e^{\beta_0 + \beta_1 x_{i1} + \cdots + \beta_{p-1} x_{i(p-1)}}}{1 + e^{\beta_0 + \beta_1 x_{i1} + \cdots + \beta_{p-1} x_{i(p-1)}}}
$$

$$
1 - p({\bf x_i}) = P[Y_i = 0 \mid {\bf X} = {\bf x_i}] = \frac{1}{1 + e^{\beta_0 + \beta_1 x_{i1} + \cdots + \beta_{p-1} x_{i(p-1)}}}
$$

<!-- slide -->

To "fit" this model, that is estimate the $\beta$ parameters, we will use maximum likelihood.

$$
\boldsymbol{{\beta}} = [\beta_0, \beta_1, \beta_2, \beta_3, \ldots, \beta_{p - 1}]
$$

We first write the likelihood given the observed data.

$$
L(\boldsymbol{{\beta}}) = \prod_{i = 1}^{n} P[Y_i = y_i \mid {\bf X_i} = {\bf x_i}]
$$

This is already technically a function of the $\beta$ parameters, but we'll do some rearrangement to make this more explicit.[^1]

$$
L(\boldsymbol{{\beta}}) = \prod_{i = 1}^{n} p({\bf x_i})^{y_i} (1 - p({\bf x_i}))^{(1 - y_i)}
$$

$$
L(\boldsymbol{{\beta}}) = \prod_{i : y_i = 1}^{n} p({\bf x_i}) \prod_{j : y_j = 0}^{n} (1 - p({\bf x_j}))
$$

$$
L(\boldsymbol{{\beta}}) = \prod_{i : y_i = 1}^{} \frac{e^{\beta_0 + \beta_1 x_{i1} + \cdots + \beta_{p-1} x_{i(p-1)}}}{1 + e^{\beta_0 + \beta_1 x_{i1} + \cdots + \beta_{p-1} x_{i(p-1)}}} \prod_{j : y_j = 0}^{} \frac{1}{1 + e^{\beta_0 + \beta_1 x_{j1} + \cdots + \beta_{p-1} x_{j(p-1)}}}
$$

[^1]: Unfortunately, unlike ordinary linear regression, there is no analytical solution for this maximization problem. Instead, it will need to be solved numerically. Fortunately, `R` will take care of this for us using an iteratively reweighted least squares algorithm. (We'll leave the details for a machine learning or optimization course, which would likely also discuss alternative optimization strategies.)

<!-- slide -->

### Fitting Issues

We should note that, if there exists some $\beta^*$ such that

$$
{\bf x_i}^{\top} \boldsymbol{{\beta}^*} > 0 \implies y_i = 1
$$

and

$$
{\bf x_i}^{\top} \boldsymbol{{\beta}^*} < 0 \implies y_i = 0
$$

for all observations, then the MLE is not unique. Such data is said to be separable.[^2]

[^2]: When this happens, the model is still "fit," but there are consequences, namely, the estimated coefficients are highly suspect. This is an issue when then trying to interpret the model. When this happens, the model will often still be useful for creating a classifier, which will be discussed later. However, it is still subject to the usual evaluations for classifiers to determine how well it is performing. For details, see [Modern Applied Statistics with S-PLUS, Chapter 7](https://link.springer.com/content/pdf/10.1007/978-1-4757-2719-7_7.pdf)

This, and similar numeric issues related to estimated probabilities near 0 or 1, will return a warning in `R`:

```{r}
message("Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred")
```

<!-- slide -->

### Simulation Examples

```{r}
sim_logistic_data = function(sample_size = 25, beta_0 = -2, beta_1 = 3) {
  x = rnorm(n = sample_size)
  eta = beta_0 + beta_1 * x
  p = 1 / (1 + exp(-eta))
  y = rbinom(n = sample_size, size = 1, prob = p)
  data.frame(y, x)
}
```

<!-- slide -->

You might think, why not simply use ordinary linear regression? Even with a binary response, our goal is still to model (some function of) $\text{E}[Y \mid {\bf X} = {\bf x}]$.

With a binary response coded as $0$ and $1$, this implies:

$\text{E}[Y \mid {\bf X} = {\bf x}] = P[Y = 1 \mid {\bf X} = {\bf x}]$ since

$$
\begin{aligned}
\text{E}[Y \mid {\bf X} = {\bf x}] &=  1 \cdot P[Y = 1 \mid {\bf X} = {\bf x}] + 0 \cdot P[Y = 0 \mid {\bf X} = {\bf x}] \\
                                  &= P[Y = 1 \mid {\bf X} = {\bf x}]
\end{aligned}
$$

<!-- slide -->


Then why can't we just use ordinary linear regression to estimate $\text{E}[Y \mid {\bf X} = {\bf x}]$, and thus $P[Y = 1 \mid {\bf X} = {\bf x}]$?

To investigate, let's simulate data from the following model:

$$
\log\left(\frac{p({\bf x})}{1 - p({\bf x})}\right) = -2 + 3 x
$$

Another way to write this, which better matches the function we're using to simulate the data:

$$
\begin{aligned}
Y_i \mid {\bf X_i} = {\bf x_i} &\sim \text{Bern}(p_i) \\
p_i &= p({\bf x_i}) = \frac{1}{1 + e^{-\eta({\bf x_i})}} \\
\eta({\bf x_i}) &= -2 + 3 x_i
\end{aligned}
$$

<!-- slide -->


```{r}
set.seed(1)
example_data = sim_logistic_data()
head(example_data)
```

After simulating a dataset, we'll then fit both ordinary linear regression and logistic regression. Notice that currently the responses variable `y` is a numeric variable that only takes values `0` and `1`. Later we'll see that we can also fit logistic regression when the response is a factor variable with only two levels. (Generally, having a factor response is preferred, but having a dummy response allows use to make the comparison to using ordinary linear regression.)

```{r}
# ordinary linear regression
fit_lm  = lm(y ~ x, data = example_data)
# logistic regression
fit_glm = glm(y ~ x, data = example_data, family = binomial)
```

<!-- slide -->

Notice that the syntax is extremely similar. What's changed?

- `lm()` has become `glm()`
- We've added `family = binomial` argument

In a lot of ways, `lm()` is just a more specific version of `glm()`. For example

```{r}
glm(y ~ x, data = example_data)
```

would actually fit the ordinary linear regression that we have seen in the past. By default, `glm()` uses `family = gaussian` argument. That is, we're fitting a GLM with a normally distributed response and the identity function as the link.

<!-- slide -->

The `family` argument to `glm()` actually specifies both the distribution and the link function. If not made explicit, the link function is chosen to be the **canonical link function**, which is essentially the most mathematical convenient link function. See `?glm` and `?family` for details. For example, the following code explicitly specifies the link function which was previously used by default.

```{r}
# more detailed call to glm for logistic regression
fit_glm = glm(y ~ x, data = example_data, family = binomial(link = "logit"))
```
<!-- slide -->



Making predictions with an object of type  `glm` is slightly different than making predictions after fitting with `lm()`. In the case of logistic regression, with `family = binomial`, we have:

| `type`             | Returned |
|--------------------|----------|
| `"link"` [default] | $\hat{\eta}({\bf x}) = \log\left(\frac{\hat{p}({\bf x})}{1 - \hat{p}({\bf x})}\right)$ |
| `"response"`       | $\hat{p}({\bf x}) = \frac{e^{\hat{\eta}({\bf x})}}{1 + e^{\hat{\eta}({\bf x})}} = \frac{1}{1 + e^{-\hat{\eta}({\bf x})}}$                                                                     |

That is, `type = "link"` will get you the log odds, while `type = "response"` will return the estimated mean, in this case, $P[Y = 1 \mid {\bf X} = {\bf x}]$ for each observation.

<!-- slide -->

# ![Lecture12-1](/assets/logit1.png)

<!-- slide -->


```{r}
plot(y ~ x, data = example_data,
     pch = 20, ylab = "Estimated Probability",
     main = "Ordinary vs Logistic Regression")
grid()
abline(fit_lm, col = "darkorange")
curve(predict(fit_glm, data.frame(x), type = "response"),
      add = TRUE, col = "dodgerblue", lty = 2)
legend("topleft", c("Ordinary", "Logistic", "Data"), lty = c(1, 2, 0),
       pch = c(NA, NA, 20), lwd = 2, col = c("darkorange", "dodgerblue", "black"))
```

<!-- slide -->


Since we only have a single predictor variable, we are able to graphically show this situation. First, note that the data, is plotted using black dots. The response `y` only takes values `0` and `1`.

The solid orange line, is the fitted ordinary linear regression.

The dashed blue curve is the estimated logistic regression. It is helpful to realize that we are not plotting an estimate of $Y$ for either. (Sometimes it might seem that way with ordinary linear regression, but that isn't what is happening.) For both, we are plotting $\hat{\text{E}}[Y \mid {\bf X} = {\bf x}]$, the estimated mean, which for a binary response happens to be an estimate of $P[Y = 1 \mid {\bf X} = {\bf x}]$.

We immediately see why ordinary linear regression is not a good idea. While it is estimating the mean, we see that it produces estimates that are less than 0! (And in other situations could produce estimates greater than 1!) If the mean is a probability, we don't want probabilities less than 0 or greater than 1.

<!-- slide -->

Enter logistic regression. Since the output of the inverse logit function is restricted to be between 0 and 1, our estimates make much more sense as probabilities. Let's look at our estimated coefficients. (With a lot of rounding, for simplicity.)

```{r}
round(coef(fit_glm), 1)
```

Our estimated model is then:

$$
\log\left(\frac{\hat{p}({\bf x})}{1 - \hat{p}({\bf x})}\right) = -2.3 + 3.7 x
$$

Because we're not directly estimating the mean, but instead a function of the mean, we need to be careful with our interpretation of $\hat{\beta}_1 = 3.7$. This means that, for a one unit increase in $x$, the log odds change (in this case increase) by $3.7$. Also, since $\hat{\beta}_1$ is positive, as we increase $x$ we also increase $\hat{p}({\bf x})$. To see how much, we have to consider the inverse logistic function.

<!-- slide -->

For example, we have:

$$
\hat{P}[Y = 1 \mid X = -0.5] = \frac{e^{-2.3 + 3.7 \cdot (-0.5)}}{1 + e^{-2.3 + 3.7 \cdot (-0.5)}} \approx 0.016
$$

$$
\hat{P}[Y = 1 \mid X = 0] = \frac{e^{-2.3 + 3.7 \cdot (0)}}{1 + e^{-2.3 + 3.7 \cdot (0)}} \approx 0.09112296
$$

$$
\hat{P}[Y = 1 \mid X = 1] = \frac{e^{-2.3 + 3.7 \cdot (1)}}{1 + e^{-2.3 + 3.7 \cdot (1)}} \approx 0.8021839
$$

<!-- slide -->

Now that we know we should use logistic regression, and not ordinary linear regression, let's consider another example. This time, let's consider the model

$$
\log\left(\frac{p({\bf x})}{1 - p({\bf x})}\right) = 1 + -4 x.
$$

Again, we could re-write this to better match the function we're using to simulate the data:

$$
\begin{aligned}
Y_i \mid {\bf X_i} = {\bf x_i} &\sim \text{Bern}(p_i) \\
p_i &= p({\bf x_i}) = \frac{1}{1 + e^{-\eta({\bf x_i})}} \\
\eta({\bf x_i}) &= 1 + -4 x_i
\end{aligned}
$$

In this model, as $x$ increases, the log odds decrease.

```{r}
set.seed(1)
example_data = sim_logistic_data(sample_size = 50, beta_0 = 1, beta_1 = -4)
```

<!-- slide -->

We again simulate some observations form this model, then fit logistic regression.

```{r}
fit_glm = glm(y ~ x, data = example_data, family = binomial)
```

```{r}
plot(y ~ x, data = example_data,
     pch = 20, ylab = "Estimated Probability",
     main = "Logistic Regression, Decreasing Probability")
grid()
curve(predict(fit_glm, data.frame(x), type = "response"),
      add = TRUE, col = "dodgerblue", lty = 2)
curve(boot::inv.logit(1 - 4 * x), add = TRUE, col = "darkorange", lty = 1)
legend("bottomleft", c("True Probability", "Estimated Probability", "Data"), lty = c(1, 2, 0),
       pch = c(NA, NA, 20), lwd = 2, col = c("darkorange", "dodgerblue", "black"))
```

<!-- slide -->

# ![Lecture12-2](/assets/logit2.png)

<!-- slide -->

Now let's look at an example where the estimated probability doesn't always simply increase or decrease. Much like ordinary linear regression, the linear combination of predictors can contain transformations of predictors (in this case a quadratic term) and interactions.

```{r}
sim_quadratic_logistic_data = function(sample_size = 25) {
  x = rnorm(n = sample_size)
  eta = -1.5 + 0.5 * x + x ^ 2
  p = 1 / (1 + exp(-eta))
  y = rbinom(n = sample_size, size = 1, prob = p)
  data.frame(y, x)
}
```

$$
\log\left(\frac{p({\bf x})}{1 - p({\bf x})}\right) = -1.5 + 0.5x + x^2.
$$

<!-- slide -->

Again, we could re-write this to better match the function we're using to simulate the data:

$$
\begin{aligned}
Y_i \mid {\bf X_i} = {\bf x_i} &\sim \text{Bern}(p_i) \\
p_i &= p({\bf x_i}) = \frac{1}{1 + e^{-\eta({\bf x_i})}} \\
\eta({\bf x_i}) &= -1.5 + 0.5x_i + x_i^2
\end{aligned}
$$

```{r}
set.seed(42)
example_data = sim_quadratic_logistic_data(sample_size = 50)
```

```{r}
fit_glm = glm(y ~ x + I(x^2), data = example_data, family = binomial)
```

<!-- slide -->

# ![Lecture12-3](/assets/logit3.png)

<!-- slide -->

```{r}
plot(y ~ x, data = example_data,
     pch = 20, ylab = "Estimated Probability",
     main = "Logistic Regression, Quadratic Relationship")
grid()
curve(predict(fit_glm, data.frame(x), type = "response"),
      add = TRUE, col = "dodgerblue", lty = 2)
curve(boot::inv.logit(-1.5 + 0.5 * x + x ^ 2),
      add = TRUE, col = "darkorange", lty = 1)
legend("bottomleft", c("True Probability", "Estimated Probability", "Data"), lty = c(1, 2, 0),
       pch = c(NA, NA, 20), lwd = 2, col = c("darkorange", "dodgerblue", "black"))
```

<!-- slide -->

## Working with Logistic Regression

While the logistic regression model isn't exactly the same as the ordinary linear regression model, because they both use a **linear** combination of the predictors

$$
\eta({\bf x}) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots  + \beta_{p - 1} x_{p - 1}
$$

working with logistic regression is very similar. Many of the things we did with ordinary linear regression can be done with logistic regression in a very similar fashion. For example,

- Testing for a single $\beta$ parameter
- Testing for a set of $\beta$ parameters
- Formula specification in `R`
- Interpreting parameters and estimates
- Confidence intervals for parameters
- Confidence intervals for mean response
- Variable selection

<!-- slide -->

### Hypothesis Testing: Wald Test

In ordinary linear regression, we perform the test of

$$
H_0: \beta_j = 0 \quad \text{vs} \quad H_1: \beta_j \neq 0
$$

using a $t$-test.

For the logistic regression model,

$$
\log\left(\frac{p({\bf x})}{1 - p({\bf x})}\right) = \beta_0 + \beta_1 x_1 + \ldots  + \beta_{p - 1} x_{p - 1}
$$

we can again perform a test of

$$
H_0: \beta_j = 0 \quad \text{vs} \quad H_1: \beta_j \neq 0
$$

however, the test statistic and its distribution are no longer $t$.

<!-- slide -->

With a little algebra and mathematical work, we can see that the test statistic takes the same form

$$
z = \frac{\hat{\beta}_j - \beta_j}{\text{SE}[\hat{\beta}_j]} \overset{\text{approx}}{\sim} N(0, 1)
$$

but now we are performing a $z$-test, as the test statistic is approximated by a standard normal distribution, *provided we have a large enough sample*. (The $t$-test for ordinary linear regression, assuming the assumptions were correct, had an exact distribution for any sample size.)

We'll skip some of the exact details of the calculations, as `R` will obtain the standard error for us. The use of this test will be extremely similar to the $t$-test for ordinary linear regression. Essentially the only thing that changes is the distribution of the test statistic.

<!-- slide -->

### Likelihood-Ratio Test

Consider the following **full** model,

$$
\log\left(\frac{p({\bf x_i})}{1 - p({\bf x_i})}\right) = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_{(p-1)} x_{i(p-1)} + \epsilon_i
$$

This model has $p - 1$ predictors, for a total of $p$ $\beta$-parameters. We will denote the MLE of these $\beta$-parameters as $\hat{\beta}_{\text{Full}}$

Now consider a **null** (or **reduced**) model,

$$
\log\left(\frac{p({\bf x_i})}{1 - p({\bf x_i})}\right) = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_{(q-1)} x_{i(q-1)} + \epsilon_i
$$

where $q < p$. This model has $q - 1$ predictors, for a total of $q$ $\beta$-parameters. We will denote the MLE of these $\beta$-parameters as $\hat{\beta}_{\text{Null}}$

<!-- slide -->

As with linear regression, the difference between these two models can be codified by the null hypothesis of a test.

$$
H_0: \beta_q = \beta_{q+1} = \cdots = \beta_{p - 1} = 0.
$$

This implies that the reduced model is nested inside the full model.

We then define a test statistic, $D$,

$$
D = -2 \log \left( \frac{L(\boldsymbol{\hat{\beta}_{\text{Null}}})} {L(\boldsymbol{\hat{\beta}_{\text{Full}}})} \right) = 2 \log \left( \frac{L(\boldsymbol{\hat{\beta}_{\text{Full}}})} {L(\boldsymbol{\hat{\beta}_{\text{Null}}})} \right) = 2 \left( \ell(\hat{\beta}_{\text{Full}}) - \ell(\hat{\beta}_{\text{Null}})\right)
$$

where $L$ denotes a likelihood and $\ell$ denotes a log-likelihood. For a large enough sample, this test statistic has an approximate Chi-square distribution

$$
D \overset{\text{approx}}{\sim} \chi^2_{k}
$$

where $k = p - q$, the difference in number of parameters of the two models.
<!-- slide -->

This test, which we will call the **Likelihood-Ratio Test**, will be the analogue to the ANOVA $F$-test for logistic regression. Interestingly, to perform the Likelihood-Ratio Test, we'll actually again use the `anova()` function in `R`!.

The Likelihood-Ratio Test is actually a rather general test, however, here we have presented a specific application to nested logistic regression models.

<!-- slide -->
