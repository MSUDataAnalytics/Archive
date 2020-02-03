<!-- slide -->
# Bias–Variance Tradeoff

>“Give me a lever long enough and a fulcrum on which to place it, and I shall move the world.”
>
> **- Archimedes**

In this lecture, we will focus on one of the key concepts in statistical learning (and data analysis in general). By the end of this lecture, we will have discussed:

- Definitions of bias and variance
- What tradeoffs are implicitly made in data analytics
- How to estimate the tradeoff practically.

Note that this lecture is more theoretical than the last few lectures.

<!-- slide -->

# Bias–Variance Tradeoff

Consider the general regression setup where we are given a random pair $(X, Y) \in \mathbb{R}^p \times \mathbb{R}$. We would like to "predict" $Y$ with some function of $X$, say, $g(X)$.

To clarify what we mean by "predict," we specify that we would like $g(X)$ to be "close" to $Y$ (which is, by definition, given by $Y = f(X) + \epsilon$).

To further clarify what we mean by "close," we define the **squared error loss** of estimating $Y$ using $g(X)$.

$$
L(Y, g(X)) \triangleq (Y - g(X)) ^ 2
$$

Now we can clarify the goal of regression, which is to minimize the above loss, on average. We call this the **risk** of estimating $Y$ using $g(X)$.

$$
R(Y, f(X)) \triangleq \mathbb{E}[L(Y, g(X))] = \mathbb{E}_{X, Y}[(Y - g(X)) ^ 2]
$$

<!-- slide -->

Before attempting to minimize the risk, we first re-write the risk after conditioning on $X$.

$$
\mathbb{E}_{X, Y} \left[ (Y - g(X)) ^ 2 \right] = \mathbb{E}_{X} \mathbb{E}_{Y \mid X} \left[ ( Y - g(X) ) ^ 2 \mid X = x \right]
$$

Minimizing the right-hand side is much easier, as it simply amounts to minimizing the inner expectation with respect to $Y \mid X$. That is, we need to (essentially) minimize the risk pointwise for each $x$.

It turns out that the risk is minimized by the conditional mean of $Y$ given $X$,

$$
g(x) = \mathbb{E}(Y \mid X = x)
$$

This $g$ is often called the **regression function**; we'll see why in a moment.

<!-- slide -->

Note that the choice of squared error loss is somewhat arbitrary. Suppose instead we chose absolute error loss.

$$
L(Y, f(X)) \triangleq | Y - g(X) |
$$

The risk would then be minimized by the conditional median.

$$
g(x) = \text{median}(Y \mid X = x)
$$

Despite this possibility, our preference will still be for squared error loss. The reasons for this are numerous, including: historical reasons, ease of optimization, large-sample properties, and protecting against large deviations.

Now, given data $\mathcal{D} = (x_i, y_i) \in \mathbb{R}^p \times \mathbb{R}$, our goal becomes finding some $\hat{g}$ that is a good estimate of the regression function $g$. We'll see that this amounts to minimizing what we call the reducible error.

<!-- slide -->

## Reducible and Irreducible Error

Suppose that we obtain some $\hat{g}$. How well does it estimate $g$? We define the **expected prediction error** of predicting $Y$ using $\hat{g}(X)$. A good $\hat{g}$ will have a low expected prediction error.

$$
\text{EPE}\left(Y, \hat{g}(X)\right) \triangleq \mathbb{E}_{X, Y, \mathcal{D}} \left[  \left( Y - \hat{g}(X) \right)^2 \right]
$$

This expectation is over $X$, $Y$, and also $\mathcal{D}$. The estimate $\hat{g}$ is actually random depending on the sampled data $\mathcal{D}$. We could actually write $\hat{g}(X, \mathcal{D})$ to make this dependence explicit, but our notation will become cumbersome enough as it is.

Like before, we'll condition on $X$. This results in the expected prediction error of predicting $Y$ using $\hat{g}(X)$ when $X = x$.

$$
\text{EPE}\left(Y, \hat{g}(x)\right) =
\mathbb{E}_{Y \mid X, \mathcal{D}} \left[  \left(Y - \hat{g}(X) \right)^2 \mid X = x \right] =
\underbrace{\mathbb{E}_{\mathcal{D}} \left[  \left(f(x) - \hat{g}(x) \right)^2 \right]}_\textrm{reducible error} +
\underbrace{\mathbb{V}_{Y \mid X} \left[ Y \mid X = x \right]}_\textrm{irreducible error}
$$

<!-- slide -->
A number of things to note here:

- The expected prediction error is for a random $Y$ given a fixed $x$ and a random $\hat{f}$. As such, the expectation is over $Y \mid X$ and $\mathcal{D}$. Our estimated function $\hat{f}$ is random depending on the sampled data, $\mathcal{D}$, which is used to perform the estimation.
- The expected prediction error of predicting $Y$ using $\hat{f}(X)$ when $X = x$ has been decomposed into two errors:
    - The **reducible error**, which is the expected squared error loss of estimation $f(x)$ using $\hat{f}(x)$ at a fixed point $x$. The only thing that is random here is $\mathcal{D}$, the data used to obtain $\hat{f}$. (Both $f$ and $x$ are fixed.) We'll often call this reducible error the **mean squared error** of estimating $f(x)$ using $\hat{f}$ at a fixed point $x$. $$\text{MSE}\left(f(x), \hat{f}(x)\right) \triangleq \mathbb{E}_{\mathcal{D}} \left[  \left(f(x) - \hat{f}(x) \right)^2 \right]$$
    - The **irreducible error**. This is simply the variance of $Y$ given that $X = x$, essentially noise that we do not want to learn. This is also called the **Bayes error**.

As the name suggests, the reducible error is the error that we have some control over. But how do we control this error?

<!-- slide -->
## Bias-Variance Decomposition

After decomposing the expected prediction error into reducible and irreducible error, we can further decompose the reducible error.

The **bias** of an estimator is given by

$$
\text{bias}(\hat{\theta}) \triangleq \mathbb{E}\left[\hat{\theta}\right] - \theta.
$$

And recall the definition of the **variance** of an estimator:

$$
\mathbb{V}(\hat{\theta}) = \text{var}(\hat{\theta}) \triangleq \mathbb{E}\left [ ( \hat{\theta} -\mathbb{E}\left[\hat{\theta}\right] )^2 \right].
$$

Using this, we further decompose the reducible error (mean squared error) into bias squared and variance.

$$
\text{MSE}\left(g(x), \hat{g}(x)\right) =
\mathbb{E}_{\mathcal{D}} \left[  \left(g(x) - \hat{g}(x) \right)^2 \right] =
\underbrace{\left(g(x) - \mathbb{E} \left[ \hat{g}(x) \right]  \right)^2}_{\text{bias}^2 \left(\hat{g}(x) \right)} +
\underbrace{\mathbb{E} \left[ \left( \hat{g}(x) - \mathbb{E} \left[ \hat{g}(x) \right] \right)^2 \right]}_{\text{var} \left(\hat{g}(x) \right)}
$$
<!-- slide -->
(An aside)

This is actually a commonly explored fact in estimation theory, and it is geenrally true. But we have stated this decomposition here specifically for estimation of some regression function $g$ using $\hat{g}$ at some point $x$.

$$
\text{MSE}\left(g(x), \hat{g}(x)\right) = \text{bias}^2 \left(\hat{g}(x) \right) + \text{var} \left(\hat{g}(x) \right)
$$

<!-- slide -->
In a perfect world, we would be able to find some $\hat{f}$ which is **unbiased**, that is $\text{bias}\left(\hat{f}(x) \right) = 0$, which also has low variance. In practice, this isn't always possible.

It turns out, there is a **bias-variance tradeoff**. That is, often, the more bias in our estimation, the lesser the variance. Similarly, less variance is often accompanied by more bias. Complex models tend to be unbiased, but highly variable. Simple models are often extremely biased, but have low variance.

In the context of regression, models are biased when:

- Parametric: The form of the model [does not incorporate all the necessary variables](https://en.wikipedia.org/wiki/Omitted-variable_bias), or the form of the relationship is too simple. For example, a parametric model assumes a linear relationship, but the true relationship is quadratic.
- Non-parametric: The model provides too much smoothing.

<!-- slide -->

In the context of regression, models are variable when:

- Parametric: The form of the model incorporates too many variables, or the form of the relationship is too complex. For example, a parametric model assumes a cubic relationship, but the true relationship is linear.
- Non-parametric: The model does not provide enough smoothing. It is very, "wiggly."

So for us, to select a model that appropriately balances the tradeoff between bias and variance, and thus minimizes the reducible error, we need to select a model of the appropriate complexity for the data.

<!-- slide -->
Let's explore this relationship in greater detail.

```{r}
# This code included solely for completeness; you do not need to execute this.
x = seq(0, 100, by = 0.001)
f = function(x) {
  ((x - 50) / 50) ^ 2 + 2
}
g = function(x) {
  1 - ((x - 50) / 50)
}

par(mgp = c(1.5, 1.5, 0))
plot(x, g(x), ylim = c(0, 3), type = "l", lwd = 2,
     ylab = "Error", xlab = expression(Low %<-% Complexity %->% High),
     main = "Error versus Model Complexity", col = "darkorange",
     axes = FALSE)
grid()
axis(1, labels = FALSE)
axis(2, labels = FALSE)
box()
curve(f, lty = 6, col = "dodgerblue", lwd = 3, add = TRUE)
legend("bottomleft", c("(Expected) Test", "Train"), lty = c(6, 1), lwd = 3,
       col = c("dodgerblue", "darkorange"))
```
<!-- slide -->
# ![Lecture7-1](/assets/Lecture7-1.png)

<!-- slide -->
This theoretical relationship isn't obvious. But with a little work, we can understand why this is happening.

The expected test MSE is essentially the expected prediction error, which we now know decomposes into (squared) bias, variance, and the irreducible Bayes error. The following plots show three examples of this.

```{r}
# This code included solely for completeness; you do not need to execute this.
x = seq(0.01, 0.99, length.out = 1000)

par(mfrow = c(1, 3))
par(mgp = c(1.5, 1.5, 0))

b = 0.05 / x
v = 5 * x ^ 2 + 0.5
bayes = 4
epe = b + v + bayes

plot(x, b, type = "l", ylim = c(0, 10), col = "dodgerblue", lwd = 2, lty = 3,
     xlab = "Model Complexity", ylab = "Error", axes = FALSE,
     main = "More Dominant Variance")
axis(1, labels = FALSE)
axis(2, labels = FALSE)
grid()
box()
lines(x, v, col = "darkorange", lwd = 2, lty = 4)
lines(x, epe, col = "black", lwd = 2)
abline(h = bayes, lty = 2, lwd = 2, col = "darkgrey")
abline(v = x[which.min(epe)], col = "grey", lty = 3, lwd = 2)

b = 0.05 / x
v = 5 * x ^ 4 + 0.5
bayes = 4
epe = b + v + bayes

plot(x, b, type = "l", ylim = c(0, 10), col = "dodgerblue", lwd = 2, lty = 3,
     xlab = "Model Complexity", ylab = "Error", axes = FALSE,
     main = "Decomposition of Prediction Error")
axis(1, labels = FALSE)
axis(2, labels = FALSE)
grid()
box()
lines(x, v, col = "darkorange", lwd = 2, lty = 4)
lines(x, epe, col = "black", lwd = 2)
abline(h = bayes, lty = 2, lwd = 2, col = "darkgrey")
abline(v = x[which.min(epe)], col = "grey", lty = 3, lwd = 2)

b = 6 - 6 * x ^ (1 / 4)
v = 5 * x ^ 6 + 0.5
bayes = 4
epe = b + v + bayes

plot(x, b, type = "l", ylim = c(0, 10), col = "dodgerblue", lwd = 2, lty = 3,
     xlab = "Model Complexity", ylab = "Error", axes = FALSE,
     main = "More Dominant Bias")
axis(1, labels = FALSE)
axis(2, labels = FALSE)
grid()
box()
lines(x, v, col = "darkorange", lwd = 2, lty = 4)
lines(x, epe, col = "black", lwd = 2)
abline(h = bayes, lty = 2, lwd = 2, col = "darkgrey")
abline(v = x[which.min(epe)], col = "grey", lty = 3, lwd = 2)
legend("topright", c("Squared Bias", "Variance", "Bayes", "EPE"), lty = c(3, 4, 2, 1),
       col = c("dodgerblue", "darkorange", "darkgrey", "black"), lwd = 2)
```

<!-- slide -->
# ![Lecture7-2](/assets/Lecture7-2.png)


<!-- slide -->

The three plots show three examples of the bias-variance tradeoff. In the left panel, the variance influences the expected prediction error more than the bias. In the right panel, the opposite is true. The middle panel is somewhat neutral. In all cases, the difference between the Bayes error (the horizontal dashed grey line) and the expected prediction error (the solid black curve) is exactly the mean squared error, which is the sum of the squared bias (blue curve) and variance (orange curve). The vertical line indicates the complexity that minimizes the prediction error.

To summarize, if we assume that irreducible error can be written as

$$
\mathbb{V}[Y \mid X = x] = \sigma ^ 2
$$

then we can write the full decomposition of the expected prediction error of predicting $Y$ using $\hat{g}$ when $X = x$ as

$$
\text{EPE}\left(Y, \hat{f}(x)\right) =
\underbrace{\text{bias}^2\left(\hat{g}(x)\right) + \text{var}\left(\hat{g}(x)\right)}_\textrm{reducible error} + \sigma^2.
$$

As model complexity increases, bias decreases, while variance increases. By understanding the tradeoff between bias and variance, we can manipulate model complexity to find a model that well predict well on unseen observations.


<!-- slide -->

# ![Lecture7-3](/assets/Lecture7-3.png)

<!-- slide -->


## Simulation

We will illustrate these decompositions---most importantly the bias-variance tradeoff---through simulation. Suppose we would like to train a model to learn the function $f(x) = x^2$.

```{r}
f = function(x) {
  x ^ 2
}
```

More specifically, we'd like to predict an observation, $Y$, given that $X = x$ by using $\hat{g}(x)$ where

$$
\mathbb{E}[Y \mid X = x] = f(x) = x^2
$$

and

$$
\mathbb{V}[Y \mid X = x] = \sigma ^ 2.
$$

Alternatively, we could write this as

$$
Y = f(X) + \epsilon
$$

where $\mathbb{E}[\epsilon] = 0$ and $\mathbb{V}[\epsilon] = \sigma ^ 2$. In this formulation, we call $f(X)$ the **signal** and $\epsilon$ the **noise**.

<!-- slide -->

To carry out a concrete simulation example, we need to fully specify the data generating process. We do so with the following `R` code.

```{r}
get_sim_data = function(f, sample_size = 100) {
  x = runif(n = sample_size, min = 0, max = 1)
  y = rnorm(n = sample_size, mean = f(x), sd = 0.3)
  data.frame(x, y)
}
```

Also note that if you prefer to think of this situation using the $Y = f(X) + \epsilon$ formulation, the following code represents the same data generating process.

```{r}
get_sim_data = function(f, sample_size = 100) {
  x = runif(n = sample_size, min = 0, max = 1)
  eps = rnorm(n = sample_size, mean = 0, sd = 0.75)
  y = f(x) + eps
  data.frame(x, y)
}
```

<!-- slide -->

To completely specify the data generating process, we have made more model assumptions than simply $\mathbb{E}[Y \mid X = x] = x^2$ and $\mathbb{V}[Y \mid X = x] = \sigma ^ 2$. In particular,

- The $x_i$ in $\mathcal{D}$ are sampled from a uniform distribution over $[0, 1]$.
- The $x_i$ and $\epsilon$ are independent.
- The $y_i$ in $\mathcal{D}$ are sampled from the conditional normal distribution.

$$
Y \mid X \sim N(f(x), \sigma^2)
$$

Using this setup, we will generate datasets, $\mathcal{D}$, with a sample size $n = 100$ and fit four models.

$$
\begin{aligned}
\texttt{predict(fit0, x)} &= \hat{f}_0(x) = \hat{\beta}_0\\
\texttt{predict(fit1, x)} &= \hat{f}_1(x) = \hat{\beta}_0 + \hat{\beta}_1 x \\
\texttt{predict(fit2, x)} &= \hat{f}_2(x) = \hat{\beta}_0 + \hat{\beta}_1 x + \hat{\beta}_2 x^2 \\
\texttt{predict(fit9, x)} &= \hat{f}_9(x) = \hat{\beta}_0 + \hat{\beta}_1 x + \hat{\beta}_2 x^2 + \ldots + \hat{\beta}_9 x^9
\end{aligned}
$$

<!-- slide -->

To get a sense of the data and these four models, we generate one simulated dataset, and fit the four models.

```{r}
set.seed(1)
sim_data = get_sim_data(f)
```

```{r}
fit_0 = lm(y ~ 1,                   data = sim_data)
fit_1 = lm(y ~ poly(x, degree = 1), data = sim_data)
fit_2 = lm(y ~ poly(x, degree = 2), data = sim_data)
fit_9 = lm(y ~ poly(x, degree = 9), data = sim_data)
```

Note that by using this shorthand from the `lm()` function, we are being lazy and using orthogonal polynomials. That is, the values used for each $x$ aren't exactly what you'd expect. But the fitted values are the same, so this makes no difference for prediction purposes.

<!-- slide -->

Plotting these four trained models, we see that the zero predictor model does very poorly. The first degree model is reasonable, but we can see that the second degree model fits much better. The ninth degree model seem rather wild.

```{r, fig.height = 6, fig.width = 9, echo = FALSE}
set.seed(42)
plot(y ~ x, data = sim_data, col = "grey", pch = 20,
     main = "Four Polynomial Models fit to a Simulated Dataset")
grid()
grid = seq(from = 0, to = 2, by = 0.01)
lines(grid, f(grid), col = "black", lwd = 3)
lines(grid, predict(fit_0, newdata = data.frame(x = grid)), col = "dodgerblue",  lwd = 2, lty = 2)
lines(grid, predict(fit_1, newdata = data.frame(x = grid)), col = "firebrick",   lwd = 2, lty = 3)
lines(grid, predict(fit_2, newdata = data.frame(x = grid)), col = "springgreen", lwd = 2, lty = 4)
lines(grid, predict(fit_9, newdata = data.frame(x = grid)), col = "darkorange",  lwd = 2, lty = 5)

legend("topleft",
       c("y ~ 1", "y ~ poly(x, 1)", "y ~ poly(x, 2)",  "y ~ poly(x, 9)", "truth"),
       col = c("dodgerblue", "firebrick", "springgreen", "darkorange", "black"), lty = c(2, 3, 4, 5, 1), lwd = 2)
```

<!-- slide -->

# ![Lecture7-4](/assets/Lecture7-4.png)

<!-- slide -->

The following three plots were created using three additional simulated datasets. The zero predictor and ninth degree polynomial were fit to each.

```{r, fig.height = 4, fig.width = 12, echo = FALSE}
par(mfrow = c(1, 3))

# Note: This code is BAD. Don't use this. Or, like, clean it up.
# This is lazy and bad programming. Seriously, don't program like this. DON'T DO IT.

set.seed(430)
sim_data_1 = get_sim_data(f)
sim_data_2 = get_sim_data(f)
sim_data_3 = get_sim_data(f)
fit_0_1 = lm(y ~ 1, data = sim_data_1)
fit_0_2 = lm(y ~ 1, data = sim_data_2)
fit_0_3 = lm(y ~ 1, data = sim_data_3)
fit_9_1 = lm(y ~ poly(x, degree = 9), data = sim_data_1)
fit_9_2 = lm(y ~ poly(x, degree = 9), data = sim_data_2)
fit_9_3 = lm(y ~ poly(x, degree = 9), data = sim_data_3)

plot(y ~ x, data = sim_data_1, col = "grey", pch = 20, main = "Simulated Dataset 1")
grid()
grid = seq(from = 0, to = 2, by = 0.01)
lines(grid, predict(fit_0_1, newdata = data.frame(x = grid)), col = "dodgerblue", lwd = 2, lty = 2)
lines(grid, predict(fit_9_1, newdata = data.frame(x = grid)), col = "darkorange", lwd = 2, lty = 5)
legend("topleft", c("y ~ 1", "y ~ poly(x, 9)"), col = c("dodgerblue", "darkorange"), lty = c(2, 5), lwd = 2)

plot(y ~ x, data = sim_data_2, col = "grey", pch = 20, main = "Simulated Dataset 2")
grid()
grid = seq(from = 0, to = 2, by = 0.01)
lines(grid, predict(fit_0_2, newdata = data.frame(x = grid)), col = "dodgerblue", lwd = 2, lty = 2)
lines(grid, predict(fit_9_2, newdata = data.frame(x = grid)), col = "darkorange", lwd = 2, lty = 5)
legend("topleft", c("y ~ 1", "y ~ poly(x, 9)"), col = c("dodgerblue", "darkorange"), lty = c(2, 5), lwd = 2)

plot(y ~ x, data = sim_data_3, col = "grey", pch = 20, main = "Simulated Dataset 3")
grid()
grid = seq(from = 0, to = 2, by = 0.01)
lines(grid, predict(fit_0_3, newdata = data.frame(x = grid)), col = "dodgerblue", lwd = 2, lty = 2)
lines(grid, predict(fit_9_3, newdata = data.frame(x = grid)), col = "darkorange", lwd = 2, lty = 5)
legend("topleft", c("y ~ 1", "y ~ poly(x, 9)"), col = c("dodgerblue", "darkorange"), lty = c(2, 5), lwd = 2)
```

<!-- slide -->

These plots start to highlight the difference between the bias and variance of these two models. The zero predictor model is clearly wrong, that is, **biased**. However, it's nearly the same for each of the datasets---it has very low variance.

While the ninth degree model doesn't appear to be correct for any of these three simulations, we'll see that on average it is, and thus is performing unbiased estimation. These plots do however clearly illustrate that the ninth degree polynomial is extremely variable. Each dataset results in a very different fitted model. **Correct on average** isn't the only goal you should be after. In practice, we often only have a single dataset but wish to extrapolate our conclusions outside that dataset to the world at large. This is why we'd also like our models to exhibit low variance.

<!-- slide -->

We could have also fit $k$-nearest neighbors models to these three datasets.

```{r, fig.height = 4, fig.width = 12, echo = FALSE}
par(mfrow = c(1, 3))

# if you're reading this code
# it's BAD! don't use it. (or clean it up)
# also, note to self: clean up this code!!!

grid = seq(from = 0, to = 2, by = 0.01)

set.seed(430)
sim_data_1 = get_sim_data(f)
sim_data_2 = get_sim_data(f)
sim_data_3 = get_sim_data(f)
fit_0_1 = FNN::knn.reg(train = sim_data_1["x"], test = data.frame(x = grid), y = sim_data_1["y"], k = 5)$pred
fit_0_2 = FNN::knn.reg(train = sim_data_2["x"], test = data.frame(x = grid), y = sim_data_2["y"], k = 5)$pred
fit_0_3 = FNN::knn.reg(train = sim_data_3["x"], test = data.frame(x = grid), y = sim_data_3["y"], k = 5)$pred
fit_9_1 = FNN::knn.reg(train = sim_data_1["x"], test = data.frame(x = grid), y = sim_data_1["y"], k = 100)$pred
fit_9_2 = FNN::knn.reg(train = sim_data_2["x"], test = data.frame(x = grid), y = sim_data_2["y"], k = 100)$pred
fit_9_3 = FNN::knn.reg(train = sim_data_3["x"], test = data.frame(x = grid), y = sim_data_3["y"], k = 100)$pred

plot(y ~ x, data = sim_data_1, col = "grey", pch = 20, main = "Simulated Dataset 1")
grid()
lines(grid, fit_0_1, col = "dodgerblue", lwd = 1, lty = 1)
lines(grid, fit_9_1, col = "darkorange", lwd = 2, lty = 2)
legend("topleft", c("k = 5", "k = 100"), col = c("dodgerblue", "darkorange"), lty = c(1, 2), lwd = 2)

plot(y ~ x, data = sim_data_2, col = "grey", pch = 20, main = "Simulated Dataset 2")
grid()
lines(grid, fit_0_2, col = "dodgerblue", lwd = 1, lty = 1)
lines(grid, fit_9_2, col = "darkorange", lwd = 2, lty = 2)
legend("topleft", c("k = 5", "k = 100"), col = c("dodgerblue", "darkorange"), lty = c(1, 2), lwd = 2)

plot(y ~ x, data = sim_data_3, col = "grey", pch = 20, main = "Simulated Dataset 3")
grid()
lines(grid, fit_0_3, col = "dodgerblue", lwd = 1, lty = 1)
lines(grid, fit_9_3, col = "darkorange", lwd = 2, lty = 2)
legend("topleft", c("k = 5", "k = 100"), col = c("dodgerblue", "darkorange"), lty = c(1, 2), lwd = 2)
```

<!-- slide -->

Here we see that when $k = 100$ we have a biased model with very low variance. (It's actually the same as the 0 predictor linear model.) When $k = 5$, we again have a highly variable model.

These two sets of plots reinforce our intuition about the bias-variance tradeoff. Complex models (ninth degree polynomial and $k$ = 5) are highly variable, and often unbiased. Simple models (zero predictor linear model and $k = 100$) are very biased, but have extremely low variance.

<!-- slide -->

We will now complete a simulation study to understand the relationship between the bias, variance, and mean squared error for the estimates for $f(x)$ given by these four models at the point $x = 0.90$. We use simulation to complete this task, as performing the analytical calculations would prove to be rather tedious and difficult.

```{r}
set.seed(1)
n_sims = 250
n_models = 4
x = data.frame(x = 0.90) # Fixed point at which we make predictions
predictions = matrix(0, nrow = n_sims, ncol = n_models)
```

```{r}
for (sim in 1:n_sims) {

  # Simulate new, random, training data
  # This is the only random portion of the bias, var, and mse calculations
  # This allows us to calculate the expectation over D
  sim_data = get_sim_data(f)

  # Fit models
  fit_0 = lm(y ~ 1,                   data = sim_data)
  fit_1 = lm(y ~ poly(x, degree = 1), data = sim_data)
  fit_2 = lm(y ~ poly(x, degree = 2), data = sim_data)
  fit_9 = lm(y ~ poly(x, degree = 9), data = sim_data)

  # Get predictions
  predictions[sim, 1] = predict(fit_0, x)
  predictions[sim, 2] = predict(fit_1, x)
  predictions[sim, 3] = predict(fit_2, x)
  predictions[sim, 4] = predict(fit_9, x)
}
```

<!-- slide -->

```{r}
# Alternative simulation strategy
sim_pred_from_lm_at_point = function(x) {

  # x value to predict at
  # coerce to data frame for predict() function
  x = data.frame(x = x)

  # simulate new training data
  # expectation over D
  sim_data = get_sim_data(f)

  # fit models
  fit_0 = lm(y ~ 1,                   data = sim_data)
  fit_1 = lm(y ~ poly(x, degree = 1), data = sim_data)
  fit_2 = lm(y ~ poly(x, degree = 2), data = sim_data)
  fit_9 = lm(y ~ poly(x, degree = 9), data = sim_data)

  # get prediction at point for each model
  c(predict(fit_0, x),
    predict(fit_1, x),
    predict(fit_2, x),
    predict(fit_9, x))

}

set.seed(1)
predictions = replicate(n = 250, sim_pred_from_lm_at_point(x = 0.90))
```

<!-- slide -->

### A Programming Aside

Note that these are two of many ways we could have accomplished this task using `R`. For example we could have used a combination of `replicate()` and `*apply()` functions. Alternatively, we could have used a [`tidyverse`](https://www.tidyverse.org/) approach, which likely would have used some combination of [`dplyr`](http://dplyr.tidyverse.org/), [`tidyr`](http://tidyr.tidyverse.org/), and [`purrr`](http://purrr.tidyverse.org/).

Our approach, which would be considered a `base` `R` approach, was chosen to make it as clear as possible what is being done. The `tidyverse` approach is rapidly gaining popularity in the `R` community, but might make it more difficult to see what is happening here, unless you are already familiar with that approach.

Also of note, while it may seem like the output stored in `predictions` would meet the definition of [tidy data](http://vita.had.co.nz/papers/tidy-data.html) given by [Hadley Wickham](http://hadley.nz/) since each row represents a simulation, it actually falls slightly short. For our data to be tidy, a row should store the simulation number, the model, and the resulting prediction. We've actually already aggregated one level above this. Our observational unit is a simulation (with four predictions), but for tidy data, it should be a single prediction. This may be revised by the author later when there are [more examples of how to do this from the `R` community](https://twitter.com/hspter/status/748260288143589377).

<!-- slide -->

```{r}
# Note that this code requires tidyverse; it can get a bit fussy if running OSX.
predictions = (predictions)
colnames(predictions) = c("0", "1", "2", "9")
predictions = as.data.frame(predictions)

tall_predictions = tidyr::gather(predictions, factor_key = TRUE)
boxplot(value ~ key, data = tall_predictions, border = "darkgrey", xlab = "Polynomial Degree", ylab = "Predictions",
        main = "Simulated Predictions for Polynomial Models")
grid()
stripchart(value ~ key, data = tall_predictions, add = TRUE, vertical = TRUE, method = "jitter", jitter = 0.15, pch = 1, col = c("dodgerblue", "firebrick", "springgreen", "darkorange"))
abline(h = f(x = 0.90), lwd = 2)
```
<!-- slide -->


The plot generated from the code shows the predictions for each of the `250` simulations of each of the four models of different polynomial degrees. The truth, $g(x = 0.90) = (0.9)^2 = 0.81$, is given by the solid black horizontal line.

Two things are immediately clear:

- As complexity *increases*, **bias decreases**.

(The mean of a model's predictions is closer to the truth.)

- As complexity *increases*, **variance increases**.

(The variance about the mean of a model's predictions increases.)


<!-- slide -->

The goal of this simulation study is to show that the following holds true for each of the four models.

$$
\text{MSE}\left(g(0.90), \hat{g}_k(0.90)\right) =
\underbrace{\left(\mathbb{E} \left[ \hat{g}_k(0.90) \right] - g(0.90) \right)^2}_{\text{bias}^2 \left(\hat{g}_k(0.90) \right)} +
\underbrace{\mathbb{E} \left[ \left( \hat{g}_k(0.90) - \mathbb{E} \left[ \hat{g}_k(0.90) \right] \right)^2 \right]}_{\text{var} \left(\hat{g}_k(0.90) \right)}
$$

We'll use the empirical results of our simulations to estimate these quantities.[^1]

[^1]: Yes, we're using estimation to justify facts about estimation. Moreover, you should note that we've actually used a rather small number of simulations. In practice we should use more, but for the sake of computation time, we've performed just enough simulations to obtain the desired results. Since we're estimating estimation, the bigger the sample size, the better.

To estimate the mean squared error of our predictions, we'll use

$$
\widehat{\text{MSE}}\left(g(0.90), \hat{g}_k(0.90)\right) = \frac{1}{n_{\texttt{sims}}}\sum_{i = 1}^{n_{\texttt{sims}}} \left(g(0.90) - \hat{g}_k(0.90) \right)^2
$$

<!-- slide -->

We also write an accompanying `R` function.

```{r}
get_mse = function(truth, estimate) {
  mean((estimate - truth) ^ 2)
}
```

Similarly, for the bias of our predictions we use,

$$
\widehat{\text{bias}} \left(\hat{g}(0.90) \right)  = \frac{1}{n_{\texttt{sims}}}\sum_{i = 1}^{n_{\texttt{sims}}} \left(\hat{g}_k(0.90) \right) - f(0.90)
$$

And again, we write an accompanying `R` function.

```{r}
get_bias = function(estimate, truth) {
  mean(estimate) - truth
}
```

<!-- slide -->
Lastly, for the variance of our predictions we have

$$
\widehat{\text{var}} \left(\hat{g}(0.90) \right) = \frac{1}{n_{\texttt{sims}}}\sum_{i = 1}^{n_{\texttt{sims}}} \left(\hat{g}_k(0.90) - \frac{1}{n_{\texttt{sims}}}\sum_{i = 1}^{n_{\texttt{sims}}}\hat{g}_k(0.90) \right)^2
$$

While there is already `R` function for variance, the following is more appropriate in this situation.

```{r}
get_var = function(estimate) {
  mean((estimate - mean(estimate)) ^ 2)
}
```

To quickly obtain these results for each of the four models, we utilize the `apply()` function.

```{r}
bias = apply(predictions, 2, get_bias, truth = f(x = 0.90))
variance = apply(predictions, 2, get_var)
mse = apply(predictions, 2, get_mse, truth = f(x = 0.90))
```
<!-- slide -->
We summarize these results in the following table.

```{r, echo = FALSE, asis = TRUE}
results = data.frame(
  poly_degree = c(0, 1, 2, 9),
  round(mse, 5),
  round(bias ^ 2, 5),
  round(variance, 5)
)
colnames(results) = c("Degree", "Mean Squared Error", "Bias Squared", "Variance")
rownames(results) = NULL
knitr::kable(results, booktabs = TRUE, escape = TRUE, align = "c")
```
| Degree | Mean Squared Error | Bias Squared | Variance |
|:------:|:------------------:|:------------:|:--------:|
|   0    |      0.22643       |   0.22476    | 0.00167  |
|   1    |      0.00829       |   0.00508    | 0.00322  |
|   2    |      0.00387       |   0.00005    | 0.00381  |
|   9    |      0.01019       |   0.00002    | 0.01017  |

<!-- slide -->
A number of things to notice here:

- We use squared bias in this table. Since bias can be positive or negative, squared bias is more useful for observing the trend as complexity increases.
- The squared bias trend which we see here is **decreasing** as complexity increases, which we expect to see in general.
- The exact opposite is true of variance. As model complexity increases, variance **increases**.
- The mean squared error, which is a function of the bias and variance, decreases, then increases. This is a result of the bias-variance tradeoff. We can decrease bias, by increasing variance. Or, we can decrease variance by increasing bias. By striking the correct balance, we can find a good mean squared error!

<!-- slide -->
A note: we can check for these trends with the `diff()` function in `R`.

```{r}
all(diff(bias ^ 2) < 0)
all(diff(variance) > 0)
diff(mse) < 0
```

<!-- slide -->

The models with polynomial degrees 2 and 9 are both essentially unbiased. We see some bias here as a result of using simulation. If we increased the number of simulations, we would see both biases go down. Since they are both unbiased, the model with degree 2 outperforms the model with degree 9 due to its smaller variance.

Models with degree 0 and 1 are biased because they assume the wrong form of the regression function. While the degree 9 model does this as well, it does include all the necessary polynomial degrees.

$$
\hat{g}_9(x) = \hat{\beta}_0 + \hat{\beta}_1 x + \hat{\beta}_2 x^2 + \ldots + \hat{\beta}_9 x^9
$$

<!-- slide -->

Then, since least squares estimation is unbiased, importantly,

$$
\mathbb{E}[\hat{\beta}_d] = \beta_d = 0
$$

for $d = 3, 4, \ldots 9$, we have

$$
\mathbb{E}\left[\hat{g}_9(x)\right] = \beta_0 + \beta_1 x + \beta_2 x^2
$$

<!-- slide -->
Now we can finally verify the bias-variance decomposition.

```{r}
bias ^ 2 + variance == mse
```

But wait, this says it isn't true, except for the degree 9 model? It turns out, this is simply a computational issue. If we allow for some very small error tolerance, we see that the bias-variance decomposition is indeed true for predictions from these for models.

```{r}
all.equal(bias ^ 2 + variance, mse)
```
See `?all.equal()` for details.

<!-- slide -->

So far, we've focused our efforts on looking at the mean squared error of estimating $g(0.90)$ using $\hat{g}(0.90)$. We could also look at the expected prediction error of using $\hat{g}(X)$ when $X = 0.90$ to estimate $Y$.

$$
\text{EPE}\left(Y, \hat{g}_k(0.90)\right) =
\mathbb{E}_{Y \mid X, \mathcal{D}} \left[  \left(Y - \hat{g}_k(X) \right)^2 \mid X = 0.90 \right]
$$

We can estimate this quantity for each of the four models using the simulation study we already performed.

```{r}
get_epe = function(realized, estimate) {
  mean((realized - estimate) ^ 2)
}
```

```{r}
y = rnorm(n = nrow(predictions), mean = f(x = 0.9), sd = 0.3)
epe = apply(predictions, 2, get_epe, realized = y)
epe
```

```{r}
# hmmm, what's wrong here?
# the mean realative diff does go down with n
# is there really just that much compuational error?
sigma_hat = mean((y - f(x = 0.90)) ^ 2)
all.equal(epe, bias ^ 2 + variance + sigma_hat)
```

<!-- slide -->

What about the **unconditional** expected prediction error? That is, for any $X$, not just $0.90$. Specifically, we might want to explore the expected prediction error of estimating $Y$ using $\hat{g}(X)$. The following (new) simulation study provides an estimate of

$$
\text{EPE}\left(Y, \hat{g}_k(X)\right) = \mathbb{E}_{X, Y, \mathcal{D}} \left[  \left( Y - \hat{g}_k(X) \right)^2 \right]
$$

where we simulate using the quadratic model---that is $k = 2$, as we have defined $k$ to be the degree of our polynomial.

```{r}
set.seed(1)
# Note this is intentionally incomplete
n_sims = 1000
X = runif(n = n_sims, min = 0, max = 1)
Y = rnorm(n = n_sims, mean = f(X), sd = 0.3)

f_hat_X = rep(0, length(X))

for (i in seq_along(X)) {
  sim_data = get_sim_data(f)
  fit_2 = lm(y ~ poly(x, degree = 2), data = sim_data)
  f_hat_X[i] = predict(fit_2, newdata = data.frame(x = X[i]))
}

mean((Y - f_hat_X) ^ 2)
```

As before, note that in applying such approaches outside the classroom, we should use many more simulations.
<!-- slide -->

## Estimating Expected Prediction Error

Assuming

$$
\mathbb{V}[Y \mid X = x] = \sigma ^ 2.
$$

we have

$$
\text{EPE}\left(Y, \hat{g}(X)\right) =
\mathbb{E}_{X, Y, \mathcal{D}} \left[  (Y - \hat{g}(X))^2 \right] =
\underbrace{\mathbb{E}_{X} \left[\text{bias}^2\left(\hat{g}(X)\right)\right] + \mathbb{E}_{X} \left[\text{var}\left(\hat{g}(X)\right)\right]}_\textrm{reducible error} + \sigma^2
$$

Lastly, we note that if

$$
\mathcal{D} = \mathcal{D}_{\texttt{trn}} \cup \mathcal{D}_{\texttt{tst}} = (x_i, y_i) \in \mathbb{R}^p \times \mathbb{R}, \ i = 1, 2, \ldots n
$$

where

$$
\mathcal{D}_{\texttt{trn}} = (x_i, y_i) \in \mathbb{R}^p \times \mathbb{R}, \ i \in \texttt{trn}
$$

and

$$
\mathcal{D}_{\texttt{tst}} = (x_i, y_i) \in \mathbb{R}^p \times \mathbb{R}, \ i \in \texttt{tst}
$$

<!-- slide -->

Then, if we use $\mathcal{D}_{\texttt{trn}}$ to fit (train) a model, we can use the test mean squared error

$$
\sum_{i \in \texttt{tst}}\left(y_i - \hat{g}(x_i)\right) ^ 2
$$

as an estimate of

$$
\mathbb{E}_{X, Y, \mathcal{D}} \left[  (Y - \hat{g}(X))^2 \right],
$$

which is the expected prediction error. Note that in practice---and on Lab 3---we prefer RMSE to MSE for comparing models and reporting because of the units.

<!-- slide -->
How good is this estimate? Well, if $\mathcal{D}$ is a random sample from $(X, Y)$, and $\texttt{tst}$ are randomly sampled observations randomly sampled from $i = 1, 2, \ldots, n$, then it is a reasonable estimate. However, it is rather variable due to the randomness of selecting the observations for the test set. How variable? It turns out, pretty variable. While it's a justified estimate, eventually we'll introduce cross-validation as a procedure better suited to performing this estimation to select a model.

<!-- slide -->
