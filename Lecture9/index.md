<!-- slide -->
<<<<<<< HEAD

# Nonparametric Regression



In these last few notes, we will continue to explore models for making **predictions**, but now we will introduce **nonparametric models** that will contrast the **parametric models** that we have used previously.

<!-- slide -->


Specifically, we will discuss:

- How to use **k-nearest neighbors** for regression through the use of the `knnreg()` function from the `caret` package
- How to use **decision trees** for regression through the use of the `rpart()` function from the `rpart` package.
- How "making predictions" can be thought of as **estimating the regression function**, that is, the conditional mean of the response given values of the features.
- What is the difference between **parametric** and **nonparametric** methods?
- How do these nonparametric methods deal with **categorical variables** and **interactions**.

We will also hint at, but delay one more chapter discussion of:

- What is **model flexibility**? What are **tuning parameters**? (What are model parameters?)
- What is **overfitting**? How do we avoid it?

<!-- slide -->


## Mathematical Setup

Let's return to the setup we defined previously. Consider a random variable $Y$ which represents a **response** variable, and $p$ **feature** variables $\boldsymbol{X} = (X_1, X_2, \ldots, X_p)$. We assume that the response variable $Y$ is some function of the features, plus some random noise.

$$
Y = f(\boldsymbol{X}) + \epsilon
$$

Our goal will is to find some $f$ such that $f(\boldsymbol{X})$ is close to $Y$. More specifically we want to minimize the risk under squared error loss.

$$
\mathbb{E}_{\boldsymbol{X}, Y} \left[ (Y - f(\boldsymbol{X})) ^ 2 \right] = \mathbb{E}_{\boldsymbol{X}} \mathbb{E}_{Y \mid \boldsymbol{X}} \left[ ( Y - f(\boldsymbol{X}) ) ^ 2 \mid \boldsymbol{X} = \boldsymbol{x} \right]
$$

We saw last lecture that this risk is minimized by the **conditional mean** of $Y$ given $\boldsymbol{X}$,

$$
\mu(\boldsymbol{x}) \triangleq \mathbb{E}[Y \mid \boldsymbol{X} = \boldsymbol{x}]
$$

which we called the **regression function**.

<!-- slide -->


Our goal then is to **estimate** this **regression function**. Let's return to the example from last chapter where we know the true probability model.

$$
Y = 1 - 2x - 3x ^ 2 + 5x ^ 3 + \epsilon
$$

where $\epsilon \sim \text{N}(0, \sigma^2)$.

Recall that this implies that the regression function is

$$
\mu(x) = \mathbb{E}[Y \mid \boldsymbol{X} = \boldsymbol{x}] = 1 - 2x - 3x ^ 2 + 5x ^ 3
$$

Let's also return to pretending that we do not actually know this information, but instead have some data, $(x_i, y_i)$ for $i = 1, 2, \ldots, n$.

<!-- slide -->

```{r}
cubic_mean = function(x) {
  1 - 2 * x - 3 * x ^ 2 + 5 * x ^ 3
}
```

```{r}
gen_slr_data = function(sample_size = 100, mu) {
  x = runif(n = sample_size, min = -1, max = 1)
  y = mu(x) + rnorm(n = sample_size)
  tibble(x, y)
}
```

```{r}
set.seed(1)
sim_slr_data = gen_slr_data(sample_size = 30, mu = cubic_mean)
```

```{r}
sim_slr_data %>%
  kable() %>%
  kable_styling(full_width = FALSE)
```
<!-- slide -->

Note that we simulated a bit more data than last time to make the "pattern" clearer to recognize.

```{r}
plot(sim_slr_data, pch = 20, col = "grey", cex = 2,
     main = "Simulated Data")
grid()
```

<!-- slide -->

Recall that when we used a linear model, we first need to make an **assumption** about the form of the regression function.

For example, we could assume that

$$
\mu(x) = \mathbb{E}[Y \mid \boldsymbol{X} = \boldsymbol{x}] = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3
$$

which is fit in R using the `lm()` function

```{r}
lm(y ~ x + I(x ^ 2) + I(x ^ 3), data = sim_slr_data)
```

<!-- slide -->

Notice that what is returned are (maximum likelihood or least squares) estimates of the unknown $\beta$ coefficients. That is, the "learning" that takes place with a linear models is "learning" the values of the coefficients.

For this reason, we call linear regression models **parametric** models. They have unknown **model parameters**, in this case the $\beta$ coefficients that must be learned from the data. The form of the regression function is assumed.

What if we don't want to make an assumption about the form of the regression function? While in this case, you might look at the plot and arrive at a reasonable guess of assuming a third order polynomial, what if it isn't so clear? What if you have 100 features? Making strong assumptions might not work well.

Enter **nonparametric** models. We will consider two examples: **k-nearest neighbors** and **decision trees**.
<!-- slide -->

## k-Nearest Neighbors

We'll start with **k-nearest neighbors** which is possibly a more intuitive procedure than linear models.

If our goal is to estimate the mean function,

$$
\mu(x) = \mathbb{E}[Y \mid \boldsymbol{X} = \boldsymbol{x}]
$$

the most natural approach would be to use

$$
\text{average}(\{ y_i : x_i = x \})
$$

that is, to estimate the conditional mean at $x$, average the $y_i$ values for each data point where $x_i = x$.
<!-- slide -->

While this sounds nice, it has an obvious flaw. For most values of $x$ there will not be any $x_i$ in the data where $x_i = x$!

So what's the next best thing? Pick values of $x_i$ that are "close" to $x$.

$$
\text{average}( \{ y_i : x_i \text{ equal to (or very close to) x} \} )
$$

This is the main idea behind many nonparametric approaches.

In the case of k-nearest neighbors we use

$$
\hat{\mu}_k(x) = \frac{1}{k} \sum_{ \{i \ : \ x_i \in \mathcal{N}_k(x, \mathcal{D}) \} } y_i
$$

as our estimate of the regression function at $x$. While this looks complicated, it is actually very simple. Here, we are using an average of the $y_i$ values of for the $k$ nearest neighbors to $x$.
<!-- slide -->

The $k$ "nearest" neighbors are the $k$ data points $(x_i, y_i)$ that have $x_i$ values that are nearest to $x$. We can define "nearest" using any distance we like, but unless otherwise noted, we are referring to euclidean distance. (The usual distance when you hear distance.) We are using the notation $\{i \ : \ x_i \in \mathcal{N}_k(x, \mathcal{D}) \}$ to define the observations that have $x_i$ values that are nearest to the value $x$ in a dataset $\mathcal{D}$, in other words, the $k$ nearest neighbors.

```{r}
plot_knn = function(k = 5, x, main = "main") {
  distances = dist(c(x, sim_slr_data$x))[1:30]
  nn = order(distances)[1:k]
  plot(sim_slr_data, pch = 20, col = "grey", cex = 2, main = main)
  grid()
  points(sim_slr_data$x[nn],
         sim_slr_data$y[nn],
         col = "green", cex = 1.8, lwd = 2)
  points(x = x, y = mean(sim_slr_data$y[nn]), pch = 4, lwd = 3)
}
```

The plots begin to illustrate this idea.

```{r}
par(mfrow = c(1, 3))
plot_knn(k = 3, x = -0.5, main = "k = 3, x = -0.5")
plot_knn(k = 5, x = 0, main = "k = 5, x = 0")
plot_knn(k = 9, x = 0.75, main = "k = 9, x = 0.75")
```

<!-- slide -->

- In the left plot, to estimate the mean of $Y$ at $x = -0.5$ we use the three nearest neighbors, which are highlighted with green. Our estimate is the average of the $y_i$ values of these three points indicated by the black x.
- In the middle plot, to estimate the mean of $Y$ at $x = 0$ we use the five nearest neighbors, which are highlighted with green. Our estimate is the average of the $y_i$ values of these five points indicated by the black x.
- In the right plot, to estimate the mean of $Y$ at $x = 0.75$ we use the nine nearest neighbors, which are highlighted with green. Our estimate is the average of the $y_i$ values of these nine points indicated by the black x.

<!-- slide -->
You might begin to notice a bit of an issue here. We have to do a new calculation each time we want to estimate the regression function at a different value of $x$! For this reason, k-nearest neighbors is often said to be "fast to train" and "slow to predict." Training, is instant. You just memorize the data! Prediction involves finding the distance between the $x$ considered and all $x_i$ in the data! (For this reason, KNN is often not used in practice, but it is very useful learning tool.)

<!-- slide -->

So, how then, do we choose the value of the **tuning** parameter $k$? We validate!

First, let's take a look at what happens with this data if we consider three different values of $k$.

```{r}
knn_slr_25 = knnreg(y ~ x, data = sim_slr_data, k = 25, use.all = TRUE)
knn_slr_05 = knnreg(y ~ x, data = sim_slr_data, k = 5, use.all = TRUE)
knn_slr_01 = knnreg(y ~ x, data = sim_slr_data, k = 1, use.all = TRUE)
```
<!-- slide -->



```{r}
par(mfrow = c(1, 3))

plot(sim_slr_data, pch = 20, col = "grey", cex = 2,
     main = "k = 25")
curve(cubic_mean(x), add = TRUE, lwd = 2, lty = 2)
curve(predict(knn_slr_25, tibble(x = x)),
      col = "firebrick", lwd = 2, lty = 1, add = TRUE, n = 10000)
grid()

plot(sim_slr_data, pch = 20, col = "grey", cex = 2,
     main = "k = 5")
curve(cubic_mean(x), add = TRUE, lwd = 2, lty = 2)
curve(predict(knn_slr_05, tibble(x = x)),
      col = "dodgerblue", lwd = 2, lty = 1, add = TRUE, n = 10000)
grid()

plot(sim_slr_data, pch = 20, col = "grey", cex = 2,
     main = "k = 1")
curve(cubic_mean(x), add = TRUE, lwd = 2, lty = 2)
curve(predict(knn_slr_01, tibble(x = x)),
      col = "limegreen", lwd = 2, lty = 1, add = TRUE, n = 10000)
grid()
```
<!-- slide -->


For each plot, the black dashed curve is the true mean function.

- In the left plot we use $k = 25$. The red "curve" is the estimate of the mean function for each $x$ shown in the plot.
- In the left plot we use $k = 5$. The blue "curve" is the estimate of the mean function for each $x$ shown in the plot.
- In the left plot we use $k = 1$. The green "curve" is the estimate of the mean function for each $x$ shown in the plot.

<!-- slide -->
Some things to notice here:

- The left plot with $k = 25$ is performing poorly. The estimated "curve" does not "move" enough. This is an example of an **inflexible** model.
- The right plot with $k = 1$ might not perform too well. The estimated "curve" seems to "move" too much. (Notice, that it goes through each point. We've fit to the noise.) This is an example of a **flexible** model.

While the middle plot with $k = 5$ is not "perfect" it seems to roughly capture the "motion" of the true regression function. We can begin to see that if we generated new data, this estimated regression function would perform better than the other two.

<!-- slide -->
But remember, in practice, we won't know the true regression function, so we will need to determine how our model performs using only the available data!

This $k$, the number of neighbors, is an example of a **tuning parameter**. Instead of being learned from the data, like model parameters such as the $\beta$ coefficients in linear regression, a tuning parameter tells us *how* to learn from data. It is user-specified. To determine the value of $k$ that should be used, many models are fit to the estimation data, then evaluated on the validation. Using the information from the validation data, a value of $k$ is chosen. (More on this in a bit.)
<!-- slide -->
This tuning parameter $k$ also defines the **flexibility** of the model. In KNN, a small value of $k$ is a flexible model, while a large value of $k$ is inflexible. (Many text use the term complex instead of flexible. We feel this is confusing as complex is often associated with difficult. KNN with $k = 1$ is actually a very simple model, but it is very flexible.)

Before moving to an example of tuning a KNN model, we will first introduce decision trees.
<!-- slide -->
## Decision Trees

Decision trees are similar to k-nearest neighbors but instead of looking for neighbors, decision trees create neighborhoods. We won't explore the full details of trees, but just start to understand the basic concepts, as well as learn to fit them in R.

Neighborhoods are created via recursive binary partitions. In simpler terms, pick a feature and a possible cutoff value. Data that have a value less than the cutoff for the selected feature are in one neighborhood (the left) and data that have a value greater than the cutoff are in another (the right). Within these two neighborhoods, repeat this procedure until a stopping rule is satisfied. (More on that in a moment.) To make a prediction, check which neighborhood a new piece of data would belong to and predict the average of the $y_i$ values of data in that neighborhood.

With the data above, which has a single feature $x$, consider three possible cutoffs: -0.5, 0.0, and 0.75.
<!-- slide -->
```{r}
plot_tree_split = function(cut = 0, main = "main") {
  plot(sim_slr_data, pch = 20, col = "grey", cex = 2, main = main)
  grid()
  abline(v = cut, col = "black", lwd = 2)
  left_pred = mean(sim_slr_data$y[sim_slr_data$x < cut])
  right_pred = mean(sim_slr_data$y[sim_slr_data$x > cut])
  segments(x0 = -2, y0 = left_pred, x1 = cut, y1 = left_pred, col = "limegreen", lwd = 2)
  segments(x0 = cut, y0 = right_pred, x1 = 2, y1 = right_pred, col = "firebrick", lwd = 2)
}
```

```{r}
par(mfrow = c(1, 3))
plot_tree_split(cut = -0.5, main = "cut @ x = -0.5")
plot_tree_split(cut = 0, main = "cut @ x = 0.0")
plot_tree_split(cut = 0.75, main = "cut @ x = 0.75")
```
<!-- slide -->
For each plot, the black vertical line defines the neighborhoods. The green horizontal lines are the average of the $y_i$ values for the points in the left neighborhood. The red horizontal lines are the average of the $y_i$ values for the points in the right neighborhood.

What makes a cutoff good? Large differences in the average $y_i$ between the two neighborhoods. More formally we want to find a cutoff value that minimizes

$$
\sum_{i \in N_L}(y_i - \hat{\mu}_{N_L}) ^ 2 + \sum_{i \in N_R}(y_i - \hat{\mu}_{N_R}) ^ 2
$$

where

- $N_L$ are the data in the left neighborhood
- $\hat{\mu}_{N_L}$ is the mean of the $y_i$ for data in the left neighborhood
<!-- slide -->
```{r}
calc_split_mse = function(cut = 0) {
  l_pred = mean(sim_slr_data$y[sim_slr_data$x < cut])
  r_pred = mean(sim_slr_data$y[sim_slr_data$x > cut])

  l_mse = round(sum((sim_slr_data$y[sim_slr_data$x < cut] - l_pred) ^ 2), 2)
  r_mse = round(sum((sim_slr_data$y[sim_slr_data$x > cut] - r_pred) ^ 2), 2)
  t_mse = round(l_mse + r_mse, 2)
  c("Total MSE" = t_mse, "Left MSE" = l_mse, "Right MSE" = r_mse)
}
```

```{r}
cbind(
  "Cutoff" = c(-0.5, 0.0, 0.75),
  rbind(
    calc_split_mse(cut = -0.5),
    calc_split_mse(cut = 0),
    calc_split_mse(cut = 0.75)
  )
) %>%
  kable() %>%
  kable_styling(full_width = FALSE)
```
<!-- slide -->
The table generated with the code above summarizes the results of the three potential splits. We see that (of the splits considered, which are not exhaustive) the split based on a cutoff of $x = -0.50$ creates the best partitioning of the space.

Now let's consider building a full tree.

```{r}
tree_slr = rpart(y ~ x, data = sim_slr_data)
```

```{r}
plot(sim_slr_data, pch = 20, col = "grey", cex = 2,
     main = "")
curve(cubic_mean(x), add = TRUE, lwd = 2, lty = 2)
curve(predict(tree_slr, tibble(x = x)),
      col = "darkorange", lwd = 2, lty = 1, add = TRUE, n = 10000)
grid()
```
<!-- slide -->
In the plot generated from the code above, the true regression function is the dashed black curve, and the solid orange curve is the estimated regression function using a decision tree. We see that there are two splits, which we can visualize as a tree.

```{r}
rpart.plot::rpart.plot(tree_slr)
```

The above "tree" shows the splits that were made. It informs us of the variable used, the cutoff value, and some summary of the resulting neighborhood. In "tree" terminology the resulting neighborhoods are "terminal nodes" of the tree. In contrast, "internal nodes" are neighborhoods that are created, but then further split.

The "root node" or neighborhood before any splitting is at the top of the plot. We see that this node represents 100% of the data. The other number, 0.21, is the mean of the response variable, in this case, $y_i$.

Looking at a terminal node, for example the bottom left node, we see that 23% of the data is in this node. The average value of the $y_i$ in this node is -1, which can be seen in the plot above.
<!-- slide -->
We also see that the first split is based on the $x$ variable, and a cutoff of $x = -0.52$. (Note that because there is only one variable here, all splits are based on $x$, but in the future, we will have multiple features that can be split and neighborhoods will no longer be one-dimensional. However, this is hard to plot.)

END HERE. 
<!-- slide -->

Let's build a bigger, more flexible tree.

```{r, echo = FALSE}
tree_slr = rpart(y ~ x, data = sim_slr_data, cp = 0, minsplit = 5)
```

```{r, echo = FALSE}
plot(sim_slr_data, pch = 20, col = "grey", cex = 2,
     main = "")
curve(cubic_mean(x), add = TRUE, lwd = 2, lty = 2)
curve(predict(tree_slr, tibble(x = x)),
      col = "darkorange", lwd = 2, lty = 1, add = TRUE, n = 10000)
grid()
```

```{r, echo = FALSE}
rpart.plot::rpart.plot(tree_slr)
```

There are two tuning parameters at play here which we will call by their names in R which we will see soon:

- `cp` or the "complexity parameter" as it is called. (Flexibility parameter would be a better name.) This parameter determines which splits are considered. A split must improve the performance of the tree by more than `cp` in order to be considered. When we get to R, we will see that the default value is 0.1.
- `minsplit`, the minimum number of observations in a node (neighborhood) in order to split again.

There are actually many more possible tuning parameters for trees, possibly differing depending on who wrote the code you're using. We will limit discussion to these two. Note that they effect each other, and they effect other parameters which we are not discussing. The main takeaway should be how they effect model flexibility.

First let's look at what happens for a fixed `minsplit` by variable `cp`.

```{r, echo = FALSE}
tree_slr_cp_10 = rpart(y ~ x, data = sim_slr_data, cp = 0.10, minsplit = 2)
tree_slr_cp_05 = rpart(y ~ x, data = sim_slr_data, cp = 0.05, minsplit = 2)
tree_slr_cp_00 = rpart(y ~ x, data = sim_slr_data, cp = 0.00, minsplit = 2)
```
=======
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
>>>>>>> 310229356d99127def6d516c7c3e38a242fa8ab7

```{r, fig.height = 4, fig.width = 12, echo = FALSE}
par(mfrow = c(1, 3))

<<<<<<< HEAD
plot(sim_slr_data, pch = 20, col = "grey", cex = 2,
     main = "cp = 0.10, minsplit = 2")
curve(cubic_mean(x), add = TRUE, lwd = 2, lty = 2)
curve(predict(tree_slr_cp_10, tibble(x = x)),
      col = "firebrick", lwd = 2, lty = 1, add = TRUE, n = 10000)
grid()

plot(sim_slr_data, pch = 20, col = "grey", cex = 2,
     main = "cp = 0.05, minsplit = 2")
curve(cubic_mean(x), add = TRUE, lwd = 2, lty = 2)
curve(predict(tree_slr_cp_05, tibble(x = x)),
      col = "dodgerblue", lwd = 2, lty = 1, add = TRUE, n = 10000)
grid()

plot(sim_slr_data, pch = 20, col = "grey", cex = 2,
     main = "cp = 0.00, minsplit = 2")
curve(cubic_mean(x), add = TRUE, lwd = 2, lty = 2)
curve(predict(tree_slr_cp_00, tibble(x = x)),
      col = "limegreen", lwd = 2, lty = 1, add = TRUE, n = 10000)
grid()
```

We see that as `cp` *decreases*, model flexibility **increases**.

Now the reverse, fix `cp` and vary `minsplit`.

```{r, echo = FALSE}
tree_slr_ms_25 = rpart(y ~ x, data = sim_slr_data, cp = 0.01, minsplit = 25)
tree_slr_ms_10 = rpart(y ~ x, data = sim_slr_data, cp = 0.01, minsplit = 10)
tree_slr_ms_02 = rpart(y ~ x, data = sim_slr_data, cp = 0.01, minsplit = 02)
```
=======
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
>>>>>>> 310229356d99127def6d516c7c3e38a242fa8ab7

```{r, fig.height = 4, fig.width = 12, echo = FALSE}
par(mfrow = c(1, 3))

<<<<<<< HEAD
plot(sim_slr_data, pch = 20, col = "grey", cex = 2,
     main = "cp = 0.01, minsplit = 25")
curve(cubic_mean(x), add = TRUE, lwd = 2, lty = 2)
curve(predict(tree_slr_ms_25, tibble(x = x)),
      col = "firebrick", lwd = 2, lty = 1, add = TRUE, n = 10000)
grid()

plot(sim_slr_data, pch = 20, col = "grey", cex = 2,
     main = "cp = 0.01, minsplit = 10")
curve(cubic_mean(x), add = TRUE, lwd = 2, lty = 2)
curve(predict(tree_slr_ms_10, tibble(x = x)),
      col = "dodgerblue", lwd = 2, lty = 1, add = TRUE, n = 10000)
grid()

plot(sim_slr_data, pch = 20, col = "grey", cex = 2,
     main = "cp = 0.01, minsplit = 2")
curve(cubic_mean(x), add = TRUE, lwd = 2, lty = 2)
curve(predict(tree_slr_ms_02, tibble(x = x)),
      col = "limegreen", lwd = 2, lty = 1, add = TRUE, n = 10000)
grid()
```

We see that as `minsplit` *decreases*, model flexibility **increases**.

***

## Example: Credit Card Data

Let's return to the credit card data from the previous chapter. While last time we used the data to inform a bit of analysis, this time we will simply use the dataset to illustrate some concepts.

```{r}
# load data, coerce to tibble
crdt = as_tibble(ISLR::Credit)
```

Again, we are using the `Credit` data form the `ISLR` package. Note: **this is not real data.** It has been simulated.

```{r}
# data prep
crdt = crdt %>%
  select(-ID) %>%
  select(-Rating, everything())
```

We remove the `ID` variable as it should have no predictive power. We also move the `Rating` variable to the last column with a clever `dplyr` trick. This is in no way necessary, but is useful in creating some plots.

```{r}
# test-train split
set.seed(1)
crdt_trn_idx = sample(nrow(crdt), size = 0.8 * nrow(crdt))
crdt_trn = crdt[crdt_trn_idx, ]
crdt_tst = crdt[-crdt_trn_idx, ]
```

```{r}
# estimation-validation split
crdt_est_idx = sample(nrow(crdt_trn), size = 0.8 * nrow(crdt_trn))
crdt_est = crdt_trn[crdt_est_idx, ]
crdt_val = crdt_trn[-crdt_est_idx, ]
```

After train-test and estimation-validation splitting the data, we look at the train data.

```{r}
# check data
head(crdt_trn, n = 10)
```

Recall that we would like to predict the `Rating` variable. This time, let's try to use only demographic information as predictors. In particular, let's focus on `Age` (numeric), `Gender` (categorical), and `Student` (categorical).

Let's fit KNN models with these features, and various values of $k$. To do so, we use the `knnreg()` function from the `caret` package. (There are many other KNN functions in R. However, the operation and syntax of `knnreg()` better matches other functions we will use in this course.) Use `?knnreg` for documentation and details.

```{r}
crdt_knn_01 = knnreg(Rating ~ Age + Gender + Student, data = crdt_est, k = 1)
crdt_knn_10 = knnreg(Rating ~ Age + Gender + Student, data = crdt_est, k = 10)
crdt_knn_25 = knnreg(Rating ~ Age + Gender + Student, data = crdt_est, k = 25)
```

Here, we fit three models to the estimation data. We supply the variables that will be used as features as we would with `lm()`. We also specify how many neighbors to consider via the `k` argument.

But wait a second, what is the distance from non-student to student? From male to female? In other words, how does KNN handle categorical variables? It doesn't! Like `lm()` it creates dummy variables under the hood.

**Note:** To this point, and until we specify otherwise, we will always coerce categorical variables to be factor variables in R. We will then let modeling functions such as `lm()` or `knnreg()` deal with the creation of dummy variables internally.

```{r}
head(crdt_knn_10$learn$X)
```

Once these dummy variables have been created, we have a numeric $X$ matrix, which makes distance calculations easy. For example, the distance between the 4th and 5th observation here is `r round(dist(head(crdt_knn_10$learn$X))[10], 2)`.

```{r}
dist(head(crdt_knn_10$learn$X))
```

```{r}
sqrt(sum((crdt_knn_10$learn$X[4, ] - crdt_knn_10$learn$X[5, ]) ^ 2))
```

(What about interactions? Basically, you'd have to create them the same way as you do for linear models. We only mention this to contrast with trees in a bit.)

OK, so of these three models, which one performs best? (Where for now, "best" is obtaining the lowest validation RMSE.)

First, note that we return to the `predict()` function as we did with `lm()`.

```{r}
predict(crdt_knn_10, crdt_val[1:5, ])
```

This uses the 10-NN (10 nearest neighbors) model to make predictions (estimate the regression function) given the first five observations of the validation data. (**Note:** We did not name the second argument to `predict()`. Again, you've been warned.)

Now that we know we already know how to use the `predict()` function, let's calculate the validation RMSE for each of these models.

```{r}
knn_mod_list = list(
  crdt_knn_01 = knnreg(Rating ~ Age + Gender + Student, data = crdt_est, k = 1),
  crdt_knn_10 = knnreg(Rating ~ Age + Gender + Student, data = crdt_est, k = 10),
  crdt_knn_25 = knnreg(Rating ~ Age + Gender + Student, data = crdt_est, k = 25)
)
```

```{r}
knn_val_pred = map(knn_mod_list, predict, crdt_val)
```

```{r}
calc_rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}
```

```{r}
map_dbl(knn_val_pred, calc_rmse, crdt_val$Rating)
```

So, of these three values of $k$, the model with $k = 25$ achieves the lowest validation RMSE.

This process, fitting a number of models with different values of the *tuning parameter*, in this case $k$, and then finding the "best" tuning parameter value based on performance on the validation data is called **tuning**. In practice, we would likely consider more values of $k$, but this should illustrate the point.

In the next chapters, we will discuss the details of model flexibility and model tuning, and how these concepts are tied together. However, even though we will present some theory behind this relationship, in practice, **you must tune and validate your models**. There is no theory that will inform you ahead of tuning and validation which model will be the best. By teaching you *how* to fit KNN models in R and how to calculate validation RMSE, you already have all the tools you need to find a good model.

Let's turn to decision trees which we will fit with the `rpart()` function from the `rpart` package. Use `?rpart` and `?rpart.control` for documentation and details.

We'll start by using default tuning parameters.

```{r}
crdt_tree = rpart(Rating ~ Age + Gender + Student, data = crdt_est)
```

```{r}
crdt_tree
```

Above we see the resulting tree printed, however, this is difficult to read. Instead, we use the `rpart.plot()` function from the `rpart.plot` package to better visualize the tree.

```{r}
rpart.plot(crdt_tree)
```

At each split, the variable used to split is listed together with a condition. (If the condition is true for a data point, send it to the left neighborhood.) Although the `Gender` variable was used, we only see splits based on `Age` and `Student`. (This hints at the relative importance of these variables for prediction. More on this much later.)

Categorical variables are split based on potential categories! This is **excellent**. This means that trees naturally handle categorical features without needing to convert to numeric under the hood. We see a split that puts students into one neighborhood, and non-students into another.

Notice that the splits happen in order. So for example, the third terminal node (with an average rating of 298) is based on splits of:

- `Age < 83`
- `Age < 70`
- `Age > 39`
- `Student = Yes`

In other words, individuals in this terminal node are students who are between the ages of 39 and 70. (Only 5% of the data is represented here.) This is basically an interaction between `Age` and `Student` without any need to directly specify it! What a great feature of trees.

To recap:

- Trees do not make assumptions about the form of the regression function.
- Trees automatically handle categorical features.
- Trees naturally incorporate interaction.

Now let's fit another tree that is more flexible by relaxing some tuning parameters. (By default, `cp = 0.1` and `minsplit = 20`.)

```{r}
crdt_tree_big = rpart(Rating ~ Age + Gender + Student, data = crdt_est,
                      cp = 0.0, minsplit = 20)
```

```{r}
rpart.plot(crdt_tree_big)
```

To make the tree even bigger, we could reduce `minsplit`, but in practice we mostly consider the `cp` parameter. (We will occasionally modify the `minsplit` parameter on quizzes.) Since `minsplit` has been kept the same, but `cp` was reduced, we see the same splits as the smaller tree, but many additional splits.

Now let's fit a bunch of trees, with different values of `cp`, for tuning.

```{r}
tree_mod_list = list(
  crdt_tree_0000 = rpart(Rating ~ Age + Gender + Student, data = crdt_est, cp = 0.000),
  crdt_tree_0001 = rpart(Rating ~ Age + Gender + Student, data = crdt_est, cp = 0.001),
  crdt_tree_0010 = rpart(Rating ~ Age + Gender + Student, data = crdt_est, cp = 0.010),
  crdt_tree_0100 = rpart(Rating ~ Age + Gender + Student, data = crdt_est, cp = 0.100)
)
```

```{r}
tree_val_pred = map(tree_mod_list, predict, crdt_val)
```

```{r}
map_dbl(tree_val_pred, calc_rmse, crdt_val$Rating)
```

Here we see the least flexible model, with `cp = 0.100`, performs best.

Note that by only using these three features, we are severely limiting our models performance. Let's quickly assess using all available predictors.

```{r}
crdt_tree_all = rpart(Rating ~ ., data = crdt_est)
```

```{r}
rpart.plot(crdt_tree_all)
```

Notice that this model **only** splits based on `Limit` despite using all features. (This should be a big hint about which variables are useful for prediction.)

```{r}
calc_rmse(
  actual = crdt_val$Rating,
  predicted = predict(crdt_tree_all, crdt_val)
)
```

This model performs much better. You should try something similar with the KNN models above. Also, consider comparing this result to results from last chapter using linear models.

Notice that we've been using that trusty `predict()` function here again.

```{r}
predict(crdt_tree_all, crdt_val[1:5, ])
```

What does this code do? It estimates the mean `Rating` given the feature information (the "x" values) from the first five observations from the validation data using a decision tree model with default tuning parameters. Hopefully a theme is emerging.

***

## Source

- `R` Markdown: [`07-nonparametric-regression.Rmd`](07-nonparametric-regression.Rmd)

***
=======
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
>>>>>>> 310229356d99127def6d516c7c3e38a242fa8ab7
