<!-- slide -->

# Nonparametric Regression

> I see the future better than your computers
> Apple software got the next 10 years scared I'm Fred Krueger
> --- **Kendrick Lamar**


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

<!-- slide -->
