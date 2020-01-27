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

## Other Considerations for Linear Regressions

**Qualitative Predictors**
- Some predictors are not quantitative but are qualitative,
taking a discrete set of values.
- These are also called categorical predictors or factor
variables.
- You have explored these a bit already. See, for instance `marital`, `education`, or `job` in the bank data from the previous lecture.
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
