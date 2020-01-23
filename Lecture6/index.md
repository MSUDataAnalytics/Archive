<!-- slide -->

# Assessing Our Toolkit

The follow is a reasonably set of questions to ask about a model:
1. Is **at least one** of the predictors $X_1, X_2, . . . , X_p$ useful in predicting the response?
2. Do **all** the predictors help to explain $Y$ , or is **only a subset** of the predictors useful?
3. **How well** does the model fit the data?
4. Given a set of predictor values, what response value should we **predict**, and **how accurate** is our prediction?

<!-- slide -->

## Is $X$ useful in predicting $Y$?


<!-- slide -->

## Could we merely use a subset of predictors?

This is a complicated question and a very common one in social science.

- The most direct approach is called **all subsets** or **best subsets regression**: we compute the least squares fit for all possible subsets and then choose between them based on some criterion that balances training error with model size.
- However we often can’t examine all possible models, since they are $2^p$ of them; for example when $p = 40$ there are over a billion potential models we could explore.
- Instead we need an automated approach that searches through a subset of them. We discuss two commonly use approaches next.

<!-- slide -->
