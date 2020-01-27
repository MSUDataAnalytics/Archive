<!-- slide -->
## Extending the Linear Model
Over these next few slides, we will remove two assumptions about additivity. These are related to **interactions** between variables and **non-linearity** in variables.

#### Interactions
- Let's extend our previous (fictional) analysis of the `bank.csv` data. Suppose we now want to explore how years of war service (`yearsservice`) affects balances.
- We could write the model:
\[y_i = \beta_0 + \beta_1 \mathbb{D}_{i1} + \beta_2 \mathbb{D}_{i2}+ \beta_3 \mathbb{D}_{i3} + \beta_4 (\texttt{yearsservice}) + \epsilon \]

**Try it:** What's the interpretation of the coefficient $\beta_4$?

<!-- slide -->
