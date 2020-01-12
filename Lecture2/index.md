<!-- slide -->
# Group Projects

As previousy mentioned, your "final" is a group project.

Accordingly, you need to start planning relatively soon.

To aid your planning, here are the required elements of that project:

1. You must find existing data to analyze. Aggregating data from multiple sources encouraged but not required.
2. You must visualize 3 interesting features of that data.
3. You must come up with some analysis---using tools from the course---which relates your data to either a prediction or a policy conclusion.
4. You must present your analysis as if presenting to a C-suite executive.

<!-- slide -->
# Teams

You must let me know your team by this Sunday. This will allow us to assign teams by next Tuesday.

If you fail to report your team, then you will be added to the "willing to be randomly assigned" pool.

**The course website** has a survey to help aid us in putting together teams.

<!-- slide -->
# More on Teams

- Your team must come up with a name and a Github site for your project and labs.
- Your team will earn the same scores on all projects and labs.
  - Labs can receive either 4,8, or 10 points (out of 10).
- Teams will only submit one write-up.
- For attendance score: one member of a few teams (chosen at random) will present their working code and analysis. I'll select the team, then the team is free to send whomever they like to present their working code and discuss their output. If it doesn't work, the whole team is punished.

To combat additional freeloading, we will use a reporting system. Any team member can email me to report another team member's lack of participation secretly. Two strikes will result in a 10% grade deduction; three strikes will result in a 20% deduction.



<!-- slide -->
# Learning from Data

The following are the basic requirements for statistical learning:

1. A pattern exists.
2. This pattern is not easily expressed in a closed mathematical form.
3. You have data.

<!-- slide -->
# Social Science Example

@import "/assets/gdp.jpg" {width="700px"}
<!-- slide -->
# Social Science Example

@import "/assets/us_gdp.png" {width="700px"}

<!-- slide -->
# Formalization

Here `emissions` is a **response** or **target** that we wish to predict.

We generically refer to the response as $Y$.

`GDP` is a **feature**, or **input**, or **predictor**, or **regressor**; call it $X_1$.

Likewise let's test our postulate and call `westernhem` our $X_2$, and so on.

We can refer to the input vector collectively as

$$X = \begin{bmatrix} x_{11} & x_{12}\\
x_{21} & x_{22} \\
x_{31} & x_{32} \\
\vdots & \vdots
\end{bmatrix}$$

We are seeking some unknow function that maps $X$ to $Y$.

Put another way, we are seeking to explain $Y$ as follows:
\[
Y = f (X) + \epsilon
\]

<!-- slide -->
# Formalization

We call the function $f : \mathcal{X} \to \mathcal{Y}$ the **target function**.

The target function is **always unknown**. It is the object of learning.

### Methodology:
- Observe data $(x_1, y_1) \dots (x_N, y_N)$.
- Use some algorithm to approximate $f$.
- Produce final hypothesis function $g \approx f$.
- Evaluate how well $g$ approximates $f$; iterate as needed.


<!-- slide -->
# What is the Purpose of $g(X)$?

With a good estimate of $f$ we can make predictions of $Y$ at new points $X = x$.


We can understand which components of $X = (X_1, X_2, \dots , X_m)$ are important in explaining $Y$ , and which are (potentially) irrelevant.

- e.g., `GDP` and `yearsindustrialized` have a big impact on `emissions`, but `hydroutilization` typically does not.

Depending on the complexity of $f$, we may be able to meaningfully understand how each component of $X$ affects $Y$.
(But we should be careful about assigning causal interpretations)

<!-- slide -->
# The Learning Problem
A "solution" to the learning problem does **not** consist of $g$.

Rather, the solutions are the algorithm and the hypotheses that the algorithm may choose from---aka the **hypothesis set**, denoted $\mathcal{H}$.

- While the final guess is $g$, a generic member of $\mathcal{H}$ is $h$.

The algorithm and hypothesis set are inseparable.

For example, if one restricts attention to hypotheses that take a linear form, then the hypothesis set could be functions such that

\[
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots \beta_m X_m + \epsilon
\]


<!-- slide -->
# Reducible vs Irreducible Error

Suppose we want to minimize the squared difference between our predictions and the truth.


That is, we wish to minimize:
\[
 E \left( Y - \hat{Y} \right )^2 = E \left( f(X) + \epsilon - g(X)  \right) ^2 \\
 = E \left( f(X) - g(X)  \right) ^2 + E (\epsilon)^2
\]

Note $E (\epsilon)^2 = \text{var}(\epsilon)$. This is the **irreducible error** in the learning problem.

The term $ E \left( f(X) - g(X)  \right) ^2$ represents the **reducible error** in the problem.

<!-- slide -->
# Binary Classification

Examining binary outcomes: `signedKyotoProtocol` is our response, coded as $\pm 1$.

Given some input vector $X = (X_1,\dots,X_m )$, we categorize

\[
\sum_{j=1}^n \sum_{i=1}^m w_i^j x_i^j > ~ \text{some threshold},
\]

 as "likely" members of Kyoto Protocol.

- How to choose the importance weights} $w_i$
  - Give importance weights to the different inputs and compute a ``score".
  - Determine likely signatory if ``score'' is acceptable.
    - input $x_i$ is important (e.g., `G8country`) $\rightarrow$ large weight $|w_i|$
    - input $x_i$ beneficial (e.g., `inEurope`) $\rightarrow$ $w_i>0$.






<!-- slide -->
# Linear Learning
A simple form of binary learning takes the following mathematical form:

\[
\text{Categorize as signer if} \quad \sum_{j=1}^n  \sum_{i=1}^m w_i^j x_i^j > ~ \text{some threshold}, \]


\[\text{Categorize as non-signer if} \quad \sum_{j=1}^n  \sum_{i=1}^m w_i^j x_i^j < ~ \text{some threshold.}
\]


This can be formally written as

\[
h(X) = \text{sign}\left ( \left (\sum_{j=1}^n  \sum_{i=1}^m  w_i^j x_i^j + w_0  \right ) \right )
\]


where the "bias weight" $w_0$ corresponds to the threshold.




<!-- slide -->
# Linear Learning
This is equivalent to a hypothesis set $\mathcal{H} = \left \{ h(X) = \text{sign}\left (W^T X \right ) \right \}$.


\[
X = \begin{pmatrix}
	1 \\
	X_{1} \\
	\vdots \\
	X_{m}
\end{pmatrix}
\]
\[
W = \begin{pmatrix}
w_0 \\
w_{1} \\
\vdots \\
w_{m}
\end{pmatrix}
\]

This hypothesis set is called the **linear separator**.



<!-- slide -->
# Geometric / Visual Interpretation
# ![Percep1](/assets/percep1.png)

<!-- slide -->
# Geometric / Visual Interpretation
# ![Percep1](/assets/percep2.png)



<!-- slide -->
# Perceptron Learning Algorithm

A **perceptron** predicts the data by using a line or a plane to separate the `red` from `blue` data.

#### Fitting the data
How to find a hyperplane that separates the data?
- "It's obvious - just look at the data and draw the line," is not a valid solution.

We want to select $g \in \mathcal{H}$ such that $g \approx f$.

We **certainly** want $g \approx f$ on the data set $\mathcal{D}$.
- Ideally, $g(x) = y$ for all $n$ data-points.

How do we find such a $g$ in the infinite hypothesis set $\mathcal{H}$, if it exists?

$\Rightarrow$ Start with some weight vector and try to improve it.



<!-- slide -->
# Perceptron Learning Algorithm
A simple iterative method in `psuedocode`:

0. `set` the values `red` = -1, `blue` = +1
1. `initialize` $w(1)=0$
2. `for` each iteration $t = 1,2,3,\dots$ where the weight vector is $w(t)$
  - `choose` one misclassified example $(x_1, y_1), \dots , (x_n , y_n )$
  - Let's call the misclassified example $(x_*, y_*)$.
  - That is, `sign`$\left (w(t) \cdot x_* \right ) \neq y_*$.
  - `update` the weight such that:
\[ w(t + 1) = w(t) + y_* x_* \]



<!-- slide -->
# Perceptron Learning: Success?

PLA implements our idea: start at some weights and try to improve.
- This form of "incremental learning" will pop up a lot.

'" "'> **Theorem:** If the data can be fit by a linear separator, then after some finite number of steps, the perceptron learning algorithm will find one.

...but after how many steps and what if it can't be separated and is there a faster way?




<!-- slide -->
# Human Learning: a "Test"
# ![Percep1](/assets/outside.png)



<!-- slide -->
# Outside the Data

An easy visual learning problem is seemingly very messy.

For every $f$ that fits the data and is ``+1'' on the new point, there is one that is ``−1.''

Since $f$ is unknown, it can take on any value outside the data, no matter how large the data.

- This is called *No Free Lunch*.

You cannot know anything for sure about $f$ outside the data without making assumptions.

Is there any hope to know anything about $f$ outside the data set without making assumptions about $f$?

**Yes**, if we are willing to give up the "for sure."

<!-- slide -->

# The Parable of the Marbles
# ![marbles](/assets/marbles.jpg)

Within this bag of marbles are $\clubsuit$ and $\diamondsuit$ marbles

We are going to pick a sample of $n$ marbles (with replacement).

<!-- slide -->
# The Parable of the Marbles
Consider a sample composed of  $~\clubsuit~\clubsuit~\clubsuit~\diamondsuit~\clubsuit~\diamondsuit~\clubsuit$

- Let $\mu$ be the **objective** probability to pick a $\clubsuit$.
- Let $\nu$ be fraction of $\clubsuit$ marbles in the sample.

**Question:** Can we say anything about $\mu$ (outside the data) after observing $\nu$ (the data)?

- No. It is possible for the sample to be all $\clubsuit$ marbles and the bag to be $\approx \diamondsuit$.

**Question:** Then why do we do polling (e.g. to predict the outcome of the presidential election)?
- The bad case is *possible*, but not **probable**.

<!-- slide -->
# Hoeffding's Inequality

**Hoeffding's Inequality** states, loosely, that $\nu$ cannot be too far from $\mu$.

### Theorem (Hoeffding's Inequality)
$$
\mathbb{P} \left [ | \nu - \mu |  > \epsilon \right ] \leq  2 e^{-2\epsilon^2 n}
$$

$\nu \approx \mu$ is called **probably approximately correct** (PAC-learning)

<!-- slide -->
# Hoeffding's Inequality: Example
**Example:** $n = 1, 000$; draw a sample and observe $\nu$.


- 99\% of the time $\mu - 0.05 \leq \nu \leq \mu + 0.05$
- (This is implied from setting $\epsilon = 0.05$ and using given $n$)
- 99.9999996\% of the time $\mu - 0.10 \leq \nu \leq \mu + 0.10$ %


**What does this mean?**

If I repeatedly pick a sample of size 1,000, observe $\nu$ and claim that
$\mu \in  [\nu - 0.05, \nu + 0.05]$ (or that the error bar is $\pm 0.05$) I will be right 99\% of the time.

On any particular sample you may be wrong, but not often.
