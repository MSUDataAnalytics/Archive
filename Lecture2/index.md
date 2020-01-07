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

You must let me know your team by this coming Tuesday.

If you fail to report your team, then you will

<!-- slide -->
# More on Teams

- Your team must come up with a name and a Github site for your project.
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

# ![GDP](/assets/gdp.jpg)

<!-- slide -->
#Social Science Example

# ![USGDP](/assets/us_gdp.png)

<!-- slide -->
# Formalization

Here `emissions` is a **response** or **target** that we wish to predict.

We generically refer to the response as $Y$.

`GDP` is a **feature**, or **input**, or **predictor**, or **regressor**; call it $X_1$.

Likewise let's test our postulate and call `westernhem` our $X_2$, and so on.

We can refer to the input vector collectively as

$$X = \begin{bmatrix} x_{11} & x_{12}\\
x_{21} & x{22} \\
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
- This form of ``incremental learning'' will pop up a lot.

> **Theorem:** If the data can be fit by a linear separator, then after some finite number of steps, the perceptron learning algorithm will find one.

...but after how many steps and what if it can't be separated and is there a faster way?




<!-- slide -->
# Human Learning: a "Test"
# ![Percep1](/assets/outside.png)



<!-- slide -->
\frametitle{Outside the Data}

An easy visual learning problem is seemingly very messy.

 pace*{0.2in}

For every $f$ that fits the data and is ``+1'' on the new point, there is one that is ``−1.''

 pace*{0.2in}
Since f is unknown, it can take on any value outside the data, no matter how large the data.

\begin{itemize}
	\item This is called \textit{No Free Lunch}.
\end{itemize}

 pace*{0.2in}

You cannot know anything for sure about $f$ outside the data without making assumptions.

 pace*{0.2in}

Is there any hope to know anything about $f$ outside the data set without making assumptions about $f$?
\begin{center}
	\alert{Yes, if we are willing to give up the ``for sure.''}
\end{center}



<!-- slide -->
\frametitle{The Parable of the Marbles}
\begin{figure}
	\centering
	\includegraphics[width=0.3\linewidth]{graphics/marbles}
\end{figure}

Within this bag of marbles are green marbles ~\greencirc ~ and white marbles \whitecirc

 pace*{0.2in}

We are going to pick a sample of $n$ marbles (with replacement).


<!-- slide -->
\frametitle{The Parable of the Marbles}
Consider a sample composed of  ~\greencirc \greencirc \greencirc \whitecirc \greencirc \whitecirc \greencirc

 pace*{0.2in}
\begin{itemize}
	\item[] Let $\mu$ be the \underline{objective} probability to pick a white marble.
	\item[] Let $\nu$ be fraction of white marbles in the sample.
\end{itemize}

 pace*{0.2in}
\textbf{Question:} Can we say anything about $\mu$ (outside the data) after observing $\nu$ (the data)?
\begin{itemize}
	\item[] \alert{No.} It is possible for the sample to be all green marbles and the bag to be mostly white.
\end{itemize}

 pace*{0.2in}
\textbf{Question:} Then why do we do polling (e.g. to predict the outcome of the presidential election)?
\begin{itemize}
	\item[] The bad case is \textit{possible}, but not \textit{probable}.
\end{itemize}



<!-- slide -->
\frametitle{Hoeffding's Inequality}

\textbf{Hoeffding's Inequality} states, loosely, that $\nu$ cannot be too far from $\mu$.

 pace*{0.2in}

\begin{theorem}[Hoeffding's Inequality]
\begin{center}
	$\mathbb{P} \left [ | \nu - \mu |  > \epsilon \right ] \leq  2 e^{-2\epsilon^2 n}$
\end{center}
\end{theorem}

 pace*{0.2in}

$\nu \approx \mu$ is called \textit{probably approximately correct} (PAC-learning)



<!-- slide -->
\frametitle{Hoeffding's Inequality: Example}
\textbf{Example:} $n = 1, 000$; draw a sample and observe $\nu$.
	 pace*{0.1in}

\begin{itemize}
	\item 99\% of the time $\mu - 0.05 \leq \nu \leq \mu + 0.05$
	\item[] (This is implied from setting $\epsilon = 0.05$ and using given $n$)
	 pace*{0.1in}
	\item 99.9999996\% of the time $\mu - 0.10 \leq \nu \leq \mu + 0.10$
\end{itemize}

 pace*{0.1in}

\textbf{What does this mean?}

 pace*{0.05in}

If I repeatedly pick a sample of size 1,000, observe $\nu$ and claim that
$\mu \in  [\nu - 0.05, \nu + 0.05]$ (or that the error bar is $\pm 0.05$) I will be right 99\% of the time.

 pace*{0.2in}

On any particular sample you may be wrong, but not often.


<!-- slide -->
\frametitle{Extending the Example}

\textbf{Critical requirement:} samples must be independent.
\begin{itemize}
	\item If the sample is constructed in some arbitrary fashion, then indeed we cannot say anything.
\end{itemize}

 pace*{0.2in}

Even with independence, $\nu$ can take on arbitrary values.
\begin{itemize}
	\item Some values are way more likely than others. This is what allows us to learn something – it is likely that $\nu \approx \mu$.
\end{itemize}

 pace*{0.2in}

The bound $2 e^{-2\epsilon^2 n}$ does not depend on $\mu$ or the size of the bag.
\begin{itemize}
	\item The bag can be infinite.
	\item It’s great that it does not depend on $\mu$ because $\mu$ is \textbf{unknown}.
\end{itemize}

 pace*{0.2in}

The key player in the bound $2 e^{-2\epsilon^2 n}$ is $n$.
\begin{itemize}
	\item If $n \to \infty, \nu \approx \mu$ with very very high probabilty, but not for sure.
\end{itemize}



<!-- slide -->
\frametitle{Learning a Target Function}
In learning, the unknown is an entire function $f$; in the bag it was a single number $\mu$.

 pace*{0.2in}

\begin{figure}
	\centering
	\includegraphics[width=0.5\linewidth]{graphics/targfunc}
\end{figure}





<!-- slide -->
\frametitle{Learning a Target Function}
\begin{figure}
	\centering
	\visible<1->{\includegraphics[width=0.4\linewidth]{graphics/targfunc2}} \\
	\visible<2->{\includegraphics[width=0.4\linewidth]{graphics/targfunc3}}
\end{figure}

\visible<3>{White area in second figure: $h(x) = f(x)$

 pace*{0.1in}

Green area in second figure: $h(x) \neq f(x)$}




<!-- slide -->
\frametitle{Learning a Target Function}

Define the following notion:
\[
\E (h) = \mathbb{P}_x \left [ h(x) \neq f(x) \right ]
\]

That is, this is the ``size'' of the green region.

 pace*{0.2in}

\begin{itemize}
	\item White ``marble'': $h(x) = f(x)$
	\item Green ``marble'': $h(x) \neq f(x)$.
\end{itemize}

 pace*{0.1in}
We can re-frame $\mu, \nu$ in terms of $\E(h) = \mathbb{P}_x \left [ h(x) \neq f(x) \right ]$




<!-- slide -->
\frametitle{Closing the Metaphor}

\textbf{Out-of-sample error:} $\E_\text{out}(h) = \mathbb{P}_x \left [ h(x) \neq f(x) \right ]$

 pace*{0.2in}

\textbf{In-sample error:} $\E_\text{in}(h) = \frac{1}{n} \sum_{i=1}^n \mathbb{I} \left [ h(x) \neq f(x) \right ]$

 pace*{0.2in}
Hoeffding's Inequality, restated:
\[
\mathbb{P} \left [ |\E_\text{in}(h)  - \E_\text{out}(h)  |  > \epsilon \right ] \leq  2 e^{-2\epsilon^2 n}
\]

Victory! If we just minimize in-sample error, we are likely to be right out of sample!

 pace*{0.1in}

\qquad \qquad {\tiny ...right?}



<!-- slide -->
\frametitle{Verification vs Learning}
The entire previous argument assumed a FIXED $h$ and then came the data.

 pace*{0.2in}
Given $h \in \mathcal{H}$, a sample can verify whether or not it is good (w.r.t. $f$):
\begin{itemize}
	\item if $\E_\text{in}$ is small, $h$ is good, with high confidence.
	\item if $\E_\text{in}$ is large, $h$ is bad with high confidence.
\end{itemize}

 pace*{0.2in}
In this world: we have no control over $\E_\text{in}$.

 pace*{0.2in}

In learning, you actually try to fit the data!
\begin{itemize}
	\item e.g., perceptron model $g$ results from searching an entire hypothesis set $\mathcal{H}$ for a hypothesis with small $\E_\text{in}$.
\end{itemize}
