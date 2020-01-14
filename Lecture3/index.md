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

<!-- slide -->
# In-Class Exercise

1. Suppose a bag is filled with $\clubsuit$x9 and $\diamondsuit$x1.

2. Calculate the odds of drawing 3 $\diamondsuit$ in a row (obviously, with replacement).

3. Provide the formula for $n$ draws in a row of $\diamondsuit$.

4. Show that Hoeffding's Inequality holds for six consecutive draws of $\diamondsuit$.

<!-- slide -->


# Extending the Example

- **Critical requirement:** samples must be independent.

- If the sample is constructed in some arbitrary fashion, then indeed we cannot say anything.

Even with independence, $\nu$ can take on arbitrary values.

- Some values are way more likely than others. This is what allows us to learn something – it is likely that $\nu \approx \mu$.
- The bound $2 e^{-2\epsilon^2 n}$ does not depend on $\mu$ or the size of the bag.
- The bag can be infinite.
- It’s great that it does not depend on $\mu$ because $\mu$ is **unknown**.

The key player in the bound $2 e^{-2\epsilon^2 n}$ is $n$.
- If $n \to \infty, \nu \approx \mu$ with very very high probabilty, but not for sure.

<!-- slide -->

# Learning a Target Function
In learning, the unknown object is an entire function $f$; in the bag it was a single number $\mu$.

# ![Targfunc](/assets/targfunc.png)


<!-- slide -->

# Learning a Target Function
# ![Targfunc2](/assets/targfunc2.png)

<!-- slide -->

# Learning a Target Function
# ![Targfunc3](/assets/targfunc3.png)

<!-- slide -->

# Learning a Target Function

1. White area in second figure: $h(x) = f(x)$

2. Green area in second figure: $h(x) \neq f(x)$

Define the following notion:
\[
\mathbb{E}(h) = \mathbb{P}_x \left [ h(x) \neq f(x) \right ]
\]
That is, this is the "size" of the green region.

- $\clubsuit$ "marble": $h(x) = f(x)$
- $\diamondsuit$ "marble": $h(x) \neq f(x)$.

We can re-frame $\mu, \nu$ in terms of $\mathbb{E}(h) = \mathbb{P}_x \left [ h(x) \neq f(x) \right ]$


<!-- slide -->
# Closing the Metaphor

**Out-of-sample error:** $\mathbb{E}_\text{out}(h) = \mathbb{P}_x \left [ h(x) \neq f(x) \right ]$ %\pause


**In-sample error:** $\mathbb{E}_\text{in}(h) = \frac{1}{n} \sum_{i=1}^n \mathbb{I} \left [ h(x) \neq f(x) \right ]$

### Hoeffding's Inequality, restated:
\[
\mathbb{P} \left [ |\mathbb{E}_\text{in}(h)  - \mathbb{E}_\text{out}(h)  |  > \epsilon \right ] \leq  2 e^{-2\epsilon^2 n}
\]

Victory! If we just minimize in-sample error, we are likely to be right out of sample!

...right?

<!-- slide -->
# Verification vs Learning
The entire previous argument assumed a FIXED $h$ and then came the data.

Given $h \in \mathcal{H}$, a sample can verify whether or not it is good (w.r.t. $f$):
- if $\mathbb{E}_\text{in}$ is small, $h$ is good, with high confidence.
- if $\mathbb{E}_\text{in}$ is large, $h$ is bad with high confidence.

In this (artificial example) world: we have no control over $\mathbb{E}_\text{in}$.

In learning, you actually try to *fit* the data!
- e.g., perceptron model $g$ results from searching an entire hypothesis set $\mathcal{H}$ for a hypothesis with small $\mathbb{E}_\text{in}$.
