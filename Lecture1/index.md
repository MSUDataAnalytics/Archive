---
presentation:
  width: 1920
  height: 1080
---

<!-- slide -->
# Introduction


> I keep saying that the sexy job in the next 10 years will be statisticians. And I'm not kidding.

### Hal Varian, Chief Economist, Google


<!-- slide -->

# Introduction: About Me

**Me:** My primary area of expertise is psychology and economics.

This class is totally, unapologetically a work in progress.

Material is a mish-mash of stuff from:


- Caltech (undergrad)

- Stanford University (graduate course)

- Harvard (graduate course)

...so, yeah, it will be challenging. Hopefully, you'll find it fun!

My research: occasionally touches the topics in the course, but mostly utilizes things in the course as tools.

<!-- slide -->
# Introduction: About You

*New phone who dis?*  Please write down your
- name
- major
- desired graduation year and semester
- interest in this course

You **must** spend 5 minutes telling me a little bit about your interests before the end of the week.

<!-- slide -->

# Introduction: This Course

The syllabus is posted on the course website

I'll walk through highlights now, but read it later -- it's long.
- But eventually, please read it. It is "required."

	Syllabus highlights:

- Grade is composed of problem sets, exams, and a written assignment.
  - Attendance and Participation: 30\%
  - Exams: 20\%
  - Labs and Final Project: 50\%

- Although exams are given a relatively low weight, you must attempt both exams to pass the course.
- Labs consist of a practical implementation of something we've covered in the course (e.g., code your own Recommender System).

<!-- slide -->

# Introduction: This Course

Grading: **come to class.**

If you are the type of student that doesn't generally enjoy coming to class, this is not the course for you.

I suspect the exams will be much like my exams in my other course. Students have described those exams as ``painfully difficult". You are only *entitled* to the rubric on the previous slide.

**If** you complete all assignments and attend all class dates, I will utilize the following curve for grading:

`4.0` Came to class regularly, contributed substantive comments to discussions, did modestly well on exams, turned in all assignments.

`3.5` Came to class regularly, said some stuff (sometimes interesting), did modestly poorly on exams, turned in all assignments.

`3.0` Came to class regularly and said some stuff (mostly uninteresting), did very poorly on exams, turned in all assignments.

`< 3.0` Didn't come to class regularly or didn't turn in all assignments.

<!-- slide -->

# Introduction: This Course

There are sort of three texts for this course and sort of zero.

The main text is free and available online (see syllabus or Google it). The secondary text is substantially more difficult, but also free online. The third text costs about \$25.

**Please please please please please:** Ask questions during class.
- Some of the material is quite hard.
- Sometimes (often?) the material itself will be confusing or interesting---or both!
- Think of your role as similar to a participant in a focus group.

Return of the Please: If there is some topic that you really want to learn about, ask. If you are uncomfortable asking in front of the whole group, please see me during office hours.

<!-- slide -->

# Introduction: This Course

Because this is a new course:

- Some of the lectures will be way too long or too short.

- Some (most?) of the lectures won't make sense.

- Some of the time I'll forget what I intended to say and awkwardly stare at you for a few moments (sorry).

Comment **throughout** the course, not just at the end.


The material will improve with time and feedback.


I encourage measured feedback and thoughtful responses to questions. If I call on you and you don't know immediately, don't freak out. If you don't know, it's totally okay to say you don't know.

<!-- slide -->

# Introduction: This Course

### SUPER BIG IMPORTANT EXPLANATION OF THE COURSE


I teach using ``math''.

...Don't be afraid. The math won't hurt you.

I fundamentally believe that true knowledge of how we learn from data depends on a basic understanding of the underlying mathematics.

-Good news: no black boxes.
  - You'll **actually learn** stuff. (Probably. Hopefully?)
- Also good news: level of required math is reasonably low. High-school algebra or equivalent should be fine.

-Bad news: notation-heavy slides and reading.

<!-- slide -->

# Introduction: This Course

Finally: I cannot address field-specific questions in areas outside economics to any satisfying degree.

Good news: I'm good at knowing what I don't know and have a very small ego, which means that I'm much less likely to blow smoke up your ass than other professors.

Bad news: I can't help with certain types of questions.

This course should be applicable broadly, but many of the examples will lean on my personal expertise (sorry).

<!-- slide -->

# Last Intro Slide

Your "assignment": read syllabus and Lab 0.

Things to stress from syllabus and Lab 0:

- E-mail isn't the ideal solution for technical problems
- No appointments necessary for regularly scheduled office hours; or by appointment.
- Can only reschedule exams (with good reason) if you tell me \textbf{before} the exam that you have a conflict.
  - Notify me immediately if you need accommodations because of RCPD or religious convictions; If you approach me at the last minute, I may not be able to help.

Despite my hard-assness in these intro slides: I'm here to help and I am not in the business of giving bad grades for no reason.

<!-- slide -->

# What is "Data Analytics"?

How do **you** define "data analytics"?  (Not a rhetorical question!)

- This course will avoid this nomenclature. It is confusing and imprecise. But you signed up (suckers) and I owe an explanation of what this course will cover.


Some "data analytics" topics we will cover:

- Linear regression: *il classico*.
- Models of classification or discrete choice.
- Analysis of ``wide'' data.
- Decision trees and other non-linear models.
- Neural networks and other things that have deceptively cool names (not as fun as they sound).

<!-- slide -->
## Starting point for this course

Better utilizing existing data can improve our predictive power whilst providing interpretable outputs for making policies.

<!-- slide -->

# Outline of the Course

`[I.]` Theoretical Underpinnings of Statistical Learning

    [A.] Setup and a "Case Study"
    [B.] The Learning Problem
    [C.] Linear Regression
    [D.] Bias versus Variance
    [E.] Training versus Testing
    [F.] The VC Dimension
    [G.] Bias versus Variance

`[II.]` Parametric Models in Statistical Learning

`[IIa.]` Models of Classification

`[IIb.]` Linear Model Selection

`[III.]` Non-Parametric Models in Statistical Learning

`[IIIa.]` Tree-Based Methods

`[IIIb.]` Neural Networks

`[IV]` Unsupervised Learning

<!-- slide -->

## Non-Social Science Approaches to Statistical Learning
### A Brief History

Suppose you are a researcher and you want to teach a computer to recognize images of a tree.

Note: this is an ``easy" problem. If you show pictures to a 3-year-old, that child will probably be able to tell you if there is a tree in the picture.

Computer scientists spent about 20 years on this problem because they thought about the problem like nerds and tried to write down a series of rules.

Rules are difficult to form, and simply writing rules misses the key insight: the data can tell you something.

<!-- slide -->

## Social Science Approaches to Statistical Learning
### A Brief History

Suppose you are a researcher and you want to know whether prisons reduce crime.


from ``A Call for a Moratorium on Prison Building'' (1976)

- Between 1955 and 1975, fifteen states increased the collective capacity of their adult prison systems by 56\% (from, on average, 63,100 to 98,649).
- Fifteen other states increased capacity by less than 4\% (from 49,575 to 51,440).
- In "heavy-construction" states the crime rate increased by 167\%; in "low-construction" states the crime rate increased by 145\%.

| | Prison Capacity | Crime Rate
| --- | --- | --- |
High construction | $\uparrow$~56\% | $\uparrow$~167\%
Low construction | $\uparrow$~4\% | $\uparrow$~145\%

<!-- slide -->
# The Pros and Cons of Correlation

Pros:
  - Nature gives you correlations for free.
  - In principle, everyone can agree on the facts.

Cons:
  - Correlations are not very helpful.
  - They show what has happened, but not why.
  - For many things, we care about why.

<!-- slide -->
## Why a Correlation Exists Between X and Y

1. $X \rightarrow Y$
  X causes Y (causality)

2. $X \leftarrow Y$
  Y causes X (reverse causality)

3. $Z \rightarrow X$; $Z \rightarrow Y$
  Z causes X and Y (common cause)

4. $X \rightarrow Y$; $Y \rightarrow X$
  X causes Y and Y causes X (simultaneous equations)

<!-- slide -->
## Uniting Social Science and Computer Science

We will start in this course by examining situations where we do **not** care about why something has happened, but instead we care about our ability to predict its occurrence from existing data.

(But of course keep in back of mind that if you are making policy, you must care about why something happened).

We will also borrow a few other ideas from CS:
- Anything is data
  + Satellite data
  + Unstructured text or audio
  + Facial expressions or vocal intonations
- Subtle improvements on existing techniques
- An eye towards practical implementability over ``cleanliness"

<!-- slide -->
# A Case Study in Prediction

**Example:** a firm wishes to predict user behavior based on previous purchases or interactions.

![Netflix](/assets/netflix.jpg)

<!-- slide -->
# A Case Study in Prediction

Small margins $\rightarrow$ huge payoffs.  $10\% \rightarrow$ \$1 million.

Not obvious why this was true for Netflix; quite obvious why this is true in financial markets.

### More Recent Examples of Prediction

- Identify the risk factors for prostate cancer.
- Classify a tissue sample into one of several cancer classes, based on a gene expression profile.
- Classify a recorded phoneme based on a log-periodogram.
- Predict whether someone will have a heart attack on the basis of demographic, diet and clinical measurements.
- Customize an email spam detection system.
- Identify a hand-drawn object.
- Determine which oscillations of stellar luminosity are likely due to exoplanets.
- Establish the relationship between salary and demographic variables in population survey data.

<!-- slide -->
# An Aside: Nomenclature

**Machine learning** arose as a subfield of Artificial Intelligence.

**Statistical learning** arose as a subfield of Statistics.

There is much overlap; however, a few points of distinction:

- Machine learning has a greater emphasis on large scale applications and prediction accuracy.

- Statistical learning emphasizes models and their interpretability, and precision and uncertainty.
  - But the distinction has become more and more blurred, and there is a great deal of "cross-fertilization".

**Obviously true:** machine learning has the upper hand in marketing.


<!-- slide -->
# Learning from Data

The following are the basic requirements for statistical learning:

1. A pattern exists.
2. This pattern is not easily expressed in a closed mathematical form.
3. You have data.

<!-- slide -->
# Social Science Example

# ![GDP](/SSC442/assets/gdp.jpg)

<!-- slide -->
#Social Science Example

# ![USGDP](/SSC442/assets/us_gdp.png)

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
#The Learning Problem

The coming lectures will express when the learning problem is feasible, and describe the most common solution to the learning problem (the linear regression).
