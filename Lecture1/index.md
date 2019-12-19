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

Me: My primary area of expertise is psychology and economics.

This class is totally, unapologetically a work in progress.

Material is a mish-mash of stuff from:


- Caltech (undergrad)

- Stanford University (graduate course)

- Harvard (graduate course)

...so, yeah, it will be challenging. Hopefully, you'll find it fun!

My research: Occasionally touches the topics in the course; mostly utilizes things in the course as tools.

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
  - Attendance and Participation: 60\%
  - Exams and Labs: 20\%
  - Writing Assignment: 20\%

- Although exams are given a relatively low weight, you must attempt both exams to pass the course.
- Labs consist of a practical implementation of something we've covered in the course (e.g., code your own Recommender System).

<!-- slide -->

# Introduction: This Course

Grading: **come to class.**

If you are the type of student that doesn't generally enjoy coming to class, this is not the course for you.

I suspect the exams will be much like my exams in my other course. Students have described those exams as ``painfully difficult". You are only *entitled* to the rubric on the previous slide. However, if you complete all assignments, I will utilize the following rough guidelines for grading:

4.0 Came to class regularly and contributed substantive comments to discussions, did modestly well on exams, turned in all assignments.

3.5 Came to class regularly and said some stuff (sometimes interesting), did modestly poorly on exams, turned in all assignments.

3.0 Came to class regularly and said some stuff (mostly uninteresting), did very poorly on exams, turned in all assignments.

< 3.0 Didn't come to class regularly or didn't turn in all assignments.

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


\begin{itemize}
	\item Some of the lectures will be way too long or too short.

	\item Some (most?) of the lectures won't make sense.

	\item Some of the time I'll forget what I intended to say and awkwardly stare at you for a few moments (sorry).

\end{itemize}


Comment \underline{throughout} the course, not just at the end.


The material will improve with time and feedback.


I encourage measured feedback and thoughtful responses to questions. If I call on you and you don't know immediately, don't freak out. If you don't know, it's totally okay to say you don't know.


\end{frame}

\<!-- slide -->

#Introduction: This Course

SUPER BIG IMPORTANT EXPLANATION OF THE COURSE:


I teach using ``math''.


...Don't be afraid. The math won't hurt you.


I fundamentally believe that true knowledge of how we learn from data depends on a basic understanding of the underlying mathematics.

	\item Good news: no black boxes. \\You'll \underline{actually learn} stuff. (Probably. Hopefully?)
	\item Also good news: level of required math is reasonably low. High-school algebra or equivalent should be fine.
	\item Bad news: notation-heavy slides and reading.

<!-- slide -->

# Introduction: This Course

Finally: I cannot address field-specific questions in areas outside economics to any satisfying degree.

Good news: I'm good at knowing what I don't know and have a very small ego, which means that I'm much less likely to blow smoke up your ass than other professors.

Bad news: I can't help with certain types of questions.

This course should be applicable broadly, but many of the examples will lean on my personal expertise (sorry).

<!-- slide -->

# Last Intro Slide

Your "assignment": read syllabus.

Things to stress from syllabus:

		\item E-mail is the best way to contact me.
		\item No appointments necessary for regularly scheduled office hours; or by appointment.
		\item Can only reschedule exams (with good reason) if you tell me \textbf{before} the exam that you have a conflict.
		\item Notify me immediately if you need accommodations because of RCPD or religious convictions; If you approach me at the last minute, I may not be able to help.

	Despite my hard-assness in these intro slides: I'm here to help and I am not in the business of giving bad grades for no reason.



<!-- slide -->

#What is \textquotedblleft Data Analytics\textquotedblright ?}

How do you define \textquotedblleft data analytics\textquotedblright ?  (Not a rhetorical question!)

\begin{itemize}
	\item This course will avoid this nomenclature. It is confusing and imprecise. But you signed up (suckers) and I owe an explanation of what this course will cover.
\end{itemize}

\bigskip

Some \textquotedblleft data analytics \textquotedblright topics we will cover:

\begin{itemize}
	\item Linear regression: \textit{il classico.}

	\item Models of classification or discrete choice.

	\item Analysis of ``wide'' data.

	\item Decision trees and other non-linear models.

	\item Neural networks and other things that have deceptively cool names (not as fun as they sound).
\end{itemize}

\bigskip

Starting point for this course:

\begin{itemize}
	\item Better utilizing existing data can improve our \textbf{predictive power} whilst providing interpretable outputs for making policies.
\end{itemize}


\end{frame}

\<!-- slide -->
#Outline of the Course}

\begin{itemize}
\item[I.] Theoretical Underpinnings of Statistical Learning

\begin{itemize}
\item[A.]  Setup and a ``Case Study''
\item[B.]  The Learning Problem
\item[C.]  Linear Regression
\item[D.]  Bias versus Variance
\item[E.]  Training versus Testing
\item[F.]  The VC Dimension
\item[G.]  	Bias versus Variance
\end{itemize}

\item[II.] Parametric Models in Statistical Learning


\item[IIa.] Models of Classification


\item[IIb.] Linear Model Selection


\item[III.] Non-Parametric Models in Statistical Learning


\item[IIIa.] Tree-Based Methods


\item[IIIb]. Neural Networks


\item[IV] Unsupervised Learning



\end{itemize}

\end{frame}


\<!-- slide -->
#\textquotedblleft Non-Social Science\textquotedblright\ Approaches to Statistical Learning: A Brief History}

Suppose you are a researcher and you want to teach a computer to recognize images of a tree.

\vspace*{0.2in}

Note: this is an ``easy" problem. If you show pictures to a 3-year-old, that child will probably be able to tell you if there is a tree in the picture.

\vspace*{0.2in}

Computer scientists spent about 20 years on this problem because they thought about the problem like nerds and tried to write down a series of rules.  \\\medskip~\\ Rules are difficult to form, and simply writing rules misses the key insight: the data can tell you something.


\end{frame}



\<!-- slide -->

#\textquotedblleft Social Science\textquotedblright\ Approaches to Data: A Brief History}

Suppose you are a researcher and you want to know whether prisons reduce crime.

\vspace*{0.1in}

from ``A Call for a Moratorium on Prison Building'' (1976)

\begin{itemize}
	\item Between 1955 and 1975, fifteen states increased the collective capacity of their adult prison systems by 56\% (from, on average, 63,100 to 98,649).
	\item Fifteen other states increased capacity by less than 4\% (from 49,575 to 51,440).
	\item In ``heavy-construction" states the crime rate increased by 167\%; in ``low-construction'' states the crime rate increased by 145\%.
\end{itemize}



\begin{table}\centering
	\ra{1.3}
	\begin{tabular}{@{}ccc@{}}\toprule
		& Prison Capacity& Crime Rate \\ \midrule
		High construction & $\uparrow$~56\%& $\uparrow$~167\%\\
		Low construction & $\uparrow$~4\%& $\uparrow$~145\%\\ \bottomrule
	\end{tabular}
\end{table}


\end{frame}


\<!-- slide -->
#The Pros and Cons of Correlation}

\newtheorem{pros}{Pros}
\begin{pros}
	Nature gives you correlations for free.  \\
	In principle, everyone can agree on the facts.
\end{pros}

\vspace*{0.3in}

\newtheorem{cons}{Cons}
\begin{cons}
	Correlations are not very helpful.  \\
	They show what has happened, but not why. \\
	For many things, we care about why.
\end{cons}

\end{frame}


\<!-- slide -->
#Why a Correlation Exists Between X and Y}

\begin{enumerate}
	\item $X \rightarrow Y$
	\item[] X causes Y (causality)
	\medskip
	\item $X \leftarrow Y$
	\item[] Y causes X (reverse causality)
	\medskip
	\item $Z \rightarrow X$; $Z \rightarrow Y$
	\item[] Z causes X and Y (common cause)
	\medskip
	\item $X \rightarrow Y$; $Y \rightarrow X$
	\item[] X causes Y and Y causes X (simultaneous equations)

\end{enumerate}
\end{frame}


\<!-- slide -->
#Uniting Social Science and Computer Science}

We will start in this course by examining situations where we do \textbf{not} care about why something has happened, but instead we care about our ability to predict its occurrence from existing data.

\vspace*{0.2in}

(But of course keep in back of mind that if you are making policy, you must care about why something happened).

\vspace*{0.2in}

We will also borrow a few other ideas from CS:
\begin{itemize}
	\item ``Anything is data"
	\begin{enumerate}
		\item Satellite data
		\item Unstructured text or audio
		\item Facial expressions or vocal intonations
	\end{enumerate}
\item Subtle improvements on existing techniques
\item An eye towards practical implementability over ``cleanliness"
\end{itemize}

\end{frame}


\<!-- slide -->
#A Case Study in Prediction}

\textbf{Example:} a firm wishes to predict user behavior based on previous purchases or interactions.

\begin{figure}
	\centering
	\includegraphics[width=0.7\linewidth]{netflix}
\end{figure}

\end{frame}

\<!-- slide -->
#A Case Study in Prediction}

Small margins $\rightarrow$ huge payoffs.  $10\% \rightarrow$ \$1 million.

\vspace*{0.2in}

Not obvious why this was true for Netflix; quite obvious why this is true in financial markets.

\end{frame}

\<!-- slide -->
#Less Tired Examples of Prediction}

\begin{itemize}
	\item Identify the risk factors for prostate cancer.
	\item Classify a tissue sample into one of several cancer classes, based on a gene expression profile.
	\item Classify a recorded phoneme based on a log-periodogram.
	\item Predict whether someone will have a heart attack on the basis of demographic, diet and clinical measurements.
	\item Customize an email spam detection system.
	\item Identify a hand-drawn object.
	\item Determine which oscillations of stellar luminosity are likely due to exoplanets.
	\item Establish the relationship between salary and demographic variables in population survey data.
\end{itemize}

\end{frame}

\<!-- slide -->
#An Aside: Nomenclature}

\textbf{Machine learning} arose as a subfield of Artificial Intelligence.

\vspace*{0.2in}

\textbf{Statistical learning} arose as a subfield of Statistics.

\vspace*{0.2in}

There is much overlap; however, a few points of distinction:
\begin{itemize}
\item Machine learning has a greater emphasis on large scale applications and prediction accuracy.
\item Statistical learning emphasizes models and their interpretability, and precision and uncertainty.
\item But the distinction has become more and more blurred, and there is a great deal of ``cross-fertilization''.
\end{itemize}

\textbf{Obviously true:} machine learning has the upper hand in marketing.
\end{frame}


\<!-- slide -->
#Learning from Data}

The following are the basic requirements for statistical learning:

\begin{enumerate}
	\item A pattern exists.
	\medskip
	\item This pattern is not easily expressed in a closed mathematical form.
	\medskip
	\item You have data.
\end{enumerate}

\end{frame}


\<!-- slide -->
#Social Science Example}

\begin{figure}
	\centering
	\includegraphics[width=0.7\linewidth]{gdp}
\end{figure}


\end{frame}


\<!-- slide -->
#Social Science Example}

\begin{figure}
	\centering
	\includegraphics[width=0.7\linewidth]{us_gdp}
\end{figure}

\end{frame}


\<!-- slide -->
#Formalization}

Here \alert{\texttt{emissions}} is a \textit{response} or \textit{target} that we wish to predict.

\vspace*{0.1in}

We generically refer to the response as $Y$.

\vspace*{0.1in}

\alert{\texttt{GDP}}  is a \textit{feature}, or \textit{input}, or \textit{predictor}, or \textit{regressor}; call it $X_1$.

\vspace*{0.1in}

Likewise let's test our postulate and call \alert{\texttt{westernhem}} as $X_2$, and so on.

\vspace*{0.1in}

We can refer to the input vector collectively as
\begin{align*}
X &= \begin{pmatrix}
X_{1} \\
X_{2} \\
\vdots \\
X_{m}
\end{pmatrix}
\end{align*}

We are seeking
\[
Y = f (X) + \epsilon
\]
\end{frame}

\<!-- slide -->
#Formalization}

We call the function $f : \mathcal{X} \to \mathcal{Y}$ the \textit{target function}.

\vspace*{0.2in}

The target function is \alert{always unknown}. It is the object of learning.

\vspace*{0.2in}

Methodology:
\begin{itemize}
	\item Observe data $(x_1, y_1) \dots (x_N, y_N)$.
	\item Use some algorithm to approximate $f$.
	\item Produce final hypothesis function $g \approx f$.
\end{itemize}
\end{frame}


\<!-- slide -->
#The Learning Problem}

The coming lectures will express when the learning problem is feasible, and describe the most common solution to the learning problem (the linear regression).
