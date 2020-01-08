
<!-- slide -->
\frametitle{Extending the Example}

\textbf{Critical requirement:} samples must be independent. %\pause
\begin{itemize}
	\item If the sample is constructed in some arbitrary fashion, then indeed we cannot say anything. %\pause
\end{itemize}

\vspace*{0.2in}

Even with independence, $\nu$ can take on arbitrary values. %\pause
\begin{itemize}
	\item Some values are way more likely than others. This is what allows us to learn something – it is likely that $\nu \approx \mu$. %\pause
\end{itemize}

\vspace*{0.2in}

The bound $2 e^{-2\epsilon^2 n}$ does not depend on $\mu$ or the size of the bag.  %\pause
\begin{itemize}
	\item The bag can be infinite. %\pause
	\item It’s great that it does not depend on $\mu$ because $\mu$ is \textbf{unknown}. %\pause
\end{itemize}

\vspace*{0.2in}

The key player in the bound $2 e^{-2\epsilon^2 n}$ is $n$.  %\pause
\begin{itemize}
	\item If $n \to \infty, \nu \approx \mu$ with very very high probabilty, but not for sure.
\end{itemize}

\end{frame}

<!-- slide -->
\frametitle{Learning a Target Function}
In learning, the unknown is an entire function $f$; in the bag it was a single number $\mu$.

\vspace*{0.2in}

\begin{figure}
	\centering
	\includegraphics[width=0.5\linewidth]{graphics/targfunc}
\end{figure}



\end{frame}

<!-- slide -->
\frametitle{Learning a Target Function}
\begin{figure}
	\centering
	\visible<1->{\includegraphics[width=0.4\linewidth]{graphics/targfunc2}} \\
	\visible<2->{\includegraphics[width=0.4\linewidth]{graphics/targfunc3}}
\end{figure}

\visible<3>{White area in second figure: $h(x) = f(x)$

\vspace*{0.1in}

Green area in second figure: $h(x) \neq f(x)$}

\end{frame}


<!-- slide -->
\frametitle{Learning a Target Function}

Define the following notion:
\[
\E (h) = \mathbb{P}_x \left [ h(x) \neq f(x) \right ] %\pause
\]

That is, this is the ``size'' of the green region. %\pause

\vspace*{0.2in}

\begin{itemize}
	\item White ``marble'': $h(x) = f(x)$
	\item Green ``marble'': $h(x) \neq f(x)$. %\pause
\end{itemize}

\vspace*{0.1in}
We can re-frame $\mu, \nu$ in terms of $\E(h) = \mathbb{P}_x \left [ h(x) \neq f(x) \right ]$


\end{frame}

<!-- slide -->
\frametitle{Closing the Metaphor}

\textbf{Out-of-sample error:} $\E_\text{out}(h) = \mathbb{P}_x \left [ h(x) \neq f(x) \right ]$ %\pause

\vspace*{0.2in}

\textbf{In-sample error:} $\E_\text{in}(h) = \frac{1}{n} \sum_{i=1}^n \mathbb{I} \left [ h(x) \neq f(x) \right ]$

\vspace*{0.2in}
Hoeffding's Inequality, restated:
\[
\mathbb{P} \left [ |\E_\text{in}(h)  - \E_\text{out}(h)  |  > \epsilon \right ] \leq  2 e^{-2\epsilon^2 n} %\pause
\]

Victory! If we just minimize in-sample error, we are likely to be right out of sample!

\vspace*{0.1in}

\qquad \qquad {\tiny ...right?}

\end{frame}

<!-- slide -->
\frametitle{Verification vs Learning}
The entire previous argument assumed a FIXED $h$ and then came the data. %\pause

\vspace*{0.2in}
Given $h \in \mathcal{H}$, a sample can verify whether or not it is good (w.r.t. $f$):
\begin{itemize}
	\item if $\E_\text{in}$ is small, $h$ is good, with high confidence.
	\item if $\E_\text{in}$ is large, $h$ is bad with high confidence. %\pause
\end{itemize}

\vspace*{0.2in}
In this world: we have no control over $\E_\text{in}$. %\pause

\vspace*{0.2in}

In learning, you actually try to fit the data!
\begin{itemize}
	\item e.g., perceptron model $g$ results from searching an entire hypothesis set $\mathcal{H}$ for a hypothesis with small $\E_\text{in}$.
\end{itemize}
