# Social Science Data Analytics Applications

[Link to Instructions for Group Project Writeup](assets/writeup.md)

[Shortcut to Course Outline and Links to Lectures](#tentative-schedule-of-material)

[FAQ](assets/FAQ.md)

## What is This Course and Can / Should You Take It?

Innovations in statistical learning have created many engineering breakthroughs. From real time voice recognition to automatic categorization (and in some cases production) of news stories, machine learning is transforming the way we live our lives.  These techniques are, at their heart, novel ways to work with data, and therefore they should have implications for social science. This course explores the intersection of statistical learning (or machine learning) and social science and aims to answer two primary questions about these new techniques:

(i)	How does statistical learning work and what kinds of statistical guarantees can be made about the performance of statistical-learning algorithms?

(ii)	How can statistical learning be used to answer questions that interest social science researchers, such as testing theories or improving social policy?

In order to address these questions, we will cover so-called "standard" techniques such as supervised and unsupervised learning, statistical learning theory and nonparametric and Bayesian approaches. If it were up to me, this course would be titled "Statistical Learning for Social Scientists"—I believe this provides a more appropriate guide to the content of this course. And while this class will cover these novel statistical methodologies in some detail, it is not a substitute for the appropriate class in Computer Science or Statistics. Nor is this a class that teaches specific skills for the job market. Rather, this class will teach you to think about data analytics broadly. We will spend a great deal of time learning how to interpret the output of statistical learning algorithms and approaches, and will also spend a great deal of time on better understanding the basic ideas in statistical learning. This, of course, comes at some cost in terms of time spent on learning computational and/or programming skills.

**Enrollment for credit in this course is simply not suitable for those unprepared in or uninterested in elementary statistical theory no matter the intensity of interest in machine learning or “Big Data”.  Really.**

You will be required to understand elementary mathematics in this course and should have at least some exposure to statistical theory. The class is front-loaded technically: early lectures are more mathematically oriented, while later lectures are more applied.

The topics covered in this course are listed later in this document. I will assign readings sparingly from [Introduction to Statistical Learning](http://faculty.marshall.usc.edu/gareth-james/ISL/ISLR%20Seventh%20Printing.pdf), henceforth referred to as ISL. This text is available for free online and, for those who like physical books, can be purchased for about $25. Importantly, the lectures deviate a fair bit from the reading, and thus you will rely on your course notes much more than you might in other classes.

If---after you have read this document and preferably after attending the first lecture---you have any questions about whether this course is appropriate for you, please come talk to me.  Anybody is permitted to attend the lectures and I am delighted if people can benefit.

## What This Course is Not

The focus of this course is conceptual. The goal is to create a working understanding of when and how tools from computer science and statistics can be profitably applied to problems in social science. Though students will be required to apply some of these techniques themselves, this course is not…

*…a replacement for EC420 or a course in causal inference.*

As social scientists, we are most often concerned with causal inference in order to analyze and write policies. Statistical learning and the other methods we will discuss in this course are generally not well-suited to these problems, and while I’ll give a short overview of standard methods, this is only to build intuitions. Ultimately, this course has a different focus and you should still pursue standard methodological insights from your home departments.

*…a course on the computational aspects of the underlying methods.*

There are many important innovations that have made machine learning techniques computationally feasible. We will not discuss these, as there are computer science courses better equipped to cover them. When appropriate, we will discuss whether something **is** computable, and we will even give rough approximations of the amount of time required (e.g. **P** vs **NP**). But we will not discuss how optimizers work or best practices in programming.

*…a primer on the nitty-gritty of how to use these tools or a way to pad your resume.*

The mechanics of implementation, whether it be programming languages or learning to use APIs, will not be covered in any satisfying level of depth. Students will be expected to learn most of the programming skills on their own. Specifically, while there will be some material to remind you of basic `R` commands, this is not a good course for people who are simply looking to learn the mechanics of programming. This course is designed to get you to use both traditional analytics and, eventually, machine learning tools. We will do some review of basic programming, and you will have the opportunity to explore topics that interest you through a final project, but ultimately this is a course that largely focuses on the theoretical aspects of statistical learning as applied to social science and **not** a class on programming.

Perhaps most importantly, this course is an attempt to push toward the frontiers in social science. Thus, please allow some messiness. Some topics may be underdeveloped for a given person’s passions, but given the wide variety of technical skills and overall interests, this is a required sacrifice. Both the challenge and opportunity of this area comes from the fact that there is no fully developed, wholly unifying framework.

## Evaluations and Grades

Your grade in this course will be based on "attendance", two exams, labs, and a final project.

The general breakdown will be 20% for exams, 50% for labs and the final project, and 30% for attendance and participation, disproportionately weighted towards your participation in the group final project. Assignment of numeric grades will follow the standard, where ties (e.g., 92%) favor the student. Details on evaluations for the final project will be distributed a couple of weeks into the course – evaluations (read: grades) are designed not to deter anyone from taking this course who might otherwise be interested.

Although exams are assigned a relatively low weight, you must complete both exams to pass the course. There will be no exceptions to this rule.

Labs will be short homework assignments that require you to do something practical using a basic statistical language. Support will be provided for the `R` language only, although I may present some examples in `Python` from time to time. You must have access to computing resources and the ability to program basic statistical analyses.

As mentioned above, this course will not teach you how to program or how to write code in a specific language. If you are unprepared to do implement basic statistical coding, please take (or retake) PLS202. I highly encourage seeking coding advice from those who instruct computer science courses – it’s their job and they are better at it than I am. I’ll try to provide a good service, but I'm really not an expert in instruction as it relates to programming.

## Miscellanea

All class notes will be posted on https://msudataanalytics.github.io/SSC442.

#### Office Hours are Tues & Thur, 4:30 - 5:45 PM in 25A Marshall Adams Hall

Please use my office hours.  It would be remarkable if you didn’t need some assistance with the material, and I am here to help.  One of the benefits of open office hours is to accommodate many students at once; if fellow students are in my office, please join in and feel very free to show up in groups. Office hours will move around a little bit throughout the semester to attempt to meet the needs of all students.

In addition to drop-in office hours, I always have sign-up office hours for advising and other purposes.  They are online, linked from my web page. As a general rule, please first seek course-related help from the drop-in office hours. However, if my scheduled office hours are always infeasible for you, let me know, and then I may encourage you to make appointments with me. I ask that you schedule your studying so that you are prepared to ask questions during office hours – office hours are not a lecture and if you’re not prepared with questions we will end up awkwardly staring at each other for an hour until you leave.

Some gentle requests regarding office hours and on contacting me. First, my office hours end sharply at the end, so don’t arrive 10 minutes before the scheduled end and expect a full session. Please arrive early if you have lengthy questions, or if you don’t want to risk not having time due to others’ questions. You are free to ask me some stuff by e-mail, (e.g. a typo or something on a handout), but please know e-mail sucks for answering many types of questions. “How do I do this lab?” or “What’s up with `Python`?” are short questions with long answers. Come to office hours.

## Tentative Schedule of Material

What follows is a very rough schedule of what we will cover this semester — of course, I reserve the right to deviate from this schedule.  The number in brackets is the predicted number of lectures that we will spend on that topic, which will certainly change throughout the course. As each topic is filled, a link will appear to provide access to the content for that topic.

---
### Theoretical Underpinnings of Statistical Learning

[**Lab 0:** Personal Computer Setup](Labs/Lab0.md)

  1.  Setup and a “Case Study” [`[ 1 ]`](https://msudataanalytics.github.io/SSC442/Lecture1/index.html)
  2.  The Learning Problem [`[ 1 ]`](https://msudataanalytics.github.io/SSC442/Lecture2/index.html)[`[ 2 ]`](https://msudataanalytics.github.io/SSC442/Lecture3/index.html)

[**Lab 1:** Intro to Visualization in R](Labs/Lab1.html)

  3.  Linear Regression and Prediction	[`[ P1 ]`](https://msudataanalytics.github.io/SSC442/Lecture4/index.html)[`[ P2 ]`](https://msudataanalytics.github.io/SSC442/Lecture5/index.html) [`[ P3 ]`](https://msudataanalytics.github.io/SSC442/Lecture6/index.html)[`[ P4 ]`](https://msudataanalytics.github.io/SSC442/Lecture7/index.html)

**Reading:** ISL Chapter 3

[**Lab 2:** Linear Regression and Simple Analyses](Labs/Lab2.html)

[**Lab 3:** Training and Testing: Linear Predictions](Labs/Lab3.html)

  4.  Bias vs. Variance & Training vs. Testing	[`[ 1 ]`](https://msudataanalytics.github.io/SSC442/Lecture8/index.html)
  5.  Illustration of Central Concepts with Nearest Neighbors	[`[ P1 ]`](https://msudataanalytics.github.io/SSC442/Lecture9/index.html)
  6.  Illustration of Central Concepts with Decision Trees [`[ P1 ]`](https://msudataanalytics.github.io/SSC442/Lecture10/index.html)

```
Exam 1: Thursday, 20 February; due Tuesday, 25 February
```

---

### Parametric Models in Statistical Learning

#### Models of Classification
1. Intro to Classification [`[ P1 ]`](https://msudataanalytics.github.io/SSC442/Lecture11/index.html)

[**Class Activity**: Classification](/assets/classification.md)

2. Logistic Regression 	`[ 1 ]`
3. Linear Discriminant Analysis	`[ 1 ]`


#### Linear Model Selection
1. Shrinkage Models 	`[ 1 ]`
2. Data with High-Dimensionality  	`[ 1 ]`
3. Beyond Linearity	`[ 1 ]`
4. Support Vector Machines	`[ 2 ]`

```
Exam 2: Date TBD
```
---

### Applications and Extensions

#### Tree-Based Methods
1. Basics of Decision Trees	`[ 2 ]`
2. Forests and Random Forests  	`[ 1 ]`

#### Neural Networks
1. Basics of Neural Networks and Neural Computation 	`[ 2 ]`
2. Backpropagation and Cost Functions  	`[ 1 ]`
3. “Deep Learning”	`[ 1 ]`

#### Unsupervised Learning
1. Clustering	`[ 1 ]`
2. Miscellany `[ TBD ]`

---

The **Final Project** is due on the last day of the course (aka on the so-called final exam day). There is no final exam.

[Archived version of Syllabus](assets/syllabus.md)
