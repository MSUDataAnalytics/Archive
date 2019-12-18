# Social Science Data Analytics Applications

## What is This Course and Can / Should You Take It?

Innovations in statistical learning have created many engineering breakthroughs. From real time voice recognition to automatic categorization (and in some cases production) of news stories, machine learning is transforming the way we live our lives.  These techniques are, at their heart, novel ways to work with data, and therefore they should have implications for social science. This course explores the intersection of statistical learning (or machine learning) and social science and aims to answer two primary questions about these new techniques:

(i)	How does statistical learning work and what kinds of statistical guarantees can be made about the performance of statistical-learning algorithms?

(ii)	How can statistical learning be used to answer questions that interest social science researchers, such as testing theories or improving social policy?

In order to address these questions, we will cover so-called “standard” techniques such as supervised and unsupervised learning, statistical learning theory and nonparametric and Bayesian approaches. Although this class will cover these novel methodologies in some detail, it is not a substitute for the appropriate class in Computer Science. Nor is this a class that teaches specific skills for the job market. Rather, this class will teach you to think about data analytics broadly. We will spend a great deal of time learning how to interpret the output of statistical learning algorithms and approaches, and will also spend a great deal of time on better understanding the basic ideas in statistical learning. This, of course, comes at some cost in terms of time spent on learning computational and/or programming skills.

**Enrollment for credit in this course is simply not suitable for those unprepared in or uninterested in elementary statistical theory no matter the intensity of interest in machine learning or “Big Data”.  Really.**

You will be required to understand elementary mathematics in this course and should have at least some exposure to statistical theory. The class is front-loaded technically: early lectures are more mathematically oriented, while later lectures are more applied.

The topics covered in this course are listed later in this document. I will assign readings sparingly from Introduction to Statistical Learning. This text is available for free online. Additional readings are listed on the syllabus, and I encourage interested students to pursue them. I will provide lecture notes online for you to review and prepare for coming lectures. Importantly, the lectures deviate a fair bit from the reading, and thus you will rely on your course notes much more than you might in other classes.

If—-after you have read this document and preferably after attending the first lecture—-you have any questions about whether this course is appropriate for you, please come talk to me.  Anybody is permitted to attend the lectures and I am delighted if people can benefit.

## What This Course is Not

The focus of this course is conceptual. The goal is to create a working understanding of when and how tools from computer science and statistics can be profitably applied to problems in social science. Though students will be required to apply some of these techniques themselves, this course is not…

*…a replacement for EC420 or a course in causal inference.*

As social scientists, we are most often concerned with causal inference in order to write policy. Statistical learning and the other methods we will discuss in this course are generally not well-suited to these problems and while I’ll give a short overview of standard methods, this is only to build intuitions. Ultimately, this course has a different focus and you should still pursue standard methodological insights from your home departments.

*…a course on the computational aspects of the underlying methods.*

There are some important innovations that have made machine learning techniques computationally feasible. We will not discuss these, as there are computer science courses better equipped to cover them. When appropriate, we will discuss whether something is computable, and give rough approximations of the amount of time required (e.g. P vs NP). But we will not discuss how optimizers work or best practices in programming.

*…a primer on the nitty-gritty of how to use these tools or a way to pad your resume.*

The mechanics of implementation, whether it be programming languages or learning to use APIs, will not be covered in any satisfying level of depth. Students will be expected to learn any programming skills on their own. This is not a good course for people simply looking to learn the mechanics of using programming. It is designed to get you to use both traditional analytics and, eventually, machine learning tools. We will do some of this, and you will have the opportunity to explore topics that interest you through a final project, but ultimately this is a course that largely focuses on the theoretical aspects of statistical learning as applied to social science and **not** a class on programming.

Perhaps most importantly, this course is an attempt to push toward the frontiers in social science. Thus, please allow some messiness. Some topics may be underdeveloped for a given person’s passions, but given the wide variety of technical skills and overall interests, this is a required sacrifice. Both the challenge and opportunity of this area comes from the fact that there is no fully developed unifying framework.

## Evaluations and Grades

Your grade in this course will be based on attendance, two exams, “labs”, and a final project.

The general breakdown will be 20% for exams, 60% for labs and the final project, and 20% for attendance and participation, disproportionately weighted towards your participation in the group final project. Details on evaluations for the final project will be distributed a couple of weeks into the course – evaluations (read: grades) are designed not to deter anyone from taking this course who might otherwise be interested.

Although exams are assigned a relatively low weight, you must complete both exams to pass the course. There will be no exceptions to this rule.

Labs will be short homework assignments that require you to do something practical using a basic statistical language. Support will be provided for the `R` language only. You must have access to computing resources and the ability to program basic statistical analyses.

Finally, this course will not teach you how to program or how to write code in a specific language. If you are unprepared to do implement basic statistical coding, please take (or retake) PLS202. I highly encourage seeking coding advice from those who instruct computer science courses – it’s their job and they are better at it than I am. I’ll try to provide a good service, but I'm really not an expert in instruction as it relates to programming.
