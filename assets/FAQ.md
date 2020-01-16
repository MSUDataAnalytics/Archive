# Frequently Asked Questions

Below is a (non-exhaustive) list of frequently asked questions in this course. Throughout the term (and over the coming years) we will add to this document to better guide the class and to--hopefully--avoid confusion. If your question is not addressed below and you believe it would be appropriate for this part of the website, please [email me](mailto:bbushong@msu.edu).

[Teams](#teams)

[Labs](#labs)

[Software](#software)

[Misc](#misc)

---

**Office Hours are Tuesday & Thurday, 4:30 - 5:45 PM in 25A Marshall Adams Hall.**[^0]

[^0]: Some caveats and additional explanations are in order. First: office hours are first-come, first-served. Moreover, they will end at the designated time. Finally, an additional note: office hours for EC404 are from 3:15 - 4:30 PM; accordingly, I will be in my office speaking to students between 3:15 and 5:45. If you absolutely cannot make the assigned time but can come earlier, I will do my best to accommodate you. However, students in EC404 have first priority during that time period.

---

## Teams

> My team sucks; how can I switch teams?

Life is full of small disappointments. While we would love to spend 12 weeks carefully optimizing groups, that would require a collosal amount of effort that would ultimately not yield anything fruitful. You're stuck.

> My team sucks; how can I punish them for their lack of effort.

On this front, we will be more supportive. While you have to put up with your team regardless of their quality, you can indicate that your team members are not carrying their fair share by issuing a **strike**. This processs works as follows:
1. A team member systematically fails to exert effort on collaborative projects (for example, by not showing up for meetings or not communicating, or by simply leeching off others without contributing.)
2. Your frustration reaches a boiling point. You decide this has to stop. You decide to issue a **strike**
3. You send an email with the following information:
    - `Subject line:` [SSC442] Strike against [Last name of Recipient]
    - `Body:` You do **not** need to provide detailed reasoning. However, you must discuss the actions (plural) you took to remedy the situation before sending the strike email.

A strike is a serious matter, and will reduce that team member's grade on joint work by 10%. If any team-member gets strikes from all other members of his or her team, their grade will be reduced by 50%.

Strikes are *anonymous* so that you do not need to fear social retaliation. However, they are not anonymous to allow you to issue them without thoughtful consideration. Perhaps the other person has a serious issue that is preventing them from completing work (e.g., a relative passing away). Please be thoughtful in using this remedy and consider it a last resort.

> Do I really need to create a team GitHub repository? I don't like GitHub / programming/ work.

Yes, you need to become familiar with GitHub and you and your team will work in a central repository for both labs and your final project.

This is for two reasons. First, computer scientists spent a huge amount of time coming up with the solutions that are implemented in GitHub (and other flavors of `git`). Their efforts are largely dedicated toward solving a very concrete goal: how can two people edit the same thing at the same time without creating a ton of new issues. While you could use a paid variant of GitHub (e.g., you could all collaborate over the Microsoft Office suite as implemented by the 360 software that MSU provides), you'd ultimately have the following issues:
1. The software doesn't support some file types.
2. The software doesn't autosave versions.[^1] If someone accidentally deletes something, you're in trouble.
3. You have to learn an entirely new system every time you change classes / universities / jobs, because said institute doesn't buy the product you love.[^2]

[^1]: Some products, of course, solve this problem a little bit. For example, Dropbox allows users to share files with ease (of any file type) and saves a (coarse) version history. However, Dropbox does not allow multiple users to work on the same file, and has no way of merging edits together.

[^2]: This logic is also why we utilize only free software in this course. It sucks to get really good at, say, `SAS` (as I did many years ago) only to realize that the software costs about $10000 and many firms are unwilling to spent that. We will try our best to avoid giving you dead-end skills.

> I'm on a smaller-than-normal team. Does this mean that I have to do more work?

Your instructors are able to count and are aware the teams are imbalanced. Evaluations of final projects will take this into account. Your work on labs will be assessed solely according to your responses and (where applicable) the correctness of your answers. That said, labs are designed to be reasonably straightforward. While your final product should reflect the best ability of your team, we do not anticipate that the uneven teams will lead to substantively different outputs.

> What does the group project entail?

1. You must find existing data to analyze.[^3] Aggregating data from multiple sources is encouraged, but is not required.

[^3]: Note that **existing** is taken to mean that you are not permitted to collect data by interacting with other people. That is not to say that you cannot gather data that previously has not been gathered into a single place---this sort of exercise is encouraged.

2. You must visualize (at least) three **interesting** features of that data. Visualizations should aid the reader in understanding something about the data that might not be readily aparent.[^4]

[^4]: Pie charts of any kind will result in a 25\% grade deduction.

3. You must come up with some analysis---using tools from the course---which relates your data to either a prediction or a policy conclusion. For example, if you collected data from Major League Baseball games, you could try to "predict" whether a left-hander was pitching based solely on the outcomes of the batsmen.[^5]

[^5]: This is an extremely dumb idea for a number of reasons. Moreover, it's worth mentioning that sports data, while rich, can be overwhelming due to its sheer magnitude and the variety of approaches that can be applied. Use with caution.

4. You must present your analysis as if presenting to a **C-suite executive**. If you are not familiar with this terminology, the C-suite includes, e.g., the CEO, CFO, and COO of a given company. Generally speaking, such executives are not particularly analytically oriented, and therefore your explanations need to be clear, consise (their time is valuable) and contain actionable (or valuable) information.[^6]
    - Concretely, this requires one of the two following options:
      1. At least one member of the group **presents from a slide deck** (preferably as a PDF; alternatively as either a `Powerpoint` or `Presentation` file). If your group chooses this option, presentations will be 10 minutes in duration (preferrably, within about 30 seconds on either side of that time). This is hard: you will need to be able to quickly and clearly present a few results.
      - Whether or not a given person presents (or does not present) will not affect their grade if this alternative is chosen. Accordingly, groups should divide their work to capitalize on the skills of the members.
      2. The group **writes a memo**---less than 5 pages---which describes their data, analyses, and results. This must be clear and easy to understand for a non-expert in your field.

[^6]: This exercise provides you with an opportunity to identify your marketable skills and to practice them. I encourage those who will be looking for jobs soon to take this exercise seriously.

---

## Labs

> How do I turn in my lab? I wasn't listening.

Your **team** turns in labs together. You will all recieve the same score (including "0" if you did not turn in the assignment). Your team must have a GitHub repository. This must be open. We will evaluate your project **during the assigned class time** by going to your GitHub and running the relevant code (or, when applicable, opening a document).

> How do we format our labs when we turn them in?

We (very much) prefer that your team turns in the written answers to your homework as a PDF. These should include any figures. Why PDFs? Well, as is a theme in this course, we like file formats that are free, that render on lots of machines (e.g., mobile) and that can compress easily for space saving. PDFs meet this requirement nicely.

To create a PDF, you have a few options. You could use `Word` or a similar program, and then export as a PDF. This is gross and if you do it you're officially not getting a job at Facebook. A better approach would be to use a simple and streamlined word processor like [`Atom`](https://atom.io) or [`Sublime Text`](https://www.sublimetext.com) and then figure out how to export a PDF from there.[^7]

The associated code should be saved within your GitHub page and identified clearly. For example, if you are using `R`, then your final code for Lab 1 should be saved as `Lab1.r`. Please don't make us hunt for your code.

[^7]: Both text editors (and many others) have easy solutions to that question, and it's a good way to learn a little more about a very important part of the production process. As the human in the production system, you should recognize that your value is in the generation of text characters by pressing mechanical keys with fingers. Computers can't (yet) do this. Accordingly, the screen you stare at while you complete this process is important, and you should get to know an editor like it is your best friend. But, you know, also make sure to go outside from time to time.

> How many labs are there?

Five? Six? Certainly more than four and less than seven.

> How long should our answers be?

As long as they need to be. Typically, the question will cue you as to what is required. If something is ambiguous or unclear, use your judgement.

---

## Software

> My setup for `R` isn't working. How can I fix it?

This is the most common question in the course and we have no satisfying answer. Computers have tons of specific details that inform how they execute a given piece of code. We cannot say what random program you installed years ago---and sadly this will, to a great extent, determine how your computer performs. Here's a general strategy to dealing with your problem.

1. Google the error message you get. Scroll around until you find a reasonable solution to your problem (often on pages like StackExchange or GitHub's discussion boards).
2. Annoyingly, the second-best way to fix `R` is often to reinstall `R`. This will fix all but the biggest of issues.
3. Still not working? *Not good, Bob!* You're going to have a longer session ahead of you where you deep-dive some opaque error code or have to wipe more than just your `R` installation.[^8]
4. Come to Office Hours.

[^8]: This brings us to another recurrent point: your computer is your most important tool. You absolutely must ensure that you keep it from getting f\&$@ed up and cluttered. If you're working with an older machine (more than 5 years old), consider this the moment that you realize you need to upgrade.

> Can I complete this class in `[insert some programming language here]`?

Sure. Sadly, we cannot provide instruction in multiple programming languages; it's simply not realistic. However,

---

## Misc

> How are grades assigned?

As in the syllabus---which outlines evaluations--- grades are assigned using the "standard" cutoffs. For example, 93% and up ensures that you earn a 4.0 in the class. Between 87% and 92% yields a 3.0; and so on down the scale. We will round grades incredibly in your favor---e.g., a 92.47% will be rounded to 92.5% and then rounded to 93%, yielding a 4.0.[^9]

[^9]: Don't do this in a math class. Seriously, don't.

> What material will be covered on exams?

Exams are conceptual. They will require you to understand the mathematics of the course, but will **not** require you to replicate that mathematics. Instead, you should expect to think carefully about a problem in social science and use your judgement to come up with a reasoned solution.
