# Lab 1: Basic Data Visualizations
Our primary tool for data visualization in the course will be ``ggplot2``. ``ggplot2`` implements the grammar of graphics, a coherent and relatively straightforward system for describing and building graphs. With ``ggplot2``, you can do more faster by learning one system and applying it in many places. Other languages provide more specific tools, but require you to learn a different tool for each application.

## Using ggplot2
In order to get our hands dirty, we will first have to load ``ggplot2``. To do this, and to access the datasets, help pages, and functions that we will use in this chapter, we will load the so-called tidyverse by running this code:

```{r, load, warning=FALSE}
library(tidyverse)
```
If you run this code and get an error message “there is no package called ‘tidyverse’”, you’ll need to first install it, then run library() once again. To install packages in ``R``, we utilize the simple function install.packages(). In this case, we would write:
```{r, load, warning=FALSE}
install.packages("tidyverse")
library(tidyverse)
```

Once we're up and running, we're ready to dive into some basic exercises. ``ggplot2`` works by specifying the connections between the variables in the data and the colors, points, and shapes you see on the screen. These logical connections are called *aesthetic mappings* or simply *aesthetics*.

How to use ``ggplot2`` -- the fast and wholly unclear recipe:

- `data = mpg`: Define what your data is. Here, we'll use the mpg data frame found in ggplot2 (by using `ggplot2::mpg`). As a reminder, a data frame is a rectangular collection of variables (in the columns) and observations (in the rows). This structure of data is often called a "table" but we'll try to use terms slightly more precisely. The `mpg` data frame contains observations collected by the US Environmental Protection Agency on 38 different models of car.


- `mapping = aes(...)`: How to map the variables in the data to aesthetics
  - Axes, size of points, intensities of colors, which colors, shape of points, lines/points
- Then say what type of plot you want:
  - boxplot, scatterplot, histogram, ...
  - these are called 'geoms' in ggplot's grammar, such as `geom_point()` giving scatter plots


```{r, geoms, eval=FALSE}
library(ggplot2)
... + geom_point() # Produces scatterplots
... + geom_bar() # Bar plots
.... + geom_boxplot() # boxplots
... #
```

You link these steps by *literally* adding them together with `+` as we'll see.

**Try it:** What other types of plots are there? Try to find several more `geom_` functions.

## Mappings Link Data to Things You See

```{r}
library(gapminder)
library(ggplot2)
gapminder
p <- ggplot(data = gapminder,
            mapping = aes(x = gdpPercap, y = lifeExp))
p + geom_point()
```

Above we've loaded a different dataset and have started to explore a particular relationship. Before putting in this code yourself, try to intuit what *might* be going on.

Any ideas?

Here's a breakdown of everything that happens after the `p<- ggplot()` call:

- `data = gapminder` tells ggplot to use gapminder dataset, so if variable names are mentioned, they should be looked up in gapminder
- `mapping = aes(...)` shows that the mapping is a function call. There is a deeper logic to this that I will disucss below, but it's easiest to simply accept that this is how you write it. Put another way, the `mapping = aes(...)` argument *links variables* to *things you will see* on the plot.
- `aes(x = gdpPercap, y = lifeExp)` maps the GDP data onto `x`, which is a known aesthetic (the x-coordinate) and life expectancy data onto `x`
  - `x` and `y` are  predefined names that are used by `ggplot` and friends

---

**Exercise 1:** Let's briefly return to the `mpg` data. Among the variables in mpg are:

- `displ`, a car’s engine size, in litres.

- `hwy`, a car’s fuel efficiency on the highway, in miles per gallon (mpg). A car with a low fuel efficiency consumes more fuel than a car with a high fuel efficiency when they travel the same distance.

Generate a scatterplot between these two variables. Does it capture the intuitive relationship you expected? What happens if you make a scatterplot of `class` vs `drv`? Why is the plot not useful?

---


> "The greatest value of a picture is when it forces us to notice what we never expected to see."" — John Tukey

In the plot you made above, one group of points seems to fall outside of the linear trend. These cars have a higher mileage than you might expect. How can you explain these cars?

Let’s hypothesize that the cars are hybrids. One way to test this hypothesis is to look at the class value for each car. The `class` variable of the `mpg` dataset classifies cars into groups such as compact, midsize, and SUV. If the outlying points are hybrids, they should be classified as compact cars or, perhaps, subcompact cars (keep in mind that this data was collected before hybrid trucks and SUVs became popular).

You can add a third variable, like `class`, to a two dimensional scatterplot by mapping it to an aesthetic. An aesthetic is a visual property of the objects in your plot. Aesthetics include things like the size, the shape, or the color of your points. You can display a point (like the one below) in different ways by changing the values of its aesthetic properties. Since we already use the word "**value**" to describe data, let’s use the word "**level**" to describe aesthetic properties. Thus, we are interested in exploring `class` as a level.

You can convey information about your data by mapping the aesthetics in your plot to the variables in your dataset. For example, you can map the colors of your points to the class variable to reveal the class of each car. To map an aesthetic to a variable, associate the name of the aesthetic to the name of the variable inside `aes()`. `ggplot2` will automatically assign a unique level of the aesthetic (here a unique color) to each unique value of the variable, a process known as scaling. `ggplot2` will also add a legend that explains which levels correspond to which values.

---

**Exercise 1b:** Using your previous scatterplot of `displ` and `hwy`, map the colors of your points to the class variable to reveal the class of each car. What conclusions can we make?

---


Let's explore our previously saved `p` in greater detail. As with Exercise 1, we'll add a *layer*. This says how some data gets turned into concrete visual aspects.

```{r, scatter_plot}
p + geom_point()
p + geom_smooth()
```

**Note:** Both of the above geom's use the same mapping, where the x-axis represents `gdpPercap` and the y-axis represents `lifeExp`. You can find this yourself with some ease. But the first one maps the data to individual points, the other one maps it to a smooth line with error ranges.

We get a message that tells us that `geom_smooth()` is using the method = 'gam', so presumably we can use other methods. Let's see if we can figure out which other methods there are.

```{r, smoothing_methods, eval=FALSE}
?geom_smooth
p + geom_point() + geom_smooth() + geom_smooth(method = ...) + geom_smooth(method = ...)
p + geom_point() + geom_smooth() + geom_smooth(method = ...) + geom_smooth(method = ..., color = "red")
```

You may start to see why `ggplot2`'s way of breaking up tasks is quite powerful: the geometric objects can all reuse the *same* mapping of data to aesthetics, yet the results are quite different. And if we want later geoms to use different mappings, then we can override them -- but it isn't necessary.

Consider the output we've explored thus far. One potential issue lurking in the data is that most of it is bunched to the left. If we instead used a logarithmic scale, we should be able to spread the data out better.

```{r, scale_coordinates}
p + geom_point() + geom_smooth(method = "lm") + scale_x_log10()
```
**Try it:** Describe what the `scale_x_log10()` does. Why is it a more evenly distributed cloud of points now? (2-3 sentences.)

Nice. We're starting to get somewhere. But, you might notice that the x-axis now has scientific notation. Let's change that.

```{r, scales, eval=FALSE}
library(scales)
p + geom_point() +
  geom_smooth(method = "lm") +
  scale_x_log10(labels = scales::dollar)
p + geom_point() +
  geom_smooth(method = "lm") +
  scale_x_log10(labels = scales::...)
```

**Try it:** What does the `dollar()` call do? How can you find other ways of relabeling the scales when using `scale_x_log10()`?

```{r, dollar_answer}
?dollar()
```

## The Recipe

1. Tell the `ggplot()` function what our data is.
2. Tell `ggplot()` *what* relationships we want to see. For convenience we will put the results of the first two steps in an object called `p`.
3. Tell `ggplot` *how* we want to see the relationships in our data.
4. Layer on geoms as needed, by adding them on the `p` object one at a time.
5. Use some additional functions to adjust scales, labels, tickmarks, titles.
  - e.g. `scale_`, `labs()`, and `guides()` functions

As you start to run more `R` code, you’re likely to run into problems. Don’t worry — it happens to everyone. I have been writing code in numerous languages for years, and every day I still write code that doesn’t work. Sadly, `R` is particularly persnickity, and its error messages are often opaque.

Start by carefully comparing the code that you’re running to the code in these notes. `R` is extremely picky, and a misplaced character can make all the difference. Make sure that every ( is matched with a ) and every " is paired with another ". Sometimes you’ll run the code and nothing happens. Check the left-hand of your console: if it’s a +, it means that R doesn’t think you’ve typed a complete expression and it’s waiting for you to finish it. In this case, it’s usually easy to start from scratch again by pressing ESCAPE to abort processing the current command.

One common problem when creating ggplot2 graphics is to put the + in the wrong place: it has to come at the end of the line, not the start.



### Mapping Aesthetics vs Setting them

```{r, mapping_vs_setting}
p <- ggplot(data = gapminder,
            mapping = aes(x = gdpPercap, y = lifeExp, color = 'yellow'))
p + geom_point() + scale_x_log10()
```

This is interesting (or annoying): the points are not yellow. How can we tell ggplot to draw yellow points?

```{r, yellow_points, eval=FALSE}
p <- ggplot(data = gapminder,
            mapping = aes(x = gdpPercap, y = lifeExp, ...))
p + geom_point(...) + scale_x_log10()
```

**Try it:** describe in your words what is going on.
One way to avoid such mistakes is to read arguments inside `aes(<property> = <variable>)`as *the property <property> in the graph is determined by the data in <variable>*.

**Try it:**  Write the above sentence for the original call `aes(x = gdpPercap, y = lifeExp, color = 'yellow')`.

Aesthetics convey information about a variable in the dataset, whereas setting the color of all points to yellow conveys no information about the dataset - it changes the appearance of the plot in a way that is independent of the underlying data.

Remember: `color = 'yellow'` and `aes(color = 'yellow')` are very different, and the second makes usually no sense, as `'yellow'` is treated as *data*.

```{r, exercise_args_for_smooth}
p <- ggplot(data = gapminder,
            mapping = aes(x = gdpPercap, y = lifeExp))
p + geom_point() + geom_smooth(color = "orange", se = FALSE, size = 8, method = "lm") + scale_x_log10()
```

**Try it:**  Write down what all those arguments in `geom_smooth(...)` do.

```{r, gapminder_with_labels}
p + geom_point(alpha = 0.3) +
  geom_smooth(method = "gam") +
  scale_x_log10(labels = scales::dollar) +
  labs(x = "GDP Per Capita", y = "Life Expectancy in Years",
       title = "Economic Growth and Life Expectancy",
       subtitle = "Data Points are country-years",
       caption = "Source: Gapminder")
```

Coloring by continent:

```{r, gapminder_color_by_continent}
library(scales)
p <- ggplot(data = gapminder,
            mapping = aes(x = gdpPercap, y = lifeExp, color = continent, fill = continent))
p + geom_point()
p + geom_point() + scale_x_log10(labels = dollar)
p + geom_point() + scale_x_log10(labels = dollar) + geom_smooth()
```

**Try it:**  What does `fill = continent` do? What do you think about the match of colors between lines and error bands?

```{r, gapminder_color_by_continent_single_line}
p <- ggplot(data = gapminder,
            mapping = aes(x = gdpPercap, y = lifeExp))
p + geom_point(mapping = aes(color = continent)) + geom_smooth() + scale_x_log10()
```

**Try it:**  Notice how the above code leads to a single smooth line, not one per continent. Why?

**Try it:**  What is bad about the following example, assuming the graph is the one we want? Think about why you should set aesthetics at the top level rather than at the individual geometry level if that's your intent.

```{r, many_continents}
p <- ggplot(data = gapminder,
            mapping = aes(x = gdpPercap, y = lifeExp))
p + geom_point(mapping = aes(color = continent)) +
  geom_smooth(mapping = aes(color = continent, fill = continent)) +
  scale_x_log10() +
  geom_smooth(mapping = aes(color = continent), method = "gam")
```
---

**Exercise 2:** Imagine that you have been hired as a consultant for a Portuguese bank. This bank recently undertook a series of marketing campaigns designed to increase deposits in their long-term savings accounts. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to assess if the consumer subscribed to the product (bank term deposit; coded as `yes` and `no` in the column `y`). As a result of this out-reach, we have the dataset `bank.csv`, which you can find [here](assets/bank.csv). I have given you a subset of a larger dataset. Most of the data is self-explanatory. However, a few inputs require that I give you more information: `default` describes if the customer has credit in default. `housing` and `loan` describe if the customer has either a mortgage or another type of loan already with the bank.

You must read this data into `R` and provide two data visualizations that are pertinent to making business decisions on this dataset. Describe, in a one-page memo, what information you hope to convey to executives about the outreach or about the dataset more generally.
