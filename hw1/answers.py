r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
An ideal pattern would be about a normal distribution centered on 0 - most of the features should achieve low error,
but we don't want all the features to be very close to 0 to not overfit the model.
Comparing the top-5 features plot and the final plot, we can see that the residuals are much more sparse around 0, and we get
a larger error for many features.
(notice that the graph is also in a larger scale). 
The plot is also only for data the model trained on (no test set).
We can infer the model doesn't generalize well.

As for the final plot, the residuals are much more centered around 0, with a few outliers (and these, too, are less
drastic than the top 5 model). We also tested this model with the test set
and it seems to be quite close to the training set - mostly centered around 0 with a few outliers. 
We conclude that this model generalizes much better. 
"""

part4_q2 = r"""
**Your answer:**
1. I think we used logspace because this variable is more sensitive to changes, so we chose logspace to be more thorough with our 
parameters evaluation. Choosing logspace instead of linspace means that we take smaller steps when changing this parameter value -
we are more careful and check more options.

2. Overall the model trained once for every pair of ($lambda$, $degree$) $k_fold$ times.
We get $ 3*20*3 = 180 $ times in total.


"""

# ==============
