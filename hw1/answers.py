r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
We can look on the extreme values of k, for k=1 we see a descent generalization error however not the best one,it seems that 1 nearest neighbor classifier gives a good indication to the class with a very complex model without any generalization. 
For k=training-size we get that we classify the same class for all samples therefore we will get a bad generalization error (we generalize too much). Generally speaking, increasing k can help lower the generalization error because it reduces the effect of noise, reduces the model complexity and creates a stable decision boundaries.
However, because larger k reduces the model complexity it can generalize too much and in turn increase the generalization error. 
In our case, we get the best result at k=3 and larger k increase the generalization error. 
To conclude small k’s can not generalize enough and larger ones can generalize too much, we need to find the sweet spot for k.
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
The delta value can be arbitrary because the lambda and the delta are linked. 
If there was no regularization (lambda equals 0) the delta value would have no effect because the model can increase the weights making the same classifier to fulfill margin restriction (delta values) therefore regularization is crucial for maintaining the margin. 
For a delta value we can determine the lambda value that is needed to maintain the requested value meaning its enough to set a certain delta value and by changing the lambda value we can restrict the wights to withhold the margin restriction for the chosen delta. 

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**
1.	we can see the linear model is learning the shapes of the digits. we can see it by that for certain digits the pixels in white that represent the shape of it have higher weights meaning they have brighter colors in the visualization images, the idea is that in each class the pixels which can help differ the class from other classes get higher/lower weights we can see them in the visualization by brighter/dimmer colors, in our case we can see it resembles the digit shape in each class.
I can explain some of the classification problems, for example classifying 2 as 7, we can see both 2 and 7 have some corresponding pixels with same large values of weights so it’s understandable of different images to miss classify. In general, two visualizations of classes who resemble each other can cause misclassification, or image which is considered a noise in a class (very different image from its class) for example the 5 in the first row, can be misclassified because it doesn’t alight with the weights learned (the bright colored pixels).

2.	We firstly mention the difference between the two classifiers, the SVM classifier learns weights for each class and chooses the class with the highest score of the dot product of the weights and the image, as opposed to KNN which chooses the class of the k closest samples. The similarity between those two classifies that both the choose the class that most resembles the image, SVM uses in a way the cosine similarity, explicitly the dot product to assess the distance between the image to each class and choose the closest one which resembles the knn for choosing the class that is closest to the image. (They both use different methods to assess the distance)


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**
The learning rate is good, we see a very moderate curve of increase in accuracy and decrease in loss that seem to reach a convergence value. 
If the learning rate was too high, we would have gotten zig zag graph, because each change would have a major impact on the weights which would have prevented us reaching convergence and would have changed the accuracy and loss dramatically in each step.
If the learning rate was too low, the loss would have decreased very slowly, and the accuracy would have increased very slowly which in turn would have cause the graph to be less curved. It will take a lot of time to reach convergence.

We see a small gap between the training and the validation accuracy, and the accuracy is relatively high, as we seen in the lecture, we can say there is a slightly underfitted to the training set. It is reasonable to assume more epochs will not help because the slope is small, however maybe a bigger model can achieve better results. If there was no gap at all or the accuracy was low, we would have said it is highly underfitted, but a small gap and high accuracy can be considered as slightly underfitted.

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
We can infer the model doesn't fit the data well, because we failed earning good results even on the training set. 

As for the final plot, the residuals are much more centered around 0, with a few outliers (and these, too, are less
drastic than the top 5 model). The results on the traning set are much better than before.
We also tested this model with the test set - the error is larger, but still we mostly centered around 0 with a few outliers. 
We conclude that this model now fits the data more and generalizes much better. 
"""

part4_q2 = r"""
**Your answer:**
1. I think we used logspace because we wanted to check more smaller parameters for lambda - and be efficient while doing so. 
Choosing logspace instead of linspace means that we check more smaller parameters, but we still check larger parameters too. 
In order to get the same volume of paramters in the `10^-3` to `10^-1` range, and still be able to get values in range of `10^1`, we 
will require sampling much more in linspace.


2. Overall the model trained once for every pair of ($lambda$, $degree$) $k_fold$ times.
We get $ 3*20*3 = 180 $ times in total.

"""


# ==============
