# How to Eliminate Lost Revenue Due to Poor Customer Service

In 2014 NewVoiceMedia reported that U.S. businesses lose $41 billion per year due to poor customer service. 

From these statistics alone, it is obvious that technology that can identify, and pinpoint, the exact time when customers start to encounter poor customer service will be of great interest to brands when it comes to eliminating lost revenue due to customer service.

This project attempts develop this technology.

# Motivation

As part of this project, we sought to develop a binary classifier using supervised machine learning methods that would allow us to identify if a customer encountered a rude employee while at McDonald's. Functionally, our classifier would take in a text review and would determine if the review is indicative of the customer having an encounter with a rude employee. In this use-case, we have associated poor customer service with rude employees.

In terms of business value, our classifier could be then used to develop a real-time dashboard that would allow executive management to identify the exact location and time that customers are experiencing rude customer service. Knowing this would allow management to make changes can improve customer service and cut some of the financial losses associated with rude customer service.  To that end, our project is guided by this specific problem statement:

>Can we build a binary classifier to predict if a customer has experienced rude service by a McDonald's employee?

# Approach

For this project, we will develop a binary classifier to predict if a customer has experienced rude service by a McDonald's employee based on a text review that is provided by a customer.

In general, this project is a supervised machine learning problem.  This means that our machine learning model will need to "learn" of off text reviews that have been labeled as being rude or not. 

The data required for this project can be tricky to obtain since it would require a third party to scan each customer review and make a decision as to whether or not that particular review is referencing rude customer service. 

It is understood that a dataset such as this can introduce biases into the model since a third party would need to evaluate each review and make a decision about the review, however, we are approaching this project under the assumption that this is all of the data that our client, in this case, McDonald's, has in their databases. 

While there are many ways to build a binary classifier for our client, we will use the following methods to accomplish the task at hand. 

* Multinomial Naive Bayes
* Logistic Regression 
* k-Nearest Neighbors
* Support Vector Classifier

When it comes to assessing the performance for each of these models, we will use precision, recall, and F1 scores. The final model that was selected would then base on the highest F1 score out of all of the tuning and testing scenarios that are used.

# Dataset 

For this project, we used a dataset from Figure-Eight.com of negative McDonalds reviews. For this dataset, reviews were labeled from random areas, and reviews were asked to classify why the locations received low reviews. Options given were: Rude Service, Slow Service, Problem with Order, Bad Food, Bad, Neighborhood, Dirty Location, Cost, Missing Item.  In total, we used 1,525 labeled reviews for our assessment.  

Once the data was in hand, we undertook some steps to pre-process the data. This included removing punctuations, digits, English stop words and change the typeface to lower case. We also removed the extraneous spaces that resulted once we preprocessed the data.  Finally, we parsed out the label for each review as being rude or not rude.  We assigned rude reviews a class of `1` and not-rude reviews a class of `0`. 

One representative entry for a rude and not rude review is provided below.

```
# Rude (Class 1)
terrible customer service came in at pm and stood in front of the register and no one bothered to say
anything or help me for minutes there was no one else waiting for their food inside either just outside at
the window i left and went to chickfila next door and was greeted before i was all the way inside this
mcdonalds is also dirty the floor was covered with dropped food obviously filled with surly and unhappy
workers
```

```
# Not Rude (Class 0)
this mcdonalds has gotten much better usually my order would be wrong every single time so i would not leave
that window until i checked every single item i only hit up fast food once a month or so and it needs to be
worth it also the fries used to be cold and the cheese on the burger was never melted everything was just
lukewarm now my order has been right a few times in a row and my food hot also i love dining room usually
you wouldnt find me actually inside a fast food joint but this place has nice flooring stacked stone lots of
large windows and a flat screen tv usually on hln sometimes its nice to sneak away for a quick weekend
breakfast you know a little budget and time friendly mommy and me date

``` 

From the 1525 reviews, a total of 1022 reviews were labeled as `not rude` and 503 were labeled as `rude`.  

The null accuracy of the dataset is defined as the baseline accuracy that is achieved by always predicting the most frequent class. In this case, the most frequent class is "not rude" and has a null accuracy of 67%.  For completion, the baseline accuracy of the "rude" category is 33%. 

Once we cleaned the data, we proceeded to vectorize the text data either using the `CountVectorizer` or `tfidfvectorizer` functions in `scikit-learn`.  In terms of parameters used to vectorize the data set, we used the following:

`vect = TfidfVectorizer(stop_words='english', min_df=0.01, max_df=0.95, ngram_range=(1, 2))`

`vect = CountVectorizer(stop_words='english', min_df=0.01, max_df=0.95, ngram_range=(1, 2))`

Based on the parameters, we expected that setting `min_df=0.01` and `max_df=0.95` would be sufficient for this project since we would be building a corpus that will include words that are in 95% of the all of the documents but exclude rare tokens that are not in 1% of the documents.

Regarding the `n_gram` range, we selected a range of n_gram_range=(1,2) which would mean that we are considering unigrams and bigrams within the corpus.  Again, we assumed that this range is sufficient since we did observe tokens like `McDonald's employee` or `drive-through window` in the corpus.

Once we vectorized the dataset, we ended up with a feature matrix that was (1525, 758).  This means that we have 1,525 reviews and each review has 758 total features. The corresponding labels did not need to be vectorized since we encoded them as being 0 or 1 for reviews and that is "not rude" and "rude," respectively.  This presented us with an output vector that was 1525 x 1 in size. 

With the vectorized dataset in hand, we proceed to split the data using the `train_test_split` function in scikit-learn.  This allowed us to separate the entire dataset into testing and training data needed to evaluate performance or quality of the models that we were investigating.

`X_train, X_test, y_train, y_test = train_test_split(X_dtm, y, test_size=0.30, random_state=42)`

Once we applied train test split, we used a total of xxx reviews for the testing set and xxx reviews from the training set.


# Methods 

To build or classifier, we used a standard work flow that included:

* Instantiate a model.
* Use grid search to find the optimal hyper parameters for the given model.
* Fit the training data to the instantiated model.
* Predict the classes from the testing data.
* Evaluate confusion matrix to evaluate model performance.
* Create ROC charts.
* Create learning charts to determine over or under-fitting.

We applied this method to data that was vectorized using `CountVectorizer` as well as `TFIDFVectorizer`.

For the machine learning models used for this project, we opted to use the following.

### Multinomial Naive Bayes
Multinomial Naive Bayes is model that can classify data based on the simple or naive assumption that that every feature in the data set is independent of the others when predicting the class of a new data point. Multinomial Naive Bayes has been studied extensively in text classification examples and is most well-known for predicting if an email is spam or not and is the reason that we selected it as one of the models for our project.

### Logistic Regression 

Logistic regression is a method that predicts if something belongs to one class or another, based on the conditional probability of the features.  In this case, the lr model can predict of a review is rude or not based on the distribution and frequency of the words in all of the reviews.

### k-Nearest Neighbors

The knn model works on a concept of feature similarity. That is, given a new data point, the model will assign a class prediction based on how similar the new point to its k-nearest neighbors is.

### Support Vector Classifier

The support vector classifiers work by plotting each data point in n-dimensional space and then finding a hyperplane that can separate each of the points in the dataset. 

## Grid Search and Cross-Validation

To tune each model, we used the `grid_search` function within scikit-learn.  As noted by [Jason Brownlee](https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/. 
):

> Grid search is an approach to parameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid. 

Also, each time that we used grid search, we also incorporated 10-fold cross validation into the tuning method.  

When assessing model performance, it is vital that the model is tested on data that the model has not seen. The reason for this is that you don't want to introduce bias into the model that could affect the ability to generalize. As a result, it is common practice to split the data up into training and testing sets as in the case of `train_test_split` in scikit-learn.

In `train_test_split` the testing set, is always going to be the same data points. Since the goal of building a model is to have a model that generalizes on out of sample data, it is possible, that biases could be introduced into the dataset when doing 'train_test_split' since the testing and training set will always include the same data point.

To get around this potential source of bias, it is customary to test the performance of the model using different points in the testing and training set. The problem here, however, is that we only have a set amount of data points, and the cardinal rule is to never test a model using data points that the model has already seen.

To solve this problem, one can use k-fold cross-validation. In k-fold cross-validation, the data set is sub divided into 'k' divisions of equal size. Once the divisions have been made, the model is trained with 'k-1' data points, and the then tested with data from the kth subdivision.

# Model evaluation

To evaluate the performance of each model, we opted to analyze a variety of model evaluation metrics.

### Learning curves
The core goal of a supervised machine learning model is to develop the best estimate (or target function) of a function that can relate an output variable to a set of input data.  When developing this target function, it is possible to introduce three types of errors into the model:  bias, variance and irreproducible error.

A bias error results from assumptions that overly simplifies the model's learning process given a set of training data. 

On the other hand, a variance error results from assumptions that are used to turn a given model to a specific training data set. 

When training a given machine learning model, there is a constant battle between the bias and variance of a given model on a specific set of training data. 

As noted here, [during the training process, machine learning models] often run the risk of over-extrapolating or over-interpolating from the data that they are trained on.  

"There is a very delicate balancing act when machine learning algorithms try to predict things," writes Geng and Shih in the "Machine Learning Crash Course: Part 4 - The Bias-Variance Dilemma." 

"On the one hand," they write, "we want our algorithm to model the training data very closely, otherwise we’ll miss relevant features and interesting trends. However, on the other hand we don’t want our model to fit too closely, and risk over-interpreting every outlier and irregularity."

https://ml.berkeley.edu/blog/2017/07/13/tutorial-4/

To determine if a model is under-fitting or overfitting to the training data, we will evaluate the fit of a model using the learning curves.

[Introduce picture of the three outcomes]

### Precision 
The precision of a model is defined as the number of true positives over the sum of the number of true positives and number of false positives.  As it relates to the problem set, the recall answers the question, of all the reviews that are not rude, how many did the model predict as being not rude? 

### Recall
The recall of a model is defined as the number of true positives over the sum of the number of true positives and number of false negatives.  As it relates to the problem set, the recall answers the question, of all the reviews that are not rude, how many did the model predict as being not rude? 

### F1 Score.
The F1 score is the harmonic mean of the precision and the recall scores.  The F1 score can be between zero and one with the best scores being those that are closer to one.

### The use of precision and recall in the model selection process
The precision, recall become important metrics when it comes to evaluating performance on a specific model, therefore, four outcomes exist.  

**True positives (TP).**  True positives are those reviews where the model predicted that a given review was "not rude" and the review as actually labeled as "not rude".

**True negatives (TN).** True negatives are those reviews where the model predicted that labeled a given review was "rude" and the review as actually labeled as "rude".

**False positives (FP).** False positives are defined as those reviews where the model predicted that a given review was "not rude" however, the actual review as labeled as being "rude."

**False negatives (FN).**  False negative reviews are defined as those where the model predicted that a given review was "rude" however, the actual review as labeled as being "not rude."

When evaluating the results from the models, it is important to note that a perfect model should not result in any FP or FN classifications, however, this is almost never the case. Each model will almost always misclassify some of the points in the dataset, and that should be expected.

The degree that FP and FN classifications are tolerated are a function of the business case.  For this project, our objective is to build a classifier that can predict if a customer has experienced rude service by a McDonald's employee.  This means that our model should be optimized to ensure that all of the rude reviews are identified.  Therefore, our model needs to be achieved a low rate of of false positives even at the expense of false negatives. 

In other words, it is okay for our model to predict that a review was "rude" even though it wasn't labeled as such, however, it is **not** okay for our model to predict that a review was "not rude," when it was labeled as being "rude." The reason for this is that our customer has taken a position to review all instances of "rude" reviews even if it results in additional effort to inspect the not rude reviews.  This results in having a high precision model since FP are not acceptable, but FN can be accepted.

* Multinomial Naive Bayes (ROC, F1 and null accuracy)
* Logistic Regression (ROC, F1 and null accuracy)
* k-Nearest Neighbors (ROC, F1 and null accuracy)
* Support Vector Classifier (ROC, F1 and null accuracy)


# Conclusion/Future Work

Based on the models and evaluation metrics observed, we have concluded that the best model to classify reviews is the XXX.

Problems with the data include:

Future work includes: 
Larger data set.
Reviews that are not so negative.

***
