# The Cost of Rude Service:  How Can McDonald’s Quickly Identify Rude Customer Service using Machine Learning

**By: [Felipe Sinohui](https://www.linkedin.com/in/felipesinohui/), and [Grant Aguinaldo](https://www.linkedin.com/in/grantaguinaldo/)**

***

In 2011, [Accenture](https://www.accenture.com/us-en/new-applied-now) estimated that 67% of customers have no problem with switching providers due to poor customer service which includes's unhelpful/rude staff.  More recently, in [2017 NewVoiceMedia](https://www.newvoicemedia.com/blog/the-62-billion-customer-service-scared-away-infographic) reported that U.S. businesses lose $62 billion per year due to poor customer service. 

From these statistics, there is a clear relationship between customer service and lost revenue.  Therefore, any efforts that can identify when is a customer receiving poor customer service is of great interest to businesses who want to reduce the amount of revenue lost by poor customer service.

This project uses machine learning and the scikit-learn library to identify the best model, out of five, that can be used to predict when McDonald's customers are encountering poor customer service based of off text reviews.  

### Problem Statement

Specifically, this project will answer this problem statement:

>Can a machine learning model be developed and tuned in a testing environment to predict if customers are encountering rude service based on text reviews provided by a customer.  

When our selected model is implemented at scale, a customer service analyst would be then able to investigate all instances where the model has predicted rude service by McDonald's employees.  This would allow the analyst to make recommendations to the front-line and upper management to improve customer service.

### Dataset

The dataset used for this projecct came from the website [Figure-Eight.com](https://www.figure-eight.com/wp-content/uploads/2016/03/McDonalds-Yelp-Sentiment-DFE.csv).  

This dataset included a total of 1,525 labeled reviews provided by McDonald's customers.  Each review represents an encounter with McDonald's staff at a specific location.  To label each review, an analyst reviewed each and identified (labeled) the policy violations that have occurred simply by reading the review.  These policy violations include: 

* Rude service, 
* Slow service, 
* Problem with an order, 
* Bad food, 
* Bad neighborhood, 
* Dirty location, 
* Cost, and 
* Missing items. 

### Methods

Once we obtained the data, we proceeded to clean the data and conduct feature engineering to encode only the reviews that were labeled as being "rude." Finally, we created training and testing subsets out of the clean dataset to train our models.  

During the training phase, we followed a standard workflow to train each of the five machine learning models that included: 

* Vectorizing the text data (using CountVectorizer and TfIdfVectorizer);
* Instantiating a model;
* Using grid search to find the optimal hyperparameters for each model;
* Refitting training data to the optimized model;
* Predicting the classes from the testing data;
* Evaluating model performance using model evaluation metrics; and 
* Evaluating the fit of the model.

### Results

Out of the five total models that we trained, the recall score (for rude reviews) ranged from 1.3 to 70%. Since we sought to identify a model with the highest recall score we selcted the **CountVectorizer/Multinomial Naive Bayes** model since the recall score was calcualted to be 70%

### Discussion

By using a high recall model, that is, a model that minimizes the number of false negatives, we are ensuring that all reviews that are classified as rude have the opportunity to be reviewed by the analyst.  In passing we define a false negative as being a situation where the model predicted that a review was "not rude," but the review was *actually* "rude." 

As it realtes to the utility of this model, we have taken the postiion that, it's okay for our model to predict that a review was "rude" even though it wasn't labeled as such, however, it is *not okay* for our model to predict that a review was "not rude," when it was acutally labeled as being "rude."

### Summary

In summary, we have identified and turned a machine learning model that can predict if a customer is encountering rude customer service solely based on a text review provided by the customer. Our model uses a Multinomial Naive Bayes model that affords a high recall to ensure that all rude reviews are identified and examined so that changes to the operation can be made promptly to avoid future losses in revenue.  
