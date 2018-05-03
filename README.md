# The Cost of Rude Service:  How Can McDonald’s can Quickly Identify Rude Customer Service using Machine learning

**By: [Felipe Sinohui](https://www.linkedin.com/in/felipesinohui/), and [Grant Aguinaldo](https://www.linkedin.com/in/grantaguinaldo/)**

***

In [2017 NewVoiceMedia](https://www.newvoicemedia.com/blog/the-62-billion-customer-service-scared-away-infographic) reported that U.S. businesses lose $62 billion per year due to poor customer service. In 2011, [Accenture](https://www.accenture.com/us-en/new-applied-now) estimated that 67% of customers have no problem with switching providers due to poor customer service which includes's unhelpful/rude staff.

From these statistics, there is a clear relationship between customer service and lost revenue.  Also, any efforts that can identify when is a customer receiving poor customer service is of great interest to businesses who want to reduce the amount of revenue lost by poor customer service.

This project uses machine learning and the scikit-learn library to tune five models that can be used to predict when are McDonald's customers encountering poor customer service based of off text reviews.  

Specifically, we are answering this problem statement:

>Can a machine learning model be developed and tuned in a testing environment to predict if customers are encountering rude service.  

When implemented at scale, a customer service analyst would be able to investigate all instances where the model predicted rude service by McDonald's employees and make recommendations to the front-line and upper management to improve customer service.

The data used to train the models for this project came from [Figure-Eight.com](https://www.figure-eight.com/wp-content/uploads/2016/03/McDonalds-Yelp-Sentiment-DFE.csv).  

This dataset included a total of 1,525 labeled reviews.  Each review represents an encounter with McDonald's staff at a specific location.  To label each review, an analyst reviewed each and identified (labeled) the policy violations that have occurred that include: rude service, slow service, a problem with an order, bad food, bad, neighborhood, dirty location, cost, and missing items. 

Once we cleaned the data and encoding only the reviews that were labeled as being "rude," we created the training and testing subsets to train our models.  During the training phase, we followed a standard workflow to train each of the five machine learning models that included: 

* Vectorizing the text data;
* Instantiating a model;
* Using grid search to find the optimal hyperparameters for each model;
* Refitting training data to the optimized model;
* Predicting the classes from the testing data;
* Evaluating model performance using model evaluation metrics; and 
* Evaluating the fit of the model.

Out of the five total models that we trained, the model that afforded the best results was the Multinomial Naive Bayes/CountVectorizer model. We selected this model since it provided the highest recall score (for rude reviews) out of all of the models tested at 70% (range from 1.3 to 70%).  

By using a high recall model, that is, a model that minimizes the number of instances where the model predicted that a review was "not rude," but the review was actually "rude," we are ensuring that all reviews that are classified as rude are being reviewed by the analyst.

In summary, we have built a machine learning model that can predict if a customer is encountering rude customer service solely based on a text review provided by the customer. Our model uses a Multinomial Naive Bayes model that affords a high recall to ensure that all rude reviews are examined so that changes to the operation can be made promptly to avoid losses in revenue.  
