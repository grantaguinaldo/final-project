# Understanding The Language Patterns Behind Restaurant Reviews

![](./images/jay-wennington-2065-unsplash.jpg)

For the final group project, we will use natural language processing to discover patterns within the [Yelp Academic Dataset](https://www.kaggle.com/yelp-dataset/yelp-dataset).

As a whole, the dataset contains: 

* 5,200,000 user reviews;
* Information on 174,000 businesses; and 
* The data spans 11 metropolitan areas.

> What can we learn about the language patterns used to give five-star ratings at restaurants as compared to one-star ratings.  
> 
> In addition, we will also make recommendations/changes that business can make to ad-copy to encourage five-star reviews at restaurants.

This project is directly motivated by the fact that online reviews have become a core tool in the decisions that people make about businesses.  Because of this, we are interested in what insight, if any, can be extracted by the free-form text by people using the Yelp platform that will allow a business to *coax* reviews to leave a five-star review. 

To answer these two over arching questions, this project will answer the gather insight into the following.

* First, what are the **top ten tokens** that are used when giving a restaurant a 5-, 4-, 3-, 2- or 1-star review. 

* Second, what are the **top 10 n-grams** that are used when giving a restaurant a 5-, 4-, 3-, 2- or 1-star review.

* Third, can we train a multi-class classifier to predict the number of stars that a user will give a restaurant?

* Fourth, is there a relationship between the **length of a review and the number of stars** awarded to a single restaurant?

* Fifth, when it comes to restaurants in Los Angeles, where are the local hotbeds?

* Sixth, is there a relationship between the sentiment of a review and the number of stars given or the length of the review?  

As an optional question, we may also do a topic model in an attempt to find *topics* that may exists in the dataset.

### Tech Stack

This project will using the following technologies.

* MySQL
* Python
* Pandas
* SK Learn
* Flask
* Keras
* Heroku
* JavaScript
* HTML/CSS/Bootstrap
* Numpy
* Matplotlib/Seaborn
* Jupyter Notebook
* Plotly

## Team Members and Location of Repo
The project team includes four members. The proposed team name is "Reading Between The Lines."

* Felipe Sinohui
* Grant Aguinaldo

The repo for this project can be found here. To ensure that we have no issues with version control, we will be the common pull/branch/merge workflow for git. For more on using git, you can check out the YouTube video below or by clicking here.

## Presentation

The presentation for this project will be in the form of a **slide deck** as well as a **web-hosted app**.  The slide deck will frame the overall problem statement and present the result of a brief explore data analysis (EDA).

## Work Breakdown

This project will be broken down into several components.

### 1.  Loading the dataset into MySQL

The data for this project will come from [Yelp.com](https://www.yelp.com/dataset/documentation/sql) in the form a SQL dump.  

### 2.  Understand the dataset within the context of the problem statement.

Using the schema, preform preliminary queries of the database to understand what data is included in the database.  For example, we may need to do multiple joins to get the data we need.  In addition, I have a feeling that the city of Los Angeles is not included in the dataset.  Therefore, we need to know what data is included in order to adjust our scope accordingly. 

We will commit the SQL queries to github so that we both have access to the query since we won't be able to both access any views that we create.

### 3.  Extract the necessary data from the dataset to complete questions 1 and 2.

This can be either done as a separate step or after we have a trained model since classifiers like multinomial naive bays will expose an attribute that will give us a distribution of the tokens in each class. We will need to decide weather we want to use a simple bag-of-words or TFIDF.

* What are the **top ten tokens** that are used when giving a restaurant a 5-, 4-, 3-, 2- or 1-star review. 

* What are the **top ten n-grams** (to be decided) that are used when giving a restaurant a 5-, 4-, 3-, 2- or 1-star review.

### 4.  Train a classifier to predict the star rating as a function of the review text.
For the classifier, we will initially test a few classification algorithms:  logistic regression, naive bays, k-NN and a possibly neural network. Once we find the one that has the best accuracy, we will proceed with the front end, using the given model. At this point, I predict that the model accuracy will not be > 50%. 

### 5.  Build slide deck using Google Slides.

When it comes to the slide deck, we will follow the flow that is being used within this proposal.  The core part of our presentation is to tell a story about exposing the language patterns that are used when rating a restaurant. 

This slide deck will present statistics of thea overall dataset, as well as the results from the EDA.

### 6.  Build web app that loads the trained classifier and predicts the star rating in real-time (or as close to real-time as is possible).

More will be provided on this later, however, an initial wireframe of the app is shown below.

***