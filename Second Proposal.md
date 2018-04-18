# Building A Classifier to Identify Rude Employees at McDonald's

![](./images/jay-wennington-2065-unsplash.jpg)

For the final group project, we will use natural language processing to discover patterns within the the [McDonald's dataset](https://www.figure-eight.com/data-for-everyone/).

As a whole, the dataset contains 1,525 hand-labeled text reviews from various McDonalds in the US.  As noted the Figure Eight website, 

> A sentiment analysis of negative McDonald’s reviews. Contributors were given reviews culled from low-rated McDonald’s from random metro areas and asked to classify why the locations received low reviews. Options given were:
> 
> * Rude Service
> * Slow Service
> * Problem with Order
> * Bad Food
> * Bad Neighborhood
> * Dirty Location
> * Cost
> * Missing Item

This project is directly motivated by the fact that online reviews and feedback have become ubiquitous in society.  This creates an interesting situation for business since it allows valuable insight (both positive and negative) to be gained from these reviews and allows for changes to be made in a shorter period of time. 

In principle, it can be argued that the more a business knows/understands the sentiment and/or problems that are occurring, as reported by customers, the quicker can changes be made to correct the problem.  This in-turn, can allow for higher and a more pleasant experience between a brand and it's customers 

When it comes to customer satisfaction, [Time magazine](http://time.com/money/3976961/bad-customer-service-survey/
) notes that there are three core areas that get people really upset when it comes to customer service:

1. 75% say they’re “highly annoyed” when they can’t get a live person on the phone to help with a problem; in 2011, meanwhile, 71% of those polled by CR said they were “tremendously annoyed” when they couldn’t reach a live customer service rep over the phone. 
2. 75% are highly annoyed by rude or condescending employees.
3. 74% have been driven batty by disconnected phone calls placed to customer service lines.

Of this list, customers are highly annoyed by rude employees. Therefore, many problems can be eliminated if there was a way to determine and quantify rude employees, as noted by customers.

To that end, this project will work with the following problem statement:

Can a customer sentiment dashboard be built that will allow our management to understand the current sentiment and know if a specific location is expresses high frequency of rude service complaints. 

To address this problem statement, we will build a binary classifier that will be able to determine if employees are being rude to customers based on customer reviews and complaints. In addition, we will also gather insight into the following:

* What are the **top ten tokens** that are used when a review notes an employee is being rude? 

* What are the **top n-grams** that are used when a review notes an employee is being rude?

* Is there a relationship between the **length of a review and weather or not a review is noted to be rude or not?**

* Is there a relationship between the sentiment of a review and the number of stars given or the length of the review?  

As an optional question, we may also do a topic model or conduct k-means clustering in an attempt to find *topics* that may exists in the dataset.

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

### 1.  Download and clean the dataset.

The data for this project will come from [Figure Eight](https://www.figure-eight.com/data-for-everyone/).  Once we download the dataset, we'll need to extract out the labels (using a regex or something else) and clean the data. Cleaning the data will include:

* Removing digits
* Removing punctuation
* Converting to lower case

### 2.  Train several models using the dataset.

As part of this task, we will train several models using train/test/split and 10-fold cross validation. To start, we will use the following models:

* LR
* NB
* Ensamble
* Neural Network (keras)

For each model, we will plot the training and testing scores as well as the results from the confusion matrix to determine the accuracy from each model. We will record all of these data for the presentation.  During this process we'll also optimize the model using grid search.

### 3.  Build slide deck using Google Slides.

When it comes to the slide deck, we will follow the flow that is being used within this proposal.  The core part of our presentation is to tell a story about exposing the language patterns that are used when rating a restaurant. 

This slide deck will present statistics of the overall dataset, as well as the results from the EDA.

### 4.  Build web app that loads the trained classifier and predicts the star rating in real-time (or as close to real-time as is possible).

More will be provided on this later, however, an initial wireframe of the app is shown below.

...  more to come

***