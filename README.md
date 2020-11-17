# Lecture 8: Train, Tune, and Serve ML Models Using FLASK

In this lecture, we will go through some widely-used ML approaches ranging from Naive Bayes to Feed-Forward 
Neural Network for constructing a supervised classifier.

## Labelled Data Sets

There are several resources that we could employ to obtain labelled data sets. They come from:

1. [The UCI ML repository](http://archive.ics.uci.edu/ml/datasets.php)

    There are some well-known data sets, which are popularly used in CS classes. Such as:
    - [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection#)
    - [Breast Cancer Wisconsin (Diagnostic) Data Set](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
    - [Human Activity Recognition Using Smartphones](http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
    - [Diabetes 130-US hospitals for years 1999-2008](http://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008#)

2. [Google Dataset Search](https://datasetsearch.research.google.com/)

3. Academia: authors of academic papers usually advertise their projects by freely providing data sets.

4. [Kaggle](https://www.kaggle.com/)
    - [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

5. [sklearn toy datasets](https://scikit-learn.org/stable/datasets/index.html#datasets)

## About This Lecture
We will work on the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection#) dataset to:

### Analyse the nature of this corpus
- Pre-process the dataset to analyse some implicit characteristics 
- Split the corpus into two: training set and testing set.
### Build and train an ML model
- Vectorize inputs into real-value vectors
- Compare several ML models to choose the best one.
### Inference 
- Predict label given a unlabelled input.

## Note
To run all experiments in this project, you MUST install all required libraries 
listed in [requirements.txt](./requirements.txt). One best way for doing that is to use a virtual environment 
as followings:
```commandline
# go to the working directory
python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
