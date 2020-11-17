import os
import time
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve

from sklearn.naive_bayes import MultinomialNB  # NB
from sklearn.neighbors import KNeighborsClassifier  # k-NN
from sklearn.linear_model import SGDClassifier  # logistic regression
from sklearn.tree import DecisionTreeClassifier  # DT
from sklearn.svm import LinearSVC  # linear SVM
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    HashingVectorizer,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


def pkl_write(data, filename="data.pickle"):
    with open(filename, "wb") as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def pkl_read(filename="data.pickle"):
    with open(filename, "rb") as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
    return data


def build_cls(ml_cls="NB", tfidf=False, use_hash=False, scaler=False):
    print("- Construct the baseline...")
    start = time.time()
    if ml_cls == "kNN":
        KNeighborsClassifier(n_neighbors=5)
    elif ml_cls == "LR":
        # Logistic Regression
        classifier = SGDClassifier(verbose=5, loss="log", max_iter=100)
    elif ml_cls == "DT":
        classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
    elif ml_cls == "SVM":
        classifier = LinearSVC(verbose=5)
    elif ml_cls == "MLP":
        classifier = MLPClassifier(random_state=1, max_iter=100)
    elif ml_cls == "AB":
        classifier = AdaBoostClassifier()
    elif ml_cls == "GB":
        classifier = GradientBoostingClassifier(verbose=5)
    elif ml_cls == "RF":
        classifier = RandomForestClassifier(n_estimators=100, verbose=5)
    else:
        # DEFAULT: NB
        classifier = MultinomialNB()
    settings = []
    if use_hash:
        settings = [("vectorizer", HashingVectorizer())]
    elif tfidf:
        settings = [("vectorizer", TfidfVectorizer())]
    else:
        # DEFAULT: BOW counting
        settings = [("vectorizer", CountVectorizer())]

    # scaler cannot use with NB (MultinomialNB)
    if scaler and ml_cls != "NB":
        settings += [("scaler", StandardScaler())]

    settings += [("classifier", classifier)]
    model = Pipeline(settings)

    parameters = {
        "vectorizer__analyzer": ["word"],
        # 'vectorizer__analyzer': ['char', 'word'],
        "vectorizer__ngram_range": [(1, 5)],
        "vectorizer__min_df": [3, 5, 7],
        "vectorizer__binary": (True, False),
    }
    end = time.time()
    print("\t+ Done: %.4f(s)" % (end - start))
    return model, parameters


def train(args):
    data_train = pd.read_csv(args.train_file).sample(frac=1).reset_index(drop=True)
    # data_dev = pd.read_csv(args.dev_file).sample(frac=1).reset_index(drop=True)
    # data_merge = pd.concat([data_train, data_dev])
    # dev_fold = [-1]*len(data_train) + [0]*len(data_dev)
    # x_traindev, y_traindev = data_merge["Review"].to_numpy(), data_merge["Rating"].to_numpy()

    x_traindev, y_traindev = (
        data_train["Review"].to_numpy(),
        data_train["Rating"].to_numpy(),
    )
    pipeline, parameters = build_cls(
        args.ml_cls, args.tfidf, args.use_hash, args.scaler
    )

    print("- Train the baseline...")
    start = time.time()
    # model = GridSearchCV(pipeline, parameters, cv=PredefinedSplit(test_fold=dev_fold),
    #                      verbose=5,  scoring='f1_weighted')
    model = GridSearchCV(pipeline, parameters, cv=5, verbose=5, scoring="f1_weighted")
    model.fit(x_traindev, y_traindev)
    end = time.time()
    print("\t+ Done: %.4f(s)" % (end - start))
    best_model = model.best_estimator_
    save(best_model, args.model_name)
    return model


def save(model, mfile):
    print("- Save the model...")
    pkl_write(model, mfile)
    print("\t+ Done.")


def load(mfile):
    print("- Load the model...")
    model = pkl_read(mfile)
    print("\t+ Done.")
    return model


def evaluate(data, model_name):
    model = load(model_name)
    print("- Evaluate the baseline...")
    start = time.time()
    X_dev, y_true = data
    y_pred = model.predict(X_dev)
    mtrcs = class_metrics(y_true, y_pred)
    plot_confusion_matrix(model, X_dev, y_true)
    # plot_roc_curve(model, X_dev, y_true)
    end = time.time()
    print("\t+ Done: %.4f(s)" % (end - start))
    return mtrcs


def test(args, model_name):
    data_test = pd.read_csv(args.test_file).sample(frac=1).reset_index(drop=True)
    x_test, y_test = data_test["Review"].to_numpy(), data_test["Rating"].to_numpy()
    mtrcs = evaluate([x_test, y_test], model_name)
    return mtrcs


def class_metrics(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    f1_ma = metrics.precision_recall_fscore_support(y_true, y_pred, average="macro")
    f1_we = metrics.precision_recall_fscore_support(y_true, y_pred, average="weighted")
    f1_no = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    print("\t+ Accuracy: %.4f(%%)" % (acc * 100))
    measures = {
        "acc": acc,
        "prf_macro": f1_ma,
        "prf_weighted": f1_we,
        "prf_individual": f1_no,
    }
    return measures


def predict(sent, model_name):
    model = load(model_name)
    label = model.predict([sent]).tolist()[0]
    prob = model.predict_proba([sent]).max()
    print("- Inference...")
    print("\t+ %s with p=%.4f" % (label, prob))
    return label, prob


if __name__ == "__main__":
    """
    python baselines.py --train_file /media/data/langID/small_scale/train.csv --dev_file /media/data/langID/small_scale/dev.csv --test_file /media/data/langID/small_scale/test.csv --model_name ./results/small.NB.m --ml_cls NB
    """
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument(
        "--train_file",
        help="Trained file",
        default="../reviewscollection/tripadvisor_hotel_reviews_training.csv",
        type=str,
    )

    argparser.add_argument(
        "--dev_file",
        help="Developed file",
        default="../reviewscollection/tripadvisor_hotel_reviews_test.csv",
        type=str,
    )

    argparser.add_argument(
        "--test_file",
        help="Tested file",
        default="../reviewscollection/tripadvisor_hotel_reviews_test.csv",
        type=str,
    )

    argparser.add_argument(
        "--tfidf", action="store_true", default=False, help="tfidf flag"
    )

    argparser.add_argument(
        "--use_hash", action="store_true", default=False, help="hashing flag"
    )

    argparser.add_argument(
        "--scaler", action="store_true", default=False, help="scale flag"
    )

    argparser.add_argument(
        "--ml_cls", help="Machine learning classifier", default="MLP", type=str
    )

    argparser.add_argument(
        "--model_dir", help="Model dir", default="../reviewscollection", type=str
    )

    args = argparser.parse_args()

    model_dir, _ = os.path.split(args.model_dir)

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    args.model_name = os.path.join(args.model_dir, args.ml_cls + ".pickle")

    model = train(args)
    measures = test(args, args.model_name)
    label, prob = predict("excellent", args.model_name)