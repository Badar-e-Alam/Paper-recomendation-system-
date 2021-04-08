
import pandas as pd

import numpy, pandas, re, os ,glob, sklearn 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from bs4 import BeautifulSoup    
from nltk.corpus import stopwords 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn import preprocessing 
from sklearn . preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs







'''
import citation_extractor
import refextract
from refextract import extract_journal_reference


reference = extract_journal_reference('J.Phys.,A39,13445')
'''







def dbscan(X, eps, min_samples):
    ss = StandardScaler()
    X = ss.fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X)
    y_pred = db.fit_predict(X)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print n_clusters_
    print n_noise_
    plt.scatter(X[:,0], X[:,1],c=y_pred, cmap='Paired')
    plt.title("DBSCAN")
    plt.show()

X = pd.read_csv('cit2.csv',sep=',')
#print X
#X=numpy.array(X)
#print X.reshape(1,-1)
#labels = db.labels_



eps=0.3
min_samples=13

dbscan(X, eps, min_samples)