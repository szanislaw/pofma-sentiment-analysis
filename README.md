# Analysis of POFMA Notices over time

This project involves the **Classification of Actors** and **Sentiment Analysis** of all POFMA Notices issued from the enactment of the Protection from Online Falsehoods and Manipulation Act (POFMA) on 3 June 2019 to the latest revision date of 18 October 2024.

## Prerequisite packages (pip install):
pandas, torch, transformers, pypdf2, numpy, requests, bs4, json

## **Classification of Actors**

A classification model was developed using BERT to classify actors involved in a POFMA Notices by taking the parsed input of the POFMA notice and giving its output in a CSV file format. 

POFMA Notices can be scraped using the _scraper-pofmanotice.py_ file in the main directory.

The classification script can be ran from the folder _/model/actor-model-main.py _

### Hyperparameters:

_lr_ - controls optimizer learning rate

_weight_decay_ - controls optimizer weight decay

_epochs_ - epochs to train model

_patience_ - patience for early stopping (due to the relatively small size of the dataset)

_threshold_ - binary prediction threshold (0.5 means any probability 0.5 and above will be considered a positive match to the actor)


This outputs a model that will be saved into /models/bert and tokenizer/bert, after which the file actor-predict.py can be run to classify the actors accordingly.

## **Sentiment Analysis of POFMA Notices**
A list of Reddit discussions of POFMA related news and articles are compiled and saved as a .csv format in the /data/pofma-related-articles-reddit.csv. 

_scraper-reddit.py_ can be run to scrape all comments from the Reddit articles in the compiled .csv 

_sentiment.py_ can be run to generate a Sentiment Analysis of each comment, consisting of POS (Positive), NEU (Neutral) and NEG (Negative). This is done using a pretrained model, BERTweet, and the sentiments are aggregated and will be output in the terminal upon completion.
