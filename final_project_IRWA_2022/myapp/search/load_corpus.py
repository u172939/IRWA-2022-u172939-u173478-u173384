import pandas as pd
import json
from typing import List

from myapp.core.utils import load_json_file
from myapp.search.objects import Tweet

_corpus = {}

def load_doc_to_tweet():
    with open('doc_to_tweet.json', 'r') as f:
        doc_to_tweet = json.load(f)
    return doc_to_tweet

def load_corpus(path) -> List[Tweet]:
    """
    Load file and transform to dictionary with each document as an object for easier treatment when needed for displaying
     in results, stats, etc.
    :param path:
    :return:
    """
    df = _load_corpus_as_dataframe(path)
    df.apply(_row_to_doc_dict, axis=1)
    return _corpus


def _load_corpus_as_dataframe(path):
    """
    Load documents corpus from file in 'path'
    :return:
    """
    json_data = load_json_file(path)
    tweets_df = _load_tweets_as_dataframe(json_data)
    # Rename columns to obtain: Tweet | Username | Date | Hashtags | Likes | Retweets | Url 
    corpus = tweets_df.rename(
        columns={"tweet_id": "Tweet_id",
                "doc_id": "Doc_id",
                "full_text": "Full_text",
                "text": "Tweet",
                "username": "Username",
                "followers_count": "Followers_count",
                "hashtags":"Hashtags",
                "date": "Date",
                "likes": "Likes",
                "retweets": "Retweets",
                "url":"Url"})

    # select only interesting columns
    filter_columns = [ "Tweet_id","Doc_id","Full_text","Tweet", "Username", "Date", "Hashtags", "Likes", "Retweets", "Url","Followers_count"]
    corpus = corpus[filter_columns]
    return corpus


def _load_tweets_as_dataframe(json_data):
    data = pd.DataFrame(json_data).transpose()
    data["tweet_id"] = data.index
    # parse user data as new columns and rename some columns to prevent duplicate column names
    return data


def _build_tags(row):
    tags = []
    # for ht in row["hashtags"]:
    #     tags.append(ht["text"])
    for ht in row:
        tags.append(ht["text"])
    return tags


def _build_url(row):
    url = ""
    try:
        url = row["entities"]["url"]["urls"][0]["url"]  # tweet URL
    except:
        try:
            url = row["retweeted_status"]["extended_tweet"]["entities"]["media"][0]["url"]  # Retweeted
        except:
            url = ""
    return url


def _clean_hashtags_and_urls(df):
 
    df["Url"] = df.apply(lambda row: _build_url(row), axis=1)
    # df["Url"] = "TODO: get url from json"
    #df.drop(columns=["entities"], axis=1, inplace=True)


def load_tweets_as_dataframe2(json_data):
    """Load json into a dataframe

    Parameters:
    path (string): the file path

    Returns:
    DataFrame: a Panda DataFrame containing the tweet content in columns
    """
    # Load the JSON as a Dictionary
    tweets_dictionary = json_data.items()
    # Load the Dictionary into a DataFrame.
    dataframe = pd.DataFrame(tweets_dictionary)
    # remove first column that just has indices as strings: '0', '1', etc.
    dataframe.drop(dataframe.columns[0], axis=1, inplace=True)
    return dataframe


def load_tweets_as_dataframe3(json_data):
    """Load json data into a dataframe

    Parameters:
    json_data (string): the json object

    Returns:
    DataFrame: a Panda DataFrame containing the tweet content in columns
    """

    # Load the JSON object into a DataFrame.
    dataframe = pd.DataFrame(json_data).transpose()

    # select only interesting columns
    filter_columns = ["id", "full_text", "created_at", "entities", "retweet_count", "favorite_count", "lang"]
    dataframe = dataframe[filter_columns]
    return dataframe


def _row_to_doc_dict(row: pd.Series):
    _corpus[row['Tweet_id']] = Tweet(row['Tweet_id'], row["Doc_id"],row['Tweet'],row["Full_text"], row['Date'], row['Likes'],row['Retweets'],row['Url'], row['Hashtags'],row["Username"],row["Followers_count"])
