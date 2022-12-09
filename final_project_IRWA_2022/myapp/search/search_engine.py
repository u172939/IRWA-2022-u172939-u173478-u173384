import random
import json

from myapp.search.objects import ResultItem, Tweet
from myapp.core.utils import load_json_file

from .algorithms import create_index_tfidf, preprocessing, rank_documents, rankTweetsBM25, rank_Word2Vec

import numpy as np


def build_demo_results(corpus: dict, search_id):
    """
    Helper method, just to demo the app
    :return: a list of demo docs sorted by ranking
    """
    res = []
    size = len(corpus)
    ll = list(corpus.values())
    for index in range(random.randint(0, 40)):
        item: Tweet = ll[random.randint(0, size)]
        res.append(ResultItem(item.id, item.title, item.description, item.doc_date,
                              "doc_details?id={}&search_id={}&param2=2".format(item.id, search_id), random.random()))

    # for index, item in enumerate(corpus['Id']):
    #     # DF columns: 'Id' 'Tweet' 'Username' 'Date' 'Hashtags' 'Likes' 'Retweets' 'Url' 'Language'
    #     res.append(DocumentInfo(item.Id, item.Tweet, item.Tweet, item.Date,
    #                             "doc_details?id={}&search_id={}&param2=2".format(item.Id, search_id), random.random()))

    # simulate sort by ranking
    res.sort(key=lambda doc: doc.ranking, reverse=True)
    return res


class SearchEngine:
    """educational search engine"""
    def __init__(self,corpus,doc_to_tweet):
        self.doc_to_tweet = doc_to_tweet
        self.corpus = corpus
        index, tf, df, idf = create_index_tfidf(doc_to_tweet, tweets=corpus, num_documents=len(doc_to_tweet))
        self.index = index
        self.tf = tf
        self.df = df
        self.idf = idf
        
        

    def search(self, search_query, search_id, corpus, search_algorithm):
        print("Search query:", search_query)

        search_query = preprocessing(search_query) #preprocessing

        docs = set()
        for term in search_query:
            try:
                # store in term_docs the ids of the docs that contain "term"                        
                term_docs = [posting[0] for posting in self.index[term]]
                
                # docs = docs Union term_docs
                docs |= set(term_docs)
            except:
                #term is not in index
                pass

        docs = list(docs)

        if search_algorithm == 'TF-IDF':
            ranked_docs, rank_scores = rank_documents(search_query, docs, self.index, self.idf, self.tf)
            
        elif search_algorithm == 'BM25':
            ranked_docs, rank_scores = rankTweetsBM25(search_query, corpus, self.idf, self.doc_to_tweet, len(corpus), k1=1.2, b=0.75)
        elif search_algorithm == 'MAGIC ALGORITHM':
            pass;
        elif search_algorithm == 'WORD2VEC':
            ranked_docs, rank_scores = rank_Word2Vec(search_query, docs, self.doc_to_tweet, corpus)
        
        results = []

        for n in range(0, len(ranked_docs)):
            d_id = ranked_docs[n]
            t_id = self.doc_to_tweet[d_id]
            tweet = self.corpus[t_id]
            rank_score = np.round(rank_scores[n], 2)
            results.append(ResultItem(t_id, tweet["text"],tweet["full_text"], tweet["date"], tweet["likes"], tweet["retweets"], tweet["url"], tweet["hashtags"], tweet["username"], tweet["followers_count"], rank_score))
        

        return results