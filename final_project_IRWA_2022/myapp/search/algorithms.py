import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import collections
from collections import defaultdict
import numpy as np
from numpy import linalg as la
import math
from array import array
from gensim.models.word2vec import Word2Vec

def preprocessing(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    cleanText = text.lower() # transform to lowercase
    cleanText = re.sub('https?:\/\/.*[\r\n]*', '', cleanText, flags=re.MULTILINE) # removing the urls from tweeet, starts with https
    # removing nonalphanumeric
    cleanText = re.sub(r'[\W]+', ' ', cleanText)
    cleanText = re.sub(r'[\_]+', '', cleanText)


    text =  cleanText.split() # Tokenize the text to get a list of terms
    text = [word for word in text if word not in stop_words]   # Eliminate the stopwords 
    text = [stemmer.stem(word) for word in text] # Stemming (keeping the root of the words)

    return text


def create_index_tfidf(doc_to_tweet, tweets, num_documents):
    """
    Implement the inverted index and compute tf, df and idf
    
    Arguments:
    doc_to_tweet -- dictionary to pass from doc_id to tweet_it
    tweets -- collection of tweets
    num_documents -- total number of tweets
    
    Returns:
    index - the inverted index as a Python dictionary: {term: list of documents and positions}
    tf - normalized term frequency for each term in each document
    df - number of documents each term appear in
    idf - inverse document frequency of each term
    """

    index = defaultdict(list)
    tf = defaultdict(list)  # term frequencies of terms in documents (documents in the same order as in the main index)
    df = defaultdict(int)  # document frequencies of terms in the collection
    idf = defaultdict(float)

    print("Creating tf-idf index...")
    # Remember, tweets contain {tweet_id: {tweet info}}

    for doc_id, tweet_id in doc_to_tweet.items():  

        terms = tweets[tweet_id]['text']

        current_doc_index = {}

        for position, term in enumerate(terms): # iterate through tweet terms
            try:
                current_doc_index[term][1].append(position)  
            except:
                current_doc_index[term] = [doc_id, array('I',[position])]

        # Compute normalized term frequencies and df weights
        norm = 0

        for term, posting in current_doc_index.items(): # posting ==> [current_doc, [list of positions]] for each term
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        for term, posting in current_doc_index.items():
            tf[term].append(np.round(len(posting[1])/norm,4)) # append the tf for current term
            df[term] += 1 #increment the document frequency of current term

        # Merge the current page index with the main index
        for term_doc, posting in current_doc_index.items():
            index[term_doc].append(posting)

        # Compute IDF 
        for term in df:
            idf[term] = np.round(np.log(float(num_documents/df[term])), 4)

    return index, tf, df, idf


def rank_documents(query, tweets, index, idf, tf):
    """
    Perform the ranking of the results of a search based on the tf-idf weights
    
    Argument:
    query -- list of query terms
    tweets -- list of tweets, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies
    
    Returns:
    List of ranked documents, List of corresponding scores
    """

    # doc_vectors[k] = (k,[0]*len(terms))
    doc_vectors = defaultdict(lambda: [0] * len(query)) 
    query_vector = [0] * len(query)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(query)  # term frequency for the query terms
    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(query):  #t ermIndex is the index of the term in the query
        if term not in index:
            # do not have into account terms that do not appear in query
            continue 

        # Compute tf*idf (normalize TF as done with documents)
        query_vector[termIndex]= query_terms_count[term]/query_norm * idf[term] 

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):
         
            if doc in tweets:
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term] 

    
    # Compute the score of each doc (cosine similarity with query vector)
    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
    doc_scores.sort(reverse=True) #sort by descending order
    
    result_docs = [x[1] for x in doc_scores] # take doc_id
    result_scores = [x[0] for x in doc_scores] # take score

    return result_docs, result_scores

def rankTweetsBM25(query, tweets, idf, doc_to_tweet, N, k1=1.2, b=0.75):
    """
    Perform the ranking of the results of a search based on BM25
    
    Argument:
    query -- list of query terms (already preprocessed)
    tweets -- diccionary of tweets
    idf -- inverted document frequencies (diccionary)
    k1 -- constant k1
    b -- constant b
    N -- total number of documents in the collection
    
    Returns:
    resultScores --  List of ranked scores of tweets
    resultTweets --  List of ranked tweet ids
    """

    # Lave computation
    Lave = np.mean([len(tweet['text']) for id, tweet in tweets.items()])

    RSV =  dict()

    for tweet_id in tweets.keys():  

        doc_id = [i for i in doc_to_tweet if doc_to_tweet[i]==tweet_id][0]

        terms = tweets[tweet_id]['text'] # terms of the tweets

        Ld = len(terms) # document length
        
        # get tweet terms that are in the query
        query_terms = [t for t in terms if t in query] 

        # get raw term frequency of each query term appearing in the tweet
        raw_tf = collections.Counter(query_terms) 

        # Compute RSV score for the document by iterating through tweet terms IN the query
        RSV[doc_id] = 0
        for term in query_terms:
         
          numerador = (k1 + 1) * raw_tf[term]

          denominador = k1*((1-b) + b*(Ld/Lave)) + raw_tf[term]
  
          RSV[doc_id] += np.round(idf[term] * (numerador / denominador), 4)

    # Compute the score of each doc (cosine similarity with query vector)
    doc_scores = [[score, doc] for doc, score in RSV.items()]
    doc_scores.sort(reverse=True) #sort by descending order
    
    result_docs = [x[1] for x in doc_scores] # take doc_id
    result_scores = [x[0] for x in doc_scores] # take score      

    return result_docs, result_scores


"""
def rankTweetsOurs(query, docs, index, idf, tf, tweets, doc_to_tweet):
    search_query = preprocessing(query) #preprocessing
    ranked_docs_full, rank_scores_full = rank_documents(search_query, docs, index, idf, tf)

    likes = []
    retweets = []
    hashtags = []

    ranked_docs = []
    ranked_scores = []
    return_list = []

    for tweet in tweets.keys():
      doc_id = [i for i in doc_to_tweet if doc_to_tweet[i]==tweet][0]
      ranked_docs.append(doc_id)
      try:
        idx = ranked_docs_full.index(doc_id)
        ranked_scores.append(ranked_scores_full[idx])
        return_list.append(return_list_full[idx])
      except:
        ranked_scores.append(0)
        return_list.append("")

    #we create 3 lists with the likes, retweets and hashtags for each tweet
    for doc in ranked_docs:
      likes.append(tweets[doc_to_tweet[doc]]["likes"])
      retweets.append(tweets[doc_to_tweet[doc]]["retweets"])
      hashtags.append(len(list(set(tweets[doc_to_tweet[doc]]["hashtags"]).intersection(set(terms.split(" ")))))*5)#we give a value of 5 if the tweet has hashtags that are in the query too

    # we compute our own score by assigning a weigth to each one of the previous 3 lists
    list_of_scores = []
    for i in range(0,len(likes)):
      list_of_scores.append(likes[i]*0.3 + retweets[i]*0.4 + hashtags[i]*0.3)

    list_of_scores_full = list_of_scores

    #we map the list of scores to 0 and 5, assigning a 0 to the lower value of the list and a 5 to the higher
    list_of_scores = np.array(list_of_scores).reshape(-1,1)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 5))
    list_of_scores=list(scaler.fit_transform(list_of_scores))

    list_of_scores_norma = [float(i) for i in list_of_scores]

    list_of_scores = list_of_scores + ranked_scores


    #we sort all the 
    sorted_docs = [x for _, x in sorted(zip(list_of_scores, ranked_docs),reverse = True)]
    sorted_list = [x for _, x in sorted(zip(list_of_scores, return_list),reverse = True)]
    sorted_list_of_scores = [float(i) for i in sorted(list_of_scores,reverse=True)]

    

    #return rank punctuation and ids
    return sorted_docs, sorted_list_of_scores, sorted_list
"""

def embedding_w2v(terms, wv):
  ''' Generate the representation of a tweet as a unique vector of the same dimension of the words
  Average of vectors representing the words included in the tweet

  Arguments:
  terms -- terms of the tweet
  wv -- dictionary for {word: vector representation}

  Return:
  Embedding of the tweet (vector)
  '''
  embeddings = []

  for term in terms:
    if term in list(wv.index_to_key):
      embeddings.append(wv.word_vec(term))
    
  return np.mean(embeddings, axis = 0)


def rank_Word2Vec(query, docs, doc_to_tweet, tweets):
  ''' Generate the ranking of the tweets according to a query using word2vec algorithm

  Argument:
  query -- query terms
  docs -- documents to rank
  doc_to_tweet -- dictionary to pass from doc_id to tweet_it
  tweets -- collection of tweets

  Return:
  List of ranked documents, List of corresponding scores
  '''

  # store all tweets in an iterable
  sentences = []
  for doc_id in docs:
    sentences.append(tweets[doc_to_tweet[doc_id]]['text'])

  # train model so that it learns the embeddings and the vocabulary
  model = Word2Vec(sentences)
  word_vectors = model.wv

  # create tweet2vec representation for all tweets
  docs2vec = dict()

  for doc_id in docs:
    tweet_id = doc_to_tweet[doc_id]
    docs2vec[doc_id] = embedding_w2v(tweets[tweet_id]['text'], wv=word_vectors)

  # get the query representation
  query_vec = embedding_w2v(query, wv=word_vectors)

  # compute cosine similarity between all documents and the query
  doc_scores = [[np.dot(curDocVec, query_vec), doc] for doc, curDocVec in docs2vec.items()]
  doc_scores.sort(reverse=True) #sort by descending order

  result_docs = [x[1] for x in doc_scores] # take doc_id
  result_scores = [x[0] for x in doc_scores] # take score

  return result_docs, result_scores