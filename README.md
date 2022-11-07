# IRWA-2022-u172939-u173478-u173384

First, we should put the path to the inputs directory in the function os.chdir('.../inputs'). This way we change the directory we are in, and we won't need to write the whole path every time we need to access an input file

All necessary imports are in the 'Import libraries' section

## Functions

- clean(text: string): string
- build_terms(text: string): list of strings
- create_mapping(filename, key, value, verbose=True): dictionary. The filename must reffer to a csv
- create_index_tfidf(doc_to_tweet, tweets, num_documents): 4 dictionaries (index, tf, df, idf)
- rank_documents(query, tweets, index, idf, tf): 2 dictionaries (ranked docs, ranked scores)
- get_hashtags(tweet): string
- tweet_Searcher(tweets, id): string
- search_tf_idf(query, index, idf, tf, doc_to_tweet): 3 lists
- precision_at_k(doc_score, y_score, k=10): float
- avg_precision_at_k(doc_score, y_score, k=10): float
- map_at_k(search_res, k=10): float
- mrr_at_k(search_res, k=10): float
- dcg_at_k(y_true, y_score,  k=10): float
- ndcg_at_k(y_true, y_score, k=10): float


The rest of the code should execute correctly as everything that needed to be defined has been defined above

## Output part 1

The output of the last part of the code is a dictionary where the keys are all the ids of the tweets and, for each key, the value is another dictionary with the data of the tweet.
An example of a key-value pair from the output dictionary would be the following:

{1575918140839673873:
{‘text': ['kissimme', 'neighborhood', 'michigan', 'ave', 'hurricaneian'],
 'username': 'CHeathWFTV',
 'date': '30/09/2022 18:38:58',
 'hashtags': ['HurricaneIan'],
 'likes': 0,
 'retweets': 0,
 'url': 'https://twitter.com/CHeathWFTV/status/1575918140839673873'}
}

## Output part 2
The output of this second part of the project creates an indexing for the preprocessed documents. Then, creates a ranking of documents depending on a query. In the end, we evaluate the search system we built.
