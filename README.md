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

- embedding_w2v(terms, wv: vector of floats
- rank_Word2Vec(query, docs, doc_to_tweet, tweets): 2 lists (ranked docs, ranked scores)
- search_word2vec(query, index, doc_to_tweet, tweets): 2 lists (ranked docs, ranked scores)
- rankTweetsOurs(terms, index, idf, tf, tweets,doc_to_tweet): 2 lists (ranked docs, ranked scores)
- rankTweetsBM25(query, tweets, idf, doc_to_tweet, N, k1=1.2, b=0.75): 2 lists (ranked docs, ranked scores)


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

## Output part 3
The output of this third part is an evaluation of the ranking methods we have built. First of all we have the evaluation of our custom ranking method and the bm25 one. At the end, we have the results for the same queries of the last method: word2vector + cosine similarity.
Fist, we need to create a dataset with our queries in order to iterate through it. We have created a dataset with a baseline with our 5 queries and the ground truth files for each query. This file is in this repository stored with the name queries_df_part3.csv, it is read in the notebook in order to do the  ranking and the evaluation.

## FINAL PROJECT EXECUTION

First of all it is important to have a conda environment in order to execute the main program locally. This should be created and update as told in the README that is inside the final_project_IRWA directory.
- If you are executing it through Visual Studio Code you just have to open the web_app.py file, select the environment and run.
- If you are using the terminal you have to go to the folder which contains the web_app.py file. Then, active the conda environment and run the program with the command: python web_app.py
After all this, the app will be initiallized, you should wait a few minutes. Then some messages will appear in the terminal, copy the link that appears after Running on and copy it into an Internet searcher. The page will load and you will be free to use it then. To exit, write Control+C in the terminal.
All necessary documents are in the folder.
