# IRWA-2022-u172939-u173478-u173384

First, we should put the path to the inputs directory in the function os.chdir('.../inputs'). This way we change the directory we are in, and we won't need to write the whole path every time we need to access an input file

All necessary imports are in the 'Import libraries' section

## Functions

- clean(text: string): string
- build_terms(text: string): list of strings
- create_mapping(filename, key, value, verbose=True): dictionary. The filename must reffer to a csv

The rest of the code should execute correctly as everything that needed to be defined has been defined above

## Output

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
