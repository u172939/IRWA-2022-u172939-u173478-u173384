import json


class Tweet:
    """
    Original corpus data as an object
    """

    def __init__(self, id, doc_id, text, full_text, doc_date, likes, retweets, url, hashtags,username, followers):
        self.id = id
        self.text = text
        self.doc_id  = doc_id
        self.full_text = full_text
        self.doc_date = doc_date
        self.likes = likes
        self.retweets = retweets
        self.url = url
        self.hashtags = hashtags
        self.username = username
        self.followers = followers

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)


class StatsDocument:
    """
    Original corpus data as an object
    """

    def __init__(self, id, title, description, doc_date, url, count):
        self.id = id
        self.title = title
        self.description = description
        self.doc_date = doc_date
        self.url = url
        self.count = count

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)


class ResultItem:
    def __init__(self, id, text, full_text, doc_date, likes, retweets, url, hashtags, username, followers, ranking):
        
        self.id = id
        self.text = text
        self.full_text = full_text
        self.likes = likes
        self.retweets = retweets
        self.doc_date = doc_date
        self.hashtags = hashtags
        self.followers = followers
        self.username = username
        self.url = url
        self.ranking = ranking
