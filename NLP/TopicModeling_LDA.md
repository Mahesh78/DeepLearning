### Topic Modeling; Latent Dirichlet Allocation


```python
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk, re
from nltk.stem.wordnet import WordNetLemmatizer
```

### Load the data


```python
json_data = open('anly610Deduped.json').readlines()
file_feeds = []
for line in json_data:
    file_feeds.append(json.loads(line))

print(len(file_feeds))
```

    40
    


```python
stopwords = set(nltk.corpus.stopwords.words('english'))

feed_titles = []

for feed in file_feeds:
    feed_titles.append(str(feed['title']))

print("Total number of titles: " + str(len(feed_titles)))

```

    Total number of titles: 40
    


```python
def tokenize_titles(title):
    tokens = nltk.word_tokenize(title)
    lmtzr = WordNetLemmatizer()
    filtered_tokens = []
    
    for token in tokens:
        token = token.replace("'s", " ").replace("n’t", " not").replace("’ve", " have")
        token = re.sub(r'[^a-zA-Z0-9 ]', '', token)
        if token not in stopwords:
            filtered_tokens.append(token.lower())
    
    lemmas = [lmtzr.lemmatize(t,'v') for t in filtered_tokens]

    return lemmas
```


```python
def clstr_lda(num_topics, stories):
    # top words to be identified
    n_top_words = 10

    tf_vectorizer = CountVectorizer(max_df=0.8, min_df=0.01, max_features=500,
                                    tokenizer=tokenize_titles, ngram_range=(3,4))

    tf = tf_vectorizer.fit_transform(stories)

    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=200,
                                    learning_method='batch', learning_offset=10.,
                                    random_state = 1)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()


    topics = dict()
    for topic_idx, topic in enumerate(lda.components_):
        topics[topic_idx] = [tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print("Topic #%d:" % topic_idx)
        print(" | ".join([tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
    return topics
```


```python
n = input('Enter a positive value for n: ')
```

    Enter a value for n: 10
    


```python
topics = clstr_lda(int(n), feed_titles)
```

    Topic #0:
     subscription cost  | subscription cost  tv | tv show film | stream  launch | date  subscription cost | cost  tv show | tv show film feature | film feature  |  tv show | feature  independent
    Topic #1:
    sever tie amazon | fedex sever tie amazon | fedex sever tie | mens charge cotton | 20 show sock | 20 show sock  | charge cotton 20 | charge cotton 20 show | cotton 20 show | cotton 20 show sock
    Topic #2:
    tax dodge create recordsetting | create recordsetting  48b | dodge create recordsetting | dodge create recordsetting  |  report find | 48b luxury home | sales  report find | sales  report | home sales  | home sales  report
    Topic #3:
    fossil men  |  select fossil | 35  select |  gen 4 explorist |  gen 4 | 35  select fossil | 4 explorist hr | 4 explorist hr smartwatches | explorist hr smartwatches | slice 35  select
    Topic #4:
    touch control like airpods | airpods   | earbuds touch control like | control like airpods |  back  29 |  back  | touch control like | true wireless earbuds touch | true wireless earbuds | back  29
    Topic #5:
    woman  bloodlines |  exclusive official trailer |  bloodlines  |  bloodlines  exclusive | bloodlines  exclusive official | wonder woman  bloodlines | wonder woman  | woman  bloodlines  | bloodlines  exclusive |  exclusive official
    Topic #6:
    taste black mirror | call alexa  | son accidentally call alexa | son accidentally call | mirror son accidentally call | mirror son accidentally | charlie brooker get | taste black mirror son | charlie brooker get taste | get taste black mirror
    Topic #7:
    cast  plot  | carnival row uk release | plot  trailer | plot  trailer orlando | trailer orlando bloom  | trailer orlando bloom |  plot  |  plot  trailer | release date  | row uk release
    Topic #8:
    customers  free  | 5000   5 | 5 years less | turn  1000  | turn  1000 | 2 amazon gift | 5000   | stock turn  | stock turn  1000 | 1000  5000 
    Topic #9:
    compute  euc  | privo achieve end user | achieve end user | achieve end user compute | compute  euc | end user compute | end user compute  | euc  competency | euc  competency status |  euc  competency
    
