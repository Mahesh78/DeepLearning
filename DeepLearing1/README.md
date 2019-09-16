The folder contains series of notebooks for __*Deep Leaning*__ that start with crawling website for a topic of interest and perform sentiment analysis.

  1) __[WebhoseCrawl](https://github.com/Mahesh78/DeepLearning/blob/master/DeepLearing1/1_WebhoseCrawl.md)__: Crawls the website for a custom topic using __webhoseio__ for a total of 10,000 feeds and stores them in a JSON file for further analysis. The notebook prints out Titles, Text and URLs of first 10 posts from the JSON.
  
  2) __[NLP_Similarities_Word2Vec_BERT](https://github.com/Mahesh78/DeepLearning/blob/master/DeepLearing1/2_NLP_Similarities_Word2Vec_BERT.md)__: Using [Word2Vec](http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/) from [GoogleNews-vectors-negative300.bin.gz](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) the notebook selects a title from the previously created JSON and calculates the __pairwise similarity__ and appends the similarity score to the remaining titles. 100 titles with highest similarity score will be returned in the descending order of their score.
  
  3) __[NLP_Deduplicate_SimHash_Word2Vec](https://github.com/Mahesh78/DeepLearning/blob/master/DeepLearing1/3_NLP_Deduplicate_SimHash_Word2Vec.md)__: This notebook deduplicates the feeds from the JSON file created earlier using both __SimHash__ and __Word2Vec__ to return deduped titles with greater accuracy.
  
  4) __[SentimentAnalysis_NLTK_NaiveBayes](https://github.com/Mahesh78/DeepLearning/blob/master/DeepLearing1/4_SentimentAnalysis_NLTK_NaiveBayes.md)__: Uses __NLTK Sentiment Intensity Analyzer__ to assess the sentiment of each of the 10,000 titles and appends the sentiment scores to the previously created JSON.
