

```python
import nltk
import json
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize, sent_tokenize, ngrams
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions
```

### Sentiment Analysis using NLTK Sentiment Intensity Analyzer


```python
hotel_review = ["Great resort to stay in when you visit the Dominican Republic.",
                "Rooms were under renovation when I visited, so the availability was limited.",
                "Love the ocean breeze and the food.",
                "The food is delicious but not over the top.",
                "Service - Little slow, probably because of too many people.",
                "The place is not easy to find.",
                "Prawns cooked in local specialty sauce were tasty."]
```


```python
nltk.download('vader_lexicon')
sent_analyzer = SentimentIntensityAnalyzer()
for sentence in hotel_review:
    print(sentence)
    scores = sent_analyzer.polarity_scores(sentence)
    for k in scores:
        print('{0}: {1}, '.format(k, scores[k]), end='')
    print()
```

    Great resort to stay in when you visit the Dominican Republic.
    neg: 0.0, neu: 0.709, pos: 0.291, compound: 0.6249, 
    Rooms were under renovation when I visited, so the availability was limited.
    neg: 0.16, neu: 0.84, pos: 0.0, compound: -0.2263, 
    Love the ocean breeze and the food.
    neg: 0.0, neu: 0.588, pos: 0.412, compound: 0.6369, 
    The food is delicious but not over the top.
    neg: 0.168, neu: 0.623, pos: 0.209, compound: 0.1184, 
    Service - Little slow, probably because of too many people.
    neg: 0.0, neu: 1.0, pos: 0.0, compound: 0.0, 
    The place is not easy to find.
    neg: 0.286, neu: 0.714, pos: 0.0, compound: -0.3412, 
    Prawns cooked in local specialty sauce were tasty.
    neg: 0.0, neu: 1.0, pos: 0.0, compound: 0.0, 
    

    [nltk_data] Downloading package vader_lexicon to
    [nltk_data]     C:\Users\mitikirim\AppData\Roaming\nltk_data...
    [nltk_data]   Package vader_lexicon is already up-to-date!
    


```python
def normalize(score, alpha=15):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score/math.sqrt((score*score) + alpha)
    return norm_score
```

### Sentiment Analysis using Naive Bayes Classifier (training and validation)


```python
from nltk.tokenize import word_tokenize
training_set = [("Great resort to stay in when you visit the Dominican Republic.","pos"),
                ("Rooms were under renovation when I visited, so the availability was limited.","neg"),
                ("Love the ocean breeze and the food.","pos"),
                ("The food is delicious but not over the top.","neg"),
                ("Service - Little slow, probably because of too many people.","neg"),
                ("The place is not easy to find.","neg"),
                ("Prawns cooked in a local specialty sauce were tasty.", "pos")]

  
# Step 2 
dictionary = set(word.lower() for passage in training_set for word in word_tokenize(passage[0]))
  
# Step 3
t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in training_set]
  
# Step 4 – the classifier is trained with sample data
classifier = nltk.NaiveBayesClassifier.train(t)
  
test_data = "Hotel was great and food service is slow"
test_data_features = {word.lower(): (word in word_tokenize(test_data.lower())) for word in dictionary}
  
print (classifier.classify(test_data_features))
```

    neg
    


```python
test_data = "Our dinner meal was hot and spicy"
test_data_features = {word.lower(): (word in word_tokenize(test_data.lower())) for word in dictionary}
  
print (classifier.classify(test_data_features))
```

    pos
    

### Subjectivity analysis using NLTK


```python
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.classify import NaiveBayesClassifier
```

#### Obtain a set of 100 subjective and 100 objective sentences from NLTK subjectivity corpus


```python
n_instances = 100
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
len(subj_docs), len(obj_docs)
```

    [nltk_data] Downloading package subjectivity to
    [nltk_data]     C:\Users\mitikirim\AppData\Roaming\nltk_data...
    [nltk_data]   Package subjectivity is already up-to-date!
    




    (100, 100)




```python
subj_docs[0] # each input consist of sentence represented as a list of strings, and a label (subj or obj)
```




    (['smart',
      'and',
      'alert',
      ',',
      'thirteen',
      'conversations',
      'about',
      'one',
      'thing',
      'is',
      'a',
      'small',
      'gem',
      '.'],
     'subj')



#### Split into training and testing sets


```python
train_subj_docs = subj_docs[:80]
test_subj_docs = subj_docs[80:100]
train_obj_docs = obj_docs[:80]
test_obj_docs = obj_docs[80:100]
training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs
```


```python
sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
```

#### Use simple unigram word features, handling negation:


```python
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
len(unigram_feats)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
```

#### Apply features to obtain a feature-value representation of the datasets


```python
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)
```

#### Train classifier and output evaluation results


```python
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))
```

    Training classifier
    Evaluating NaiveBayesClassifier results...
    Accuracy: 0.8
    F-measure [obj]: 0.8
    F-measure [subj]: 0.8
    Precision [obj]: 0.8
    Precision [subj]: 0.8
    Recall [obj]: 0.8
    Recall [subj]: 0.8
    

#### Get the JSON dataset


```python
json_data = open('anly610.json').readlines()
file_feeds = []
for line in json_data:
    file_feeds.append(json.loads(line))

print(len(file_feeds))
```

    10000
    


```python
feeds_title = []
for i in range(len(file_feeds)):
    feeds_title.append(file_feeds[i]['title'])
```


```python
len(feeds_title)
```




    10000



### 1) Calculate sentiment for every article title in your deduplicated Webhose dataset using NLTK Sentiment Intensity Analyzer


```python
sent_analyzer = SentimentIntensityAnalyzer()
compound_scores = [] # Get scores
for title in feeds_title:
    compound_scores.append(sent_analyzer.polarity_scores(title))
```


```python
compound_scores[1]
```




    {'neg': 0.198, 'neu': 0.802, 'pos': 0.0, 'compound': -0.4939}



### 2) Append compound sentiment score to JSON of every article and save it back into the file


```python
feeds_compound_scores = file_feeds
for i in range(len(compound_scores)):
    feeds_compound_scores[i].update(compound_scores[i])
```


```python
feeds_compound_scores[:2]
```




    [{'thread': {'uuid': 'c395929128774c187c6087a91263c75eb75a854a',
       'url': 'https://www.independent.co.uk/arts-entertainment/tv/news/disney-plus-streaming-service-launch-date-cost-how-much-tv-shows-films-when-hulu-a9045506.html',
       'site_full': 'www.independent.co.uk',
       'site': 'independent.co.uk',
       'site_section': 'http://www.independent.co.uk/arts-entertainment/tv/rss',
       'site_categories': ['media', 'law_government_and_politics', 'politics'],
       'section_title': 'The Independent - TV & Radio',
       'title': 'Disney+ streaming: Launch date, subscription cost, which TV shows and films feature | The Independent',
       'title_full': 'Disney+ streaming: Launch date, subscription cost, which TV shows and films feature | The Independent',
       'published': '2019-08-07T23:26:00.000+03:00',
       'replies_count': 0,
       'participants_count': 1,
       'site_type': 'news',
       'country': 'GB',
       'spam_score': 0.0,
       'main_image': 'https://static.independent.co.uk/s3fs-public/thumbnails/image/2019/08/07/15/gettyimages-1134775154.jpg',
       'performance_score': 0,
       'domain_rank': 568,
       'social': {'facebook': {'likes': 0, 'comments': 0, 'shares': 0},
        'gplus': {'shares': 0},
        'pinterest': {'shares': 0},
        'linkedin': {'shares': 0},
        'stumbledupon': {'shares': 0},
        'vk': {'shares': 0}}},
      'uuid': 'c395929128774c187c6087a91263c75eb75a854a',
      'url': 'https://www.independent.co.uk/arts-entertainment/tv/news/disney-plus-streaming-service-launch-date-cost-how-much-tv-shows-films-when-hulu-a9045506.html',
      'ord_in_thread': 0,
      'author': 'independent.co.uk',
      'published': '2019-08-07T23:26:00.000+03:00',
      'title': 'Disney+ streaming: Launch date, subscription cost, which TV shows and films feature | The Independent',
      'text': 'Disney+ is a brand new streaming service arriving in 2019 to rival the likes of Netflix and Amazon Prime .It will be a comprehensive streaming platform featuring all the TV shows and films produced by Disney since 1937, as well as new exclusive content.Disney+ will host five hubs dedicated to the major franchises owned by media company: Disney, Pixar, Marvel, Star Wars and National Geographic.From extras.Here’s everything we know about Disney+ so far…Disney+ will debut in the US on 12 November 2019 with a variety of classic Disney content as well as some of its new exclusives.While there is no UK release date yet, the service is set to launch in every major region around the world within the next two years. Disney+ will cost $6.99 per month in the US, or $69.99 per year – making it less expensive than Netflix.',
      'highlightText': '',
      'highlightTitle': '',
      'language': 'english',
      'external_links': [],
      'external_images': None,
      'entities': {'persons': [],
       'organizations': [{'name': 'amazon', 'sentiment': 'negative'},
        {'name': 'disney', 'sentiment': 'negative'},
        {'name': 'netflix', 'sentiment': 'negative'},
        {'name': 'star wars', 'sentiment': 'none'},
        {'name': 'pixar', 'sentiment': 'none'}],
       'locations': [{'name': 'uk', 'sentiment': 'none'},
        {'name': 'us', 'sentiment': 'none'}]},
      'rating': None,
      'crawled': '2019-08-07T17:45:03.007+03:00',
      'neg': 0.0,
      'neu': 1.0,
      'pos': 0.0,
      'compound': 0.0},
     {'thread': {'uuid': '8d3e02afcbed15b516348303f19b1db681db344c',
       'url': 'https://www.theguardian.com/world/2019/aug/07/bolsonaro-amazon-deforestation-exploded-july-data',
       'site_full': 'www.theguardian.com',
       'site': 'theguardian.com',
       'site_section': 'http://www.theguardian.com/world/brazil/rss',
       'site_categories': ['media'],
       'section_title': 'Brazil | The Guardian',
       'title': "Bolsonaro rejects 'Captain Chainsaw' label as data shows deforestation 'exploded' | World news | The Guardian",
       'title_full': "Bolsonaro rejects 'Captain Chainsaw' label as data shows deforestation 'exploded' | World news | The Guardian",
       'published': '2019-08-07T23:25:00.000+03:00',
       'replies_count': 0,
       'participants_count': 1,
       'site_type': 'news',
       'country': 'US',
       'spam_score': 0.0,
       'main_image': 'https://i.guim.co.uk/img/media/2f4ed1341bdcca1f8ef6f9693eb8ac10cd6e781d/0_12_4920_2952/master/4920.jpg?width=1200&height=630&quality=85&auto=format&fit=crop&overlay-align=bottom%2Cleft&overlay-width=100p&overlay-base64=L2ltZy9zdGF0aWMvb3ZlcmxheXMvdGctZGVmYXVsdC5wbmc&enable=upscale&s=417a99f9f98e2d2fab6f9f887ee807d2',
       'performance_score': 0,
       'domain_rank': 170,
       'social': {'facebook': {'likes': 0, 'comments': 0, 'shares': 0},
        'gplus': {'shares': 0},
        'pinterest': {'shares': 0},
        'linkedin': {'shares': 0},
        'stumbledupon': {'shares': 0},
        'vk': {'shares': 0}}},
      'uuid': '8d3e02afcbed15b516348303f19b1db681db344c',
      'url': 'https://www.theguardian.com/world/2019/aug/07/bolsonaro-amazon-deforestation-exploded-july-data',
      'ord_in_thread': 0,
      'author': 'Tom Phillips',
      'published': '2019-08-07T23:25:00.000+03:00',
      'title': "Bolsonaro rejects 'Captain Chainsaw' label as data shows deforestation 'exploded' | World news | The Guardian",
      'text': "Data says 2,254 sq km cleared in July as president says Macron and Merkel ‘haven’t realized Brazil’s under new management’. Deforestation in the Brazilian Amazon “exploded” in July it has emerged as Jair Bolsonaro scoffed at his portrayal as Brazil’s “Captain Chainsaw” and mocked Emmanuel Macron and Angela Merkel for challenging him over the devastation.\nSpeaking in São Paulo on Tuesday, Brazil’s president attacked the leaders of France and Germany – who have both voiced concern about the surge in destruction since Bolsonaro took office in January.\n“They still haven’t realized Brazil’s under new management,” Bolsonaro declared to cheers of approval from his audience. “Now we’ve got a blooming president.”\nAmazon deforestation accelerating towards unrecoverable 'tipping point' Read more\nThe far-right populist repeated claims that his administration – which critics accuse of helping unleash a new wave of environmental destruction – was the victim of a mendacious international smear campaign based on “imprecise” satellite data showing a jump in deforestation.\nBolsonaro ridiculed what he called his depiction as “ Capitão Motoserra ” (“ Captain Chainsaw”).\nBut as he spoke, official data laid bare the scale of the environmental crisis currently unfolding in the world’s biggest rainforest, of which about 60% is in Brazil .\nAccording to a report in the Estado de São Paulo newspaper , Amazon destruction “exploded” in July with an estimated 2,254 sq km (870 sq miles) of forest cleared, according to preliminary data gathered by Brazil’s National Institute for Space Research, the government agency that monitors deforestation.\nThat is an area about half the size of Philadelphia and reportedly represents a 278% rise on the 596.6 sq km destroyed in July last year.\nRômulo Batista, an Greenpeace campaigner based in the Amazon city of Manaus, said the numbers – while preliminary – were troubling and showed a clear trend of rising deforestation under Bolsonaro. What was not yet clear was if the devastation was “going up, going up a lot, or skyrocketing”.\nBatista blamed Bolsonaro’s “anti-environmental” discourse and policies – such as slashing the budget of Brazil’s environmental agency, Ibama – for the surge.\n“It’s almost as if a licence to deforest illegally and with impunity has been given, now that you have the [environmental] inspection and control teams being attacked by no less than the president of the republic and the environment minister,” Batista added. “This is a very worrying moment.”\nThe spike in destruction under Bolsonaro – who was elected with the support of powerful mining and agricultural sectors – has come as a shock to environmentalists, but not a surprise.\nDuring a visit to the Amazon last year Bolsonaro told the Guardian that as president he would target “cowardly” environmental NGOs who were “sticking their noses” into Brazil’s domestic affairs.\n“This tomfoolery stops right here!” Bolsonaro proclaimed, praising Donald Trump’s approval of the Dakota Access and Keystone XL oil pipelines.\nBolsonaro returned to that theme on Tuesday during a gathering of car dealers in Brazil’s economic capital, São Paulo, complaining that “60% of our territory is rendered unusable by indigenous reserves and other environmental questions”.\n“You can’t imagine how much I enjoyed talking to Macron and Angela Merkel [about these issues during the recent G20 in Japan],” Bolsonaro added to guffaws from the crowd. “What a pleasure!”\nIn June Merkel described the environmental situation in Bolsonaro’s Brazil as “dramatic”.\nIn recent weeks the globally respected National Institute for Space Research has found itself at the eye of a political storm as a result of the inconvenient truths revealed by its data.\nEarlier this month, with alarm growing about the consequences of the intensifying assault on the Amazon, its director, Ricardo Galvão, was sacked after contesting Bolsonaro’s “pusillanimous” claims he was peddling lies about the state of the Amazon.\nGalvão’s successor, the air force colonel Darcton Policarpo Damião, looks set to follow a more Bolsonarian line. In an interview this week Damião said he was not convinced global heating was a manmade phenomenon and called such matters “not my cup of tea” .\nPope Francis – who is preparing to host a special synod on the Amazon in October – has also incurred Bolsonaro’s wrath on the environment.\nIn June the Argentinian leader of the Catholic church questioned “the blind and destructive mentality” of those seeking to profit from the world’s biggest rainforest. “What is happening in Amazonia will have repercussions at a global level,” he warned.\nAsked about those comments, Bolsonaro offered a characteristically unvarnished response, suggesting they reflected an international conspiracy to commandeer the Amazon.\n“Brazil is the virgin that every foreign pervert wants to get their hands on,” Bolsonaro said .\nTopics Brazil Jair Bolsonaro Amazon rainforest Deforestation Americas Conservation Trees and forests news",
      'highlightText': '',
      'highlightTitle': '',
      'language': 'english',
      'external_links': ['https://www.nytimes.com/2019/07/28/world/americas/brazil-deforestation-amazon-bolsonaro.html',
       'https://oglobo.globo.com/sociedade/nao-minha-praia-diz-novo-diretor-do-inpe-sobre-aquecimento-global-23858248',
       'https://noticias.uol.com.br/meio-ambiente/ultimas-noticias/redacao/2019/08/02/demissao-do-inpe-relembre-o-bate-boca-entre-diretor-exonerado-e-bolsonaro.htm',
       'https://www.noticias.uol.com.br/meio-ambiente/ultimas-noticias/redacao/2019/08/02/demissao-do-inpe-relembre-o-bate-boca-entre-diretor-exonerado-e-bolsonaro.htm',
       'https://www.oglobo.globo.com/sociedade/nao-minha-praia-diz-novo-diretor-do-inpe-sobre-aquecimento-global-23858248',
       'https://oantagonista.com/brasil/imagina-o-prazer-que-tive-de-falar-com-macron-e-merkel-ironiza-bolsonaro/',
       'https://sustentabilidade.estadao.com.br/noticias/geral,desmatamento-explode-em-julho-e-chega-a-2254-km-um-terco-dos-ultimos-12-meses,70002957618?utm_source=twitter:newsfeed&utm_medium=social-organic&utm_campaign=redes-sociais:082019:e&utm_content=:::&utm_term=',
       'https://www.oantagonista.com/brasil/imagina-o-prazer-que-tive-de-falar-com-macron-e-merkel-ironiza-bolsonaro/',
       'https://oglobo.globo.com/brasil/bolsonaro-brasil-a-virgem-que-todo-tarado-quer-23789972?utm_source=Twitter&utm_medium=Social&utm_campaign=compartilhar',
       'https://oglobo.globo.com/brasil/bolsonaro-brasil-a-virgem-que-todo-tarado-quer-23789972',
       'http://www.w2.vatican.va/content/francesco/en/messages/pont-messages/2019/documents/papa-francesco_20190706_messaggio-comunita-laudatosi.html',
       'http://w2.vatican.va/content/francesco/en/messages/pont-messages/2019/documents/papa-francesco_20190706_messaggio-comunita-laudatosi.html',
       'https://www.oantagonista.com/brasil/imagina-o-prazer-que-tive-de-falar-com-macron-e-merkel-ironiza-bolsonaro',
       'https://www.oglobo.globo.com/brasil/bolsonaro-brasil-a-virgem-que-todo-tarado-quer-23789972?utm_source=Twitter&utm_medium=Social&utm_campaign=compartilhar',
       'https://sustentabilidade.estadao.com.br/noticias/geral,desmatamento-explode-em-julho-e-chega-a-2254-km-um-terco-dos-ultimos-12-meses,70002957618',
       'https://www.sustentabilidade.estadao.com.br/noticias/geral,desmatamento-explode-em-julho-e-chega-a-2254-km-um-terco-dos-ultimos-12-meses,70002957618?utm_source=twitter:newsfeed&utm_medium=social-organic&utm_campaign=redes-sociais:082019:e&utm_content=:::&utm_term=',
       'https://nytimes.com/2019/07/28/world/americas/brazil-deforestation-amazon-bolsonaro.html'],
      'external_images': None,
      'entities': {'persons': [{'name': 'chainsaw', 'sentiment': 'negative'},
        {'name': 'macron', 'sentiment': 'negative'},
        {'name': 'bolsonaro', 'sentiment': 'negative'},
        {'name': 'merkel', 'sentiment': 'negative'},
        {'name': 'angela merkel', 'sentiment': 'none'},
        {'name': 'jair bolsonaro', 'sentiment': 'none'},
        {'name': 'emmanuel macron', 'sentiment': 'none'}],
       'organizations': [{'name': 'guardian data', 'sentiment': 'negative'},
        {'name': 'amazon', 'sentiment': 'none'}],
       'locations': [{'name': 'brazil', 'sentiment': 'none'},
        {'name': 'são paulo', 'sentiment': 'none'},
        {'name': 'france', 'sentiment': 'none'},
        {'name': 'germany', 'sentiment': 'none'}]},
      'rating': None,
      'crawled': '2019-08-07T18:51:51.007+03:00',
      'neg': 0.198,
      'neu': 0.802,
      'pos': 0.0,
      'compound': -0.4939}]




```python
len(feeds_compound_scores)
```




    10000



#### Store in a json file


```python
with open('anly610_scores.json', 'w') as myfile:
    for feed in feeds_compound_scores:
        line = json.dumps(feed)
        myfile.write(line)
        myfile.write('\n')
```

#### Open the json file


```python
json_data = open('anly610_scores.json').readlines()
file_scores = []
for line in json_data:
    file_scores.append(json.loads(line))

print(len(file_scores))
```

    10000
    

### 3) Output polarity scores for each sentence in any one article from your dataset

   I picked 2nd article from the feeds to get polarity scores since the size is very large compared to most of the other articles.


```python
sent_analyzer = SentimentIntensityAnalyzer()
for sentence in sent_tokenize(file_scores[1]['text']):
    print(sentence)
    scores = sent_analyzer.polarity_scores(sentence)
    for k in scores:
        print('{0}: {1}, '.format(k, scores[k]), end='')
    print('\n')
```

    Data says 2,254 sq km cleared in July as president says Macron and Merkel ‘haven’t realized Brazil’s under new management’.
    neg: 0.0, neu: 0.931, pos: 0.069, compound: 0.1027, 
    
    Deforestation in the Brazilian Amazon “exploded” in July it has emerged as Jair Bolsonaro scoffed at his portrayal as Brazil’s “Captain Chainsaw” and mocked Emmanuel Macron and Angela Merkel for challenging him over the devastation.
    neg: 0.129, neu: 0.787, pos: 0.084, compound: -0.4215, 
    
    Speaking in São Paulo on Tuesday, Brazil’s president attacked the leaders of France and Germany – who have both voiced concern about the surge in destruction since Bolsonaro took office in January.
    neg: 0.188, neu: 0.812, pos: 0.0, compound: -0.7717, 
    
    “They still haven’t realized Brazil’s under new management,” Bolsonaro declared to cheers of approval from his audience.
    neg: 0.0, neu: 0.708, pos: 0.292, compound: 0.7351, 
    
    “Now we’ve got a blooming president.”
    Amazon deforestation accelerating towards unrecoverable 'tipping point' Read more
    The far-right populist repeated claims that his administration – which critics accuse of helping unleash a new wave of environmental destruction – was the victim of a mendacious international smear campaign based on “imprecise” satellite data showing a jump in deforestation.
    neg: 0.208, neu: 0.726, pos: 0.066, compound: -0.8126, 
    
    Bolsonaro ridiculed what he called his depiction as “ Capitão Motoserra ” (“ Captain Chainsaw”).
    neg: 0.172, neu: 0.828, pos: 0.0, compound: -0.3612, 
    
    But as he spoke, official data laid bare the scale of the environmental crisis currently unfolding in the world’s biggest rainforest, of which about 60% is in Brazil .
    neg: 0.132, neu: 0.868, pos: 0.0, compound: -0.6249, 
    
    According to a report in the Estado de São Paulo newspaper , Amazon destruction “exploded” in July with an estimated 2,254 sq km (870 sq miles) of forest cleared, according to preliminary data gathered by Brazil’s National Institute for Space Research, the government agency that monitors deforestation.
    neg: 0.076, neu: 0.861, pos: 0.064, compound: -0.3818, 
    
    That is an area about half the size of Philadelphia and reportedly represents a 278% rise on the 596.6 sq km destroyed in July last year.
    neg: 0.118, neu: 0.882, pos: 0.0, compound: -0.4939, 
    
    Rômulo Batista, an Greenpeace campaigner based in the Amazon city of Manaus, said the numbers – while preliminary – were troubling and showed a clear trend of rising deforestation under Bolsonaro.
    neg: 0.107, neu: 0.762, pos: 0.131, compound: -0.0516, 
    
    What was not yet clear was if the devastation was “going up, going up a lot, or skyrocketing”.
    neg: 0.249, neu: 0.751, pos: 0.0, compound: -0.6103, 
    
    Batista blamed Bolsonaro’s “anti-environmental” discourse and policies – such as slashing the budget of Brazil’s environmental agency, Ibama – for the surge.
    neg: 0.224, neu: 0.776, pos: 0.0, compound: -0.6369, 
    
    “It’s almost as if a licence to deforest illegally and with impunity has been given, now that you have the [environmental] inspection and control teams being attacked by no less than the president of the republic and the environment minister,” Batista added.
    neg: 0.118, neu: 0.882, pos: 0.0, compound: -0.6369, 
    
    “This is a very worrying moment.”
    The spike in destruction under Bolsonaro – who was elected with the support of powerful mining and agricultural sectors – has come as a shock to environmentalists, but not a surprise.
    neg: 0.216, neu: 0.685, pos: 0.099, compound: -0.5373, 
    
    During a visit to the Amazon last year Bolsonaro told the Guardian that as president he would target “cowardly” environmental NGOs who were “sticking their noses” into Brazil’s domestic affairs.
    neg: 0.0, neu: 0.943, pos: 0.057, compound: 0.1779, 
    
    “This tomfoolery stops right here!” Bolsonaro proclaimed, praising Donald Trump’s approval of the Dakota Access and Keystone XL oil pipelines.
    neg: 0.063, neu: 0.667, pos: 0.27, compound: 0.7424, 
    
    Bolsonaro returned to that theme on Tuesday during a gathering of car dealers in Brazil’s economic capital, São Paulo, complaining that “60% of our territory is rendered unusable by indigenous reserves and other environmental questions”.
    neg: 0.052, neu: 0.948, pos: 0.0, compound: -0.2023, 
    
    “You can’t imagine how much I enjoyed talking to Macron and Angela Merkel [about these issues during the recent G20 in Japan],” Bolsonaro added to guffaws from the crowd.
    neg: 0.0, neu: 0.891, pos: 0.109, compound: 0.5106, 
    
    “What a pleasure!”
    In June Merkel described the environmental situation in Bolsonaro’s Brazil as “dramatic”.
    neg: 0.0, neu: 1.0, pos: 0.0, compound: 0.0, 
    
    In recent weeks the globally respected National Institute for Space Research has found itself at the eye of a political storm as a result of the inconvenient truths revealed by its data.
    neg: 0.068, neu: 0.765, pos: 0.167, compound: 0.5423, 
    
    Earlier this month, with alarm growing about the consequences of the intensifying assault on the Amazon, its director, Ricardo Galvão, was sacked after contesting Bolsonaro’s “pusillanimous” claims he was peddling lies about the state of the Amazon.
    neg: 0.2, neu: 0.687, pos: 0.113, compound: -0.7096, 
    
    Galvão’s successor, the air force colonel Darcton Policarpo Damião, looks set to follow a more Bolsonarian line.
    neg: 0.0, neu: 0.888, pos: 0.112, compound: 0.2263, 
    
    In an interview this week Damião said he was not convinced global heating was a manmade phenomenon and called such matters “not my cup of tea” .
    neg: 0.086, neu: 0.873, pos: 0.042, compound: -0.2865, 
    
    Pope Francis – who is preparing to host a special synod on the Amazon in October – has also incurred Bolsonaro’s wrath on the environment.
    neg: 0.0, neu: 0.82, pos: 0.18, compound: 0.5267, 
    
    In June the Argentinian leader of the Catholic church questioned “the blind and destructive mentality” of those seeking to profit from the world’s biggest rainforest.
    neg: 0.253, neu: 0.656, pos: 0.091, compound: -0.6369, 
    
    “What is happening in Amazonia will have repercussions at a global level,” he warned.
    neg: 0.149, neu: 0.851, pos: 0.0, compound: -0.2732, 
    
    Asked about those comments, Bolsonaro offered a characteristically unvarnished response, suggesting they reflected an international conspiracy to commandeer the Amazon.
    neg: 0.154, neu: 0.769, pos: 0.077, compound: -0.4019, 
    
    “Brazil is the virgin that every foreign pervert wants to get their hands on,” Bolsonaro said .
    neg: 0.18, neu: 0.82, pos: 0.0, compound: -0.5106, 
    
    Topics Brazil Jair Bolsonaro Amazon rainforest Deforestation Americas Conservation Trees and forests news
    neg: 0.0, neu: 0.876, pos: 0.124, compound: 0.1779, 
    
    


```python

```
