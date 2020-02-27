# Regex, Sentiment Analysis


```python
# Libraries

import re
from nltk import word_tokenize, sent_tokenize, ngrams
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions
```


```python
text = """Disney will have a competitive advantage over Netflix when the entertainment conglomerate launches a competing video streaming platform later this year, according to Wall Street analyst David Trainer.
”[Disney’s] got the ability to merchandise, which is another way to monetize content in a way that Netflix does not have,” Trainer said on CNBC’s “Closing Bell” Wednesday. He’s chief of the New Constructs research firm.
Netflix increased its subscription prices Tuesday, sending the stock up 6.5 percent that day as well.
However, Trainer called it a “key dilemma” for the company and it “makes their competitors more viable.” The dilemma, he explained, is that the streaming company relies too much on its subscribers to generate revenue.
“It’s a Catch-22 for a business model that, when you look at the fundamentals, really just doesn’t work,” Trainer alleged.
The Netflix price increase for U.S. subscribers ranges between 13 percent and 18 percent, which Victory Anthony of Aegis Capital sees as a positive, a view generally shared by much of the investment community.
“It’s all profit for the price increase and so they can either use that to invest in more original content or they can let that drop down to their down to the bottom line,” Anthony said on “Squawk Alley” Thursday.
Aegis put a hold on Neflix at current levels because it’s about 8 percent higher than Anthony’s price target of $325. The stock was trading steady around $351 midday Thursday, up more than 50 percent since the Christmas Eve washout. Netflix releases its fourth quarter earnings after the bell Thursday. Netflix last reported double-digit user growth, with 58 million U.S. and 78 million international subscribers.
Original content aside, Netflix has built its large subscriber base, in part, on licensed content from a number of third-party TV and movie studios that plan to crowd into the video streaming market, which already includes other established online rivals such as Amazon and Hulu.
Disney, which agreed to purchase Twenty-First Century Fox assets last summer, said it would pull its movies from Netflix when it launches Disney+ in late 2019. AT&T’s WarnerMedia announced in October it would release a platform in the fourth quarter of 2019. Apple could be dropping a service this year, Comcast’s NBCUniversal on Monday revealed plans for a free streaming program with ads slated for early 2020.
New Constructs’ Trainer said Netflix is vulnerable because it’s a “one-trick pony” with an online distribution system that is not “defensible.” The company can keep growing its subscriber base, but it will need to address cash flow, he said.
“You can count on one hand the number of firms that have, over time, successfully monetized original content. It’s an expensive, difficult proposition,” he argued. “Disney’s done it and part of the reason they’ve done [it] is because they’ve got better ways of monetizing.”"""
```


```python
text
```




    'Disney will have a competitive advantage over Netflix when the entertainment conglomerate launches a competing video streaming platform later this year, according to Wall Street analyst David Trainer.\n”[Disney’s] got the ability to merchandise, which is another way to monetize content in a way that Netflix does not have,” Trainer said on CNBC’s “Closing Bell” Wednesday. He’s chief of the New Constructs research firm.\nNetflix increased its subscription prices Tuesday, sending the stock up 6.5 percent that day as well.\nHowever, Trainer called it a “key dilemma” for the company and it “makes their competitors more viable.” The dilemma, he explained, is that the streaming company relies too much on its subscribers to generate revenue.\n“It’s a Catch-22 for a business model that, when you look at the fundamentals, really just doesn’t work,” Trainer alleged.\nThe Netflix price increase for U.S. subscribers ranges between 13 percent and 18 percent, which Victory Anthony of Aegis Capital sees as a positive, a view generally shared by much of the investment community.\n“It’s all profit for the price increase and so they can either use that to invest in more original content or they can let that drop down to their down to the bottom line,” Anthony said on “Squawk Alley” Thursday.\nAegis put a hold on Neflix at current levels because it’s about 8 percent higher than Anthony’s price target of $325. The stock was trading steady around $351 midday Thursday, up more than 50 percent since the Christmas Eve washout. Netflix releases its fourth quarter earnings after the bell Thursday. Netflix last reported double-digit user growth, with 58 million U.S. and 78 million international subscribers.\nOriginal content aside, Netflix has built its large subscriber base, in part, on licensed content from a number of third-party TV and movie studios that plan to crowd into the video streaming market, which already includes other established online rivals such as Amazon and Hulu.\nDisney, which agreed to purchase Twenty-First Century Fox assets last summer, said it would pull its movies from Netflix when it launches Disney+ in late 2019. AT&T’s WarnerMedia announced in October it would release a platform in the fourth quarter of 2019. Apple could be dropping a service this year, Comcast’s NBCUniversal on Monday revealed plans for a free streaming program with ads slated for early 2020.\nNew Constructs’ Trainer said Netflix is vulnerable because it’s a “one-trick pony” with an online distribution system that is not “defensible.” The company can keep growing its subscriber base, but it will need to address cash flow, he said.\n“You can count on one hand the number of firms that have, over time, successfully monetized original content. It’s an expensive, difficult proposition,” he argued. “Disney’s done it and part of the reason they’ve done [it] is because they’ve got better ways of monetizing.”'



### Find all matches of $ amounts in the article


```python
print(re.findall('\$(.+?) ',text))
```

    ['325.', '351']
    

### Substitute all numbers with # and print text


```python
print(re.sub(r'[0-9]','#',text))
```

    Disney will have a competitive advantage over Netflix when the entertainment conglomerate launches a competing video streaming platform later this year, according to Wall Street analyst David Trainer.
    ”[Disney’s] got the ability to merchandise, which is another way to monetize content in a way that Netflix does not have,” Trainer said on CNBC’s “Closing Bell” Wednesday. He’s chief of the New Constructs research firm.
    Netflix increased its subscription prices Tuesday, sending the stock up #.# percent that day as well.
    However, Trainer called it a “key dilemma” for the company and it “makes their competitors more viable.” The dilemma, he explained, is that the streaming company relies too much on its subscribers to generate revenue.
    “It’s a Catch-## for a business model that, when you look at the fundamentals, really just doesn’t work,” Trainer alleged.
    The Netflix price increase for U.S. subscribers ranges between ## percent and ## percent, which Victory Anthony of Aegis Capital sees as a positive, a view generally shared by much of the investment community.
    “It’s all profit for the price increase and so they can either use that to invest in more original content or they can let that drop down to their down to the bottom line,” Anthony said on “Squawk Alley” Thursday.
    Aegis put a hold on Neflix at current levels because it’s about # percent higher than Anthony’s price target of $###. The stock was trading steady around $### midday Thursday, up more than ## percent since the Christmas Eve washout. Netflix releases its fourth quarter earnings after the bell Thursday. Netflix last reported double-digit user growth, with ## million U.S. and ## million international subscribers.
    Original content aside, Netflix has built its large subscriber base, in part, on licensed content from a number of third-party TV and movie studios that plan to crowd into the video streaming market, which already includes other established online rivals such as Amazon and Hulu.
    Disney, which agreed to purchase Twenty-First Century Fox assets last summer, said it would pull its movies from Netflix when it launches Disney+ in late ####. AT&T’s WarnerMedia announced in October it would release a platform in the fourth quarter of ####. Apple could be dropping a service this year, Comcast’s NBCUniversal on Monday revealed plans for a free streaming program with ads slated for early ####.
    New Constructs’ Trainer said Netflix is vulnerable because it’s a “one-trick pony” with an online distribution system that is not “defensible.” The company can keep growing its subscriber base, but it will need to address cash flow, he said.
    “You can count on one hand the number of firms that have, over time, successfully monetized original content. It’s an expensive, difficult proposition,” he argued. “Disney’s done it and part of the reason they’ve done [it] is because they’ve got better ways of monetizing.”
    

### Print counts of ”Netflix” and “Disney” mentions.


```python
print('Netflix: '+ str(len(re.findall(r'\b'+ 'NETflix' + r'\b', text, re.IGNORECASE))))
print('Disney: '+ str(len(re.findall(r'\b'+ 'DisneY' + r'\b', text, re.IGNORECASE))))
```

    Netflix: 9
    Disney: 5
    

### 2) Use nltk library to tokenize sentences, words and print trigrams in the first 3 sentences.

#### Tokenize first 3 sentences


```python
sentences = sent_tokenize(text)[0:3]
print(sentences)
```

    ['Disney will have a competitive advantage over Netflix when the entertainment conglomerate launches a competing video streaming platform later this year, according to Wall Street analyst David Trainer.', '”[Disney’s] got the ability to merchandise, which is another way to monetize content in a way that Netflix does not have,” Trainer said on CNBC’s “Closing Bell” Wednesday.', 'He’s chief of the New Constructs research firm.']
    

#### Tokenize words in the first 3 sentences


```python
for sentence in sentences:
    print(word_tokenize(sentence))
```

    ['Disney', 'will', 'have', 'a', 'competitive', 'advantage', 'over', 'Netflix', 'when', 'the', 'entertainment', 'conglomerate', 'launches', 'a', 'competing', 'video', 'streaming', 'platform', 'later', 'this', 'year', ',', 'according', 'to', 'Wall', 'Street', 'analyst', 'David', 'Trainer', '.']
    ['”', '[', 'Disney', '’', 's', ']', 'got', 'the', 'ability', 'to', 'merchandise', ',', 'which', 'is', 'another', 'way', 'to', 'monetize', 'content', 'in', 'a', 'way', 'that', 'Netflix', 'does', 'not', 'have', ',', '”', 'Trainer', 'said', 'on', 'CNBC', '’', 's', '“', 'Closing', 'Bell', '”', 'Wednesday', '.']
    ['He', '’', 's', 'chief', 'of', 'the', 'New', 'Constructs', 'research', 'firm', '.']
    

#### Print trigrams in the first 3 sentences


```python
Trigrams = []
for grams in ngrams(''.join(sentences).split(),3):
    #print(grams)
    Trigrams.append(grams)
print(Trigrams)
```

    [('Disney', 'will', 'have'), ('will', 'have', 'a'), ('have', 'a', 'competitive'), ('a', 'competitive', 'advantage'), ('competitive', 'advantage', 'over'), ('advantage', 'over', 'Netflix'), ('over', 'Netflix', 'when'), ('Netflix', 'when', 'the'), ('when', 'the', 'entertainment'), ('the', 'entertainment', 'conglomerate'), ('entertainment', 'conglomerate', 'launches'), ('conglomerate', 'launches', 'a'), ('launches', 'a', 'competing'), ('a', 'competing', 'video'), ('competing', 'video', 'streaming'), ('video', 'streaming', 'platform'), ('streaming', 'platform', 'later'), ('platform', 'later', 'this'), ('later', 'this', 'year,'), ('this', 'year,', 'according'), ('year,', 'according', 'to'), ('according', 'to', 'Wall'), ('to', 'Wall', 'Street'), ('Wall', 'Street', 'analyst'), ('Street', 'analyst', 'David'), ('analyst', 'David', 'Trainer.”[Disney’s]'), ('David', 'Trainer.”[Disney’s]', 'got'), ('Trainer.”[Disney’s]', 'got', 'the'), ('got', 'the', 'ability'), ('the', 'ability', 'to'), ('ability', 'to', 'merchandise,'), ('to', 'merchandise,', 'which'), ('merchandise,', 'which', 'is'), ('which', 'is', 'another'), ('is', 'another', 'way'), ('another', 'way', 'to'), ('way', 'to', 'monetize'), ('to', 'monetize', 'content'), ('monetize', 'content', 'in'), ('content', 'in', 'a'), ('in', 'a', 'way'), ('a', 'way', 'that'), ('way', 'that', 'Netflix'), ('that', 'Netflix', 'does'), ('Netflix', 'does', 'not'), ('does', 'not', 'have,”'), ('not', 'have,”', 'Trainer'), ('have,”', 'Trainer', 'said'), ('Trainer', 'said', 'on'), ('said', 'on', 'CNBC’s'), ('on', 'CNBC’s', '“Closing'), ('CNBC’s', '“Closing', 'Bell”'), ('“Closing', 'Bell”', 'Wednesday.He’s'), ('Bell”', 'Wednesday.He’s', 'chief'), ('Wednesday.He’s', 'chief', 'of'), ('chief', 'of', 'the'), ('of', 'the', 'New'), ('the', 'New', 'Constructs'), ('New', 'Constructs', 'research'), ('Constructs', 'research', 'firm.')]
    

### 3) Return list of extracted entities.


```python

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2019-07-12',
    iam_apikey='Api Key here',
    url='https://gateway.watsonplatform.net/natural-language-understanding/api'
    )

response = natural_language_understanding.analyze(
    text= text,
    features=Features(entities=EntitiesOptions(sentiment=True,limit=7))).get_result()

print(json.dumps(response, indent=2))
```

    {
      "usage": {
        "text_units": 1,
        "text_characters": 2910,
        "features": 1
      },
      "language": "en",
      "entities": [
        {
          "type": "Company",
          "text": "Netflix",
          "sentiment": {
            "score": -0.411348,
            "mixed": "1",
            "label": "negative"
          },
          "relevance": 0.959082,
          "disambiguation": {
            "subtype": [
              "Organization",
              "VentureFundedCompany"
            ],
            "name": "Netflix",
            "dbpedia_resource": "http://dbpedia.org/resource/Netflix"
          },
          "count": 9,
          "confidence": 1
        },
        {
          "type": "Person",
          "text": "David Trainer",
          "sentiment": {
            "score": 0.29847,
            "label": "positive"
          },
          "relevance": 0.459234,
          "disambiguation": {
            "name": "David_Trainer",
            "dbpedia_resource": "http://dbpedia.org/resource/David_Trainer"
          },
          "count": 1,
          "confidence": 0.948275
        },
        {
          "type": "Company",
          "text": "Disney",
          "sentiment": {
            "score": 0.374508,
            "mixed": "1",
            "label": "positive"
          },
          "relevance": 0.457171,
          "disambiguation": {
            "name": "The_Walt_Disney_Company",
            "dbpedia_resource": "http://dbpedia.org/resource/The_Walt_Disney_Company"
          },
          "count": 5,
          "confidence": 1
        },
        {
          "type": "Quantity",
          "text": "6.5 percent",
          "sentiment": {
            "score": -0.653385,
            "label": "negative"
          },
          "relevance": 0.414561,
          "count": 1,
          "confidence": 0.8
        },
        {
          "type": "Quantity",
          "text": "13 percent",
          "sentiment": {
            "score": 0.956423,
            "label": "positive"
          },
          "relevance": 0.348673,
          "count": 1,
          "confidence": 0.8
        },
        {
          "type": "Quantity",
          "text": "18 percent",
          "sentiment": {
            "score": 0.956423,
            "label": "positive"
          },
          "relevance": 0.346335,
          "count": 1,
          "confidence": 0.8
        },
        {
          "type": "Company",
          "text": "Aegis Capital",
          "sentiment": {
            "score": 0.956423,
            "label": "positive"
          },
          "relevance": 0.340317,
          "count": 1,
          "confidence": 0.784203
        }
      ]
    }
    
