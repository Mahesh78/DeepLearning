
## NLP_Similarities_Word2Vec_BERT


```python
import gensim, operator
from scipy import spatial
import numpy as np
from gensim.models import KeyedVectors
import json
from collections import Counter 

model_path = '/Users/mitikirim/Downloads/'
```


```python
def load_wordvec_model(modelName, modelFile, flagBin):
    print('Loading ' + modelName + ' model...')
    model = KeyedVectors.load_word2vec_format(model_path + modelFile, binary=flagBin)
    print('Finished loading ' + modelName + ' model...')
    return model

model_word2vec = load_wordvec_model('Word2Vec', 'GoogleNews-vectors-negative300.bin.gz', True)
```

    Loading Word2Vec model...
    Finished loading Word2Vec model...
    


```python
def vec_similarity(input1, input2, vectors):
    term_vectors = [np.zeros(300), np.zeros(300)]
    terms = [input1, input2]
        
    for index, term in enumerate(terms):
        for i, t in enumerate(term.split(' ')):
            try:
                term_vectors[index] += vectors[t]
            except:
                term_vectors[index] += 0
        
    result = (1 - spatial.distance.cosine(term_vectors[0], term_vectors[1]))
    if result is 'nan':
        result = 0
        
    return result
```


```python
# function checks whether the input words are present in the vocabulary for the model
def vocab_check(vectors, words):
    
    output = list()
    for word in words:
        if word in vectors.vocab:
            output.append(word.strip())
            
    return output
```


```python
# function calculates similarity between two strings using a particular word vector model
def calc_similarity(input1, input2, vectors):
    s1words = set(vocab_check(vectors, input1.split()))
    s2words = set(vocab_check(vectors, input2.split()))
    
    output = vectors.n_similarity(s1words, s2words)
    return output
```


```python
json_data = open('anly610.json').readlines()
file_feeds = []
for line in json_data:
    file_feeds.append(json.loads(line))

title1 = file_feeds[0]['title']
```


```python
titles = []
for i in range(1,1000):
    titles.append(file_feeds[i]['title'])
```


```python
score = []
for i in range(304):
    score.append(calc_similarity(title1, titles[i],model_word2vec))
```


```python
for i in range(305,999):
    score.append(calc_similarity(title1, titles[i],model_word2vec))
```


```python
score_title = {}
for i in range(len(score)):
    score_title[titles[i]] = score[i]
```

### Final output:


```python
print('\033[1m' + '\033[91m' + '\033[4m' + 'Selected Title' + '\033[0m' + ': \n')
print(title1)
print('\n')

k = Counter(score_title)
high = k.most_common(100)
print('\033[1m' + '\033[91m' + '\033[4m' + 'Top 100 similar titles' + '\033[0m' + ': \n')
for i in high: 
    print(i[0]," :",i[1]," \n")
```

    [1m[91m[4mSelected Title[0m: 
    
    Disney+ streaming: Launch date, subscription cost, which TV shows and films feature | The Independent
    
    
    [1m[91m[4mTop 100 similar titles[0m: 
    
    Facebook to Launch TV Streaming Box with Built-in Camera, Netflix and HBO Apps  : 0.6628796  
    
    Best Amazon Prime TV shows (August 2019): the best series to watch this month  : 0.649804  
    
    Disney Plus streaming service: Release date, price, shows and movies to expect - CNET  : 0.6451072  
    
    Walmart-owned Flipkart To Launch Video Streaming Service In India  : 0.6265943  
    
    Half of British homes subscribe to a TV streaming service  : 0.59838325  
    
    The Matrix is Leaping Back Into Theaters for Its 20th Anniversary  : 0.588256  
    
    The best movies on Amazon Prime Video right now (August 2019)  : 0.5853409  
    
    The Swan Princess: Kingdom of Magic, Available Now on DVD -- It's a Must See Family-Friendly Movie for End of Summer Fun for the Whole Family! #SwanPrincessAdventures #SwanPrincess25 (Review)  : 0.5805961  
    
    realme, the No. 1 quality and emerging smartphone brand in India, is adding more cheer to Independence Day celebrations through two attractive online sales scheduled from 8th to 10th August, 2019. As part of its latest sale campaign, the young brand will make special deals, discounts, and EMI plans available to customers purchasing its range of innovative smartphones on Flipkart and Amazon, in addition to realme.com. Flipkart National Shopping Day offers As part of the National Shopping Days sale, Flipkart users purchasing realme smartphones can avail an instant 10% discount for transacting through ICICI bank debit & credit cards. The offer extends to the following realme models: realme 2 Pro, realme 3 Pro, realme 3, realme X, realme 3i and realme C1. Customers who buy realme 2 Pro will also get a flat discount of INR 500 while those going for realme C1 can avail flat INR 1000 off as part of the offer. Users can also avail No Cost EMIs of up to 6 months on realme 3 Pro (6GB variants), realme 3 (4GB variants), realme X and realme 3i. A special sale on realme X 4+128GB, 8+128GB and the Master Edition variants will also go live from August 8 at midnight. All of the aforementioned offers will also be available on realme.com also. Amazon Freedom Sale offers As part of the Freedom Sale campaign, Amazon users buying realme U1 (3GB + 32GB, 3GB + 64GB and 4GB + 64GB variants) can avail instant 10% discount on SBI Credit Cards & EMI transactions. The offer will also be applicable to realme U1 purchases made on realme.com. Shoppers using Amazon Pay will also receive INR 1000 cashback on their purchases.  : 0.5739741  
    
    Adding to the arsenal: Flipkart to take on Amazon, Netflix with free video streaming service  : 0.572226  
    
    Amazon Music cuts monthly price to 99 cents for students with Prime subscription  : 0.5713747  
    
    Midnight Mania! Disney to offer streaming bundle that includes ESPN+, Hulu, and Disney+ - MMA Mania  : 0.5637007  
    
    How to Watch US Amazon Prime Video Content from the UK  : 0.5626091  
    
    METZ 40inch Android 8.0 Smart TV to be available in Rs. 2000 discounted price at Amazon Freedom Sale starting August 7 I  : 0.5600201  
    
    Walmart’s Flipkart to add free movies, videos streaming to its app  : 0.5587646  
    
    This $1 Amazon Music Unlimited deal will have students streaming all year  : 0.5571058  
    
    Disney To Offer Streaming Bundle of Disney+, ESPN+ and Hulu  : 0.54614836  
    
    Disney to Bundle Disney+, Hulu and ESPN+ for $12.99 a Month Starting in November  : 0.5448702  
    
    Disney To Offer New Bundle Package: Disney+, Hulu And ESPN+  : 0.5444601  
    
    The Best Nintendo Switch Deals and Bundles in August 2019 - IGN  : 0.54210865  
    
    Exec reveals how studios like Netflix and Amazon decide what shows to keep and what to cancel  : 0.54171365  
    
    '​First Series Of Lord Of The Rings TV Show ‘Supposed To Be 20 Episodes’  : 0.54107916  
    
    Amazon's Twitch signs a deal with the NBA to be its exclusive digital partner for streaming USA Basketball games globally through 2020 (Hilary Russ/Reuters)  : 0.53743315  
    
    The Xbox One India Story: Failure to Find an Audience | HuffPost India  : 0.537234  
    
    ‘Friends’ Remains Top SVOD Title In UK As Nearly Half Of British Homes Take A Streaming Service  : 0.5353121  
    
    Amazon Freedom Sale 2019: Deals and offers on OnePlus 7 series, Redmi 7, Y3 and more | Technology News, The Indian Express  : 0.53359413  
    
    Disney announces $12.99 bundle for Disney+, Hulu, and ESPN+ - The Verge  : 0.5327633  
    
    From flagship killer to premium: These are the best smartphone offers during Amazon's Freedom Sale  : 0.5319081  
    
    Amazon Freedom Sale: There is a 43-inch Samsung TV For Rs 28,999 With a Fire TV Stick Bundled  : 0.529173  
    
    TV Wrap - Amazon's stunning new football docuseries an antidote to Premier League nonsense  : 0.52860004  
    
    The Fashion Guitar launches capsule collection & we want it all  : 0.5282693  
    
    The Best Nintendo Switch Deals Right Now  : 0.5264896  
    
    Flipkart National Shopping Days sale from Aug 8: Deals on Redmi Note 7 Pro, Poco F1, iPhone XR to expect  : 0.5262207  
    
    Here are the Best Xbox Deals Online Right Now  : 0.52570987  
    
    Bolsonaro rejects 'Captain Chainsaw' label as data shows deforestation 'exploded' | World news | The Guardian  : 0.52323174  
    
    Amazon Freedom Sale 2019: From Echo smart speakers to smartphones, here are 5 deals you should look out for - Technology News  : 0.52179104  
    
    The YouTube-Fire TV Spat Is Over, but the Platform Wars Will Rage On  : 0.52149117  
    
    Prime Student members can now get Amazon Music Unlimited for $0.99 per month  : 0.52046746  
    
    Amazon Prime Student members can now get Amazon Music Unlimited for $0.99 per month  : 0.52046746  
    
    Amazon Prime Video Orders Horror Comedy Series Truth Seekers  : 0.52024096  
    
    Mega Combo- Echo Dot + Fire TV Stick Bundle with 6A Oakter smart plug at Just Rs. 5499  : 0.5198074  
    
    The Guacamelee! One-Two Punch Collection is available today for PS4 & Nintendo Switch  : 0.5187744  
    
    Cross-Platform Music Streaming - Comcast Added Amazon Music Integration to Customers' Televisions (TrendHunter.com)  : 0.51591086  
    
    Flipkart National Shopping Days sale is the best time to trade-in your old smartphone  : 0.51519305  
    
    HONOR 20i Phantom Red Limited Edition to be available on Flipkart and Amazon  : 0.5139166  
    
    Both Guacamelee! Games Launch On Switch In Physical Form Today  : 0.5115995  
    
    Amazon Introduces New, Exclusive Prime Student Benefit: Amazon Music Unlimited for Just $0.99  : 0.5113273  
    
    Amazon Freedom Sale 2019: Best deals on Realme U1, Mi A2, more for Prime members  : 0.50967383  
    
    Event Horizon to Be Adapted into a TV Show By Amazon  : 0.5096721  
    
    Amazon Freedom Sale 2019 Starts From August 7; Check Best Deals On Smartphones, Other Offers On Electronics  : 0.50918186  
    
    FREE Trial of Amazon Prime for College Students + Music Unlimited just $0.99!  : 0.5086962  
    
    Amazon Music Unlimited price drops to $0.99 for Prime Student members  : 0.50792205  
    
    Scouted: The Best Essentials For a Killer Summer That You Can Easily Get on Amazon  : 0.50772285  
    
    Why Netflix canceled ‘The OA,’ and why your favorite streaming show might be on death row too  : 0.5060128  
    
    Google - Google ups the ante, updates Images to focus on shopping, Marketing & Advertising News, ET BrandEquity  : 0.50597155  
    
    Amazon Freedom Sale: Over Rs 7K discount on OPPO K3, Mi A2, Redmi Y2; Huawei Y9 Prime exclusively available  : 0.5054134  
    
    The best Amazon Echo deals for August 2019  : 0.5046832  
    
    Amazon Music Unlimited gets big price cut for Prime Student members  : 0.5035238  
    
    Amazon Is Developing An Event Horizon Series  : 0.50346977  
    
    UK viewers increasingly turning to streaming services such as Netflix and Amazon  : 0.50331277  
    
    Amazon introduces Amazon Music Unlimited for USD 0.99  : 0.5032232  
    
    Limited Time Offer: Retro Bluetooth Speaker - MyLitter - One Deal At A Time  : 0.5017246  
    
    EE phone deals just got smarter for gamers and TV fans - adding Amazon Prime Video  : 0.5016869  
    
    Amazon Music Unlimited permanently drops to $0.99 a month for Prime Student members  : 0.50048  
    
    Amazon Fire HD 10, Xbox One S, LG V35 Smartphone, and more deals for Aug. 7  : 0.50030744  
    
    News: Enter the Gungeon Now Available for Nintendo Switch  : 0.49997413  
    
    Galaxy Note 9 price crash as Samsung set to reveal huge upgrade today - The Girl Sun  : 0.4991948  
    
    Roku Q2 Earnings Preview: Active Accounts, Ads & Streaming Hours  : 0.49879885  
    
    The Sphero BB-8 collector's edition has plummeted to $70 on Amazon  : 0.49803966  
    
    Amazon Introduces New, Exclusive Prime Student Benefit: Amazon Music Unlimited Just $0.99  : 0.49750313  
    
    Amazon Vs Flipkart: Get Ready For A New Round Of Sale, Top Offers And Deals - flypped  : 0.49519187  
    
    Amazon introduces Music Unlimited for students at USD 0.99  : 0.49276838  
    
    Amazon leaks Motorola One Action, reveals price and launch date  : 0.49275  
    
    Amazon - Youth offer : Buy Prime membership worth Rs.999 and get Rs.500 cashback as Pay balance  : 0.49251628  
    
    The Best Renewed Laptop Deals On Amazon For Under $200  : 0.49170038  
    
    Desktop as a Service (DaaS) Market to Set Remarkable Growth by 2025: Leading Key Players - KEMP Technologies, Vmware, Amazon WorkSpaces  : 0.490452  
    
    Gears 5 Xbox One X Limited Edition Revealed  : 0.48980492  
    
    The Big Shift: Why Record Companies Need to Pivot From a B2B to a B2C Model (Column)  : 0.4895315  
    
    Amazon Freedom Sale: OnePlus 7 Pro is Available With Exchange Offer And Discounts  : 0.4886034  
    
    Top 10 Smartphone Deals From Flipkart & Amazon Independence Day Sale 2019  : 0.48854622  
    
    Disney will offer a streaming bundle of Disney+, Hulu, and ESPN+  : 0.48792368  
    
    Amazon’s The Boys’ Critique of Megacorporations Sure Is Rich – United States Live Feed News  : 0.48740822  
    
    Great New Benefit For Prime Students: Amazon Music Unlimited For Just $0.99/Month!  : 0.4859323  
    
    The best SIM only deals for August 2019 | from £3.99 p/m  : 0.4856839  
    
    Avengers: Endgame (4K Ultra HD + Blu-Ray) $34.98 + Delivery ($0 with Prime/ $39 Spend)  : 0.48564893  
    
    HBO one-ups Netflix in the Israeli crime genre  : 0.48556662  
    
    The Best Books That Will Make You A Master Negotiator  : 0.4853337  
    
    The first full trailer for Amazon’s Carnival Row shows off the love story at the heart of the fantasy noir  : 0.48531473  
    
    Flipkart announces National Shopping Days sale starting 8 August  : 0.48530197  
    
    Amazon's delivery robots head to the Golden State| Latest News Videos Business  : 0.48521206  
    
    YouTube and Netflix are now UK's third and fourth most popular channels, Ofcom finds  : 0.485152  
    
    Koofr Cloud Storage Plan: Lifetime Subscription for $19  : 0.48486662  
    
    Amazon Freedom Sale Offers: Best Selling Budget Smartphones To Buy This Freedom Sale  : 0.48478773  
    
    Amazon Freedom Sale 2019: Check Discounts, Cashback, Offers & Handpicked Deals  : 0.4842376  
    
    Motorola One Action listed on Amazon ahead of launch | 91mobiles.com  : 0.4840656  
    
    Huawei Y9 Prime 2019 To Go On Sale in India At 12PM; Available Via Amazon India  : 0.48177868  
    
    Amazon Freedom Sale | Amazon Sale 2019: Here are the top deals and discounts on offer  : 0.4816547  
    
    Disney plans bundle of ESPN+, Hulu and Disney+ for USD 12.99  : 0.4812339  
    
    NBA, Twitch Announce Deal for Digital Rights to USA Basketball  : 0.48052186  
    
    Gears of War 5 Xbox One X Limited Edition Revealed  : 0.48012757  
    
    
