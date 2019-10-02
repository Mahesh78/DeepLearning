

```python
!pip install -U spacy --user
!pip install -U https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz --user
```

    Requirement already up-to-date: spacy in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (2.2.0)
    Requirement already satisfied, skipping upgrade: plac<1.0.0,>=0.9.6 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy) (0.9.6)
    Requirement already satisfied, skipping upgrade: murmurhash<1.1.0,>=0.28.0 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy) (1.0.2)
    Requirement already satisfied, skipping upgrade: requests<3.0.0,>=2.13.0 in c:\users\mitikirim\appdata\local\continuum\anaconda3\lib\site-packages (from spacy) (2.21.0)
    Requirement already satisfied, skipping upgrade: srsly<1.1.0,>=0.1.0 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy) (0.1.0)
    Requirement already satisfied, skipping upgrade: numpy>=1.15.0 in c:\users\mitikirim\appdata\local\continuum\anaconda3\lib\site-packages (from spacy) (1.16.2)
    Requirement already satisfied, skipping upgrade: preshed<3.1.0,>=3.0.2 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy) (3.0.2)
    Requirement already satisfied, skipping upgrade: wasabi<1.1.0,>=0.2.0 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy) (0.2.2)
    Requirement already satisfied, skipping upgrade: blis<0.5.0,>=0.4.0 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy) (0.4.1)
    Requirement already satisfied, skipping upgrade: cymem<2.1.0,>=2.0.2 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy) (2.0.2)
    Requirement already satisfied, skipping upgrade: thinc<7.2.0,>=7.1.1 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy) (7.1.1)
    Requirement already satisfied, skipping upgrade: urllib3<1.25,>=1.21.1 in c:\users\mitikirim\appdata\local\continuum\anaconda3\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy) (1.24.1)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in c:\users\mitikirim\appdata\local\continuum\anaconda3\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy) (2019.9.11)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in c:\users\mitikirim\appdata\local\continuum\anaconda3\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy) (2.8)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in c:\users\mitikirim\appdata\local\continuum\anaconda3\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy) (3.0.4)
    Requirement already satisfied, skipping upgrade: tqdm<5.0.0,>=4.10.0 in c:\users\mitikirim\appdata\local\continuum\anaconda3\lib\site-packages (from thinc<7.2.0,>=7.1.1->spacy) (4.31.1)
    Collecting https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
      Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz (37.4MB)
    Requirement already satisfied, skipping upgrade: spacy>=2.0.0a18 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from en-core-web-sm==2.0.0) (2.2.0)
    Requirement already satisfied, skipping upgrade: wasabi<1.1.0,>=0.2.0 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy>=2.0.0a18->en-core-web-sm==2.0.0) (0.2.2)
    Requirement already satisfied, skipping upgrade: preshed<3.1.0,>=3.0.2 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy>=2.0.0a18->en-core-web-sm==2.0.0) (3.0.2)
    Requirement already satisfied, skipping upgrade: cymem<2.1.0,>=2.0.2 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy>=2.0.0a18->en-core-web-sm==2.0.0) (2.0.2)
    Requirement already satisfied, skipping upgrade: thinc<7.2.0,>=7.1.1 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy>=2.0.0a18->en-core-web-sm==2.0.0) (7.1.1)
    Requirement already satisfied, skipping upgrade: murmurhash<1.1.0,>=0.28.0 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy>=2.0.0a18->en-core-web-sm==2.0.0) (1.0.2)
    Requirement already satisfied, skipping upgrade: plac<1.0.0,>=0.9.6 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy>=2.0.0a18->en-core-web-sm==2.0.0) (0.9.6)
    Requirement already satisfied, skipping upgrade: numpy>=1.15.0 in c:\users\mitikirim\appdata\local\continuum\anaconda3\lib\site-packages (from spacy>=2.0.0a18->en-core-web-sm==2.0.0) (1.16.2)
    Requirement already satisfied, skipping upgrade: srsly<1.1.0,>=0.1.0 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy>=2.0.0a18->en-core-web-sm==2.0.0) (0.1.0)
    Requirement already satisfied, skipping upgrade: requests<3.0.0,>=2.13.0 in c:\users\mitikirim\appdata\local\continuum\anaconda3\lib\site-packages (from spacy>=2.0.0a18->en-core-web-sm==2.0.0) (2.21.0)
    Requirement already satisfied, skipping upgrade: blis<0.5.0,>=0.4.0 in c:\users\mitikirim\appdata\roaming\python\python36\site-packages (from spacy>=2.0.0a18->en-core-web-sm==2.0.0) (0.4.1)
    Requirement already satisfied, skipping upgrade: tqdm<5.0.0,>=4.10.0 in c:\users\mitikirim\appdata\local\continuum\anaconda3\lib\site-packages (from thinc<7.2.0,>=7.1.1->spacy>=2.0.0a18->en-core-web-sm==2.0.0) (4.31.1)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in c:\users\mitikirim\appdata\local\continuum\anaconda3\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.0.0a18->en-core-web-sm==2.0.0) (3.0.4)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in c:\users\mitikirim\appdata\local\continuum\anaconda3\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.0.0a18->en-core-web-sm==2.0.0) (2.8)
    Requirement already satisfied, skipping upgrade: urllib3<1.25,>=1.21.1 in c:\users\mitikirim\appdata\local\continuum\anaconda3\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.0.0a18->en-core-web-sm==2.0.0) (1.24.1)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in c:\users\mitikirim\appdata\local\continuum\anaconda3\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.0.0a18->en-core-web-sm==2.0.0) (2019.9.11)
    Building wheels for collected packages: en-core-web-sm
      Building wheel for en-core-web-sm (setup.py): started
      Building wheel for en-core-web-sm (setup.py): finished with status 'done'
      Stored in directory: C:\Users\mitikirim\AppData\Local\pip\Cache\wheels\54\7c\d8\f86364af8fbba7258e14adae115f18dd2c91552406edc3fdaa
    Successfully built en-core-web-sm
    Installing collected packages: en-core-web-sm
      Found existing installation: en-core-web-sm 2.0.0
        Uninstalling en-core-web-sm-2.0.0:
          Successfully uninstalled en-core-web-sm-2.0.0
    Successfully installed en-core-web-sm-2.0.0
    


```python
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import en_core_web_sm
from spacy.lang.en import English
```


```python
import spacy
from spacy.lang.en.examples import sentences 
```


```python
nlp = spacy.load('en_core_web_sm')
```


```python
from nltk.stem.wordnet import WordNetLemmatizer
```


```python
SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]
```


```python
def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs
```


```python
def getObjsFromConjunctions(objs):
    moreObjs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs
```


```python
def getVerbsFromConjunctions(verbs):
    moreVerbs = []
    for verb in verbs:
        rightDeps = {tok.lower_ for tok in verb.rights}
        if "and" in rightDeps:
            moreVerbs.extend([tok for tok in verb.rights if tok.pos_ == "VERB"])
            if len(moreVerbs) > 0:
                moreVerbs.extend(getVerbsFromConjunctions(moreVerbs))
    return moreVerbs
```


```python
def findSubs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == "NOUN":
        return [head], isNegated(tok)
    return [], False
```


```python
def isNegated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False
```


```python
def findSVs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs
```


```python
def getObjsFromPrepositions(deps):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and dep.dep_ == "prep":
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or (tok.pos_ == "PRON" and tok.lower_ == "me")])
    return objs
```


```python
def getObjsFromAttrs(deps):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(getObjsFromPrepositions(rights))
                    if len(objs) > 0:
                        return v, objs
    return None, None
```


```python
def getObjFromXComp(deps):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None
```


```python
def getAllSubs(v):
    verbNegated = isNegated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
    return subs, verbNegated
```


```python
def getAllObjs(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    objs.extend(getObjsFromPrepositions(rights))

    #potentialNewVerb, potentialNewObjs = getObjsFromAttrs(rights)
    #if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
    #    objs.extend(potentialNewObjs)
    #    v = potentialNewVerb

    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    return v, objs
```


```python
def findSVOs(tokens):
    svos = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            v, objs = getAllObjs(v)
            for sub in subs:
                for obj in objs:
                    objNegated = isNegated(obj)
                    svos.append((sub.lower_, "!" + v.lower_ if verbNegated or objNegated else v.lower_, obj.lower_))
    return svos
```


```python
def getAbuserOntoVictimSVOs(tokens):
    maleAbuser = {'he', 'boyfriend', 'bf', 'father', 'dad', 'husband', 'brother', 'man'}
    femaleAbuser = {'she', 'girlfriend', 'gf', 'mother', 'mom', 'wife', 'sister', 'woman'}
    neutralAbuser = {'pastor', 'abuser', 'offender', 'ex', 'x', 'lover', 'church', 'they'}
    victim = {'me', 'sister', 'brother', 'child', 'kid', 'baby', 'friend', 'her', 'him', 'man', 'woman'}

    svos = findSVOs(tokens)
    wnl = WordNetLemmatizer()
    passed = []
    for s, v, o in svos:
        s = wnl.lemmatize(s)
        v = "!" + wnl.lemmatize(v[1:], 'v') if v[0] == "!" else wnl.lemmatize(v, 'v')
        o = "!" + wnl.lemmatize(o[1:]) if o[0] == "!" else wnl.lemmatize(o)
        if s in maleAbuser.union(femaleAbuser).union(neutralAbuser) and o in victim:
            passed.append((s, v, o))
    return passed
```

## Example


```python
def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])

def testSVOs():
    #nlp = English()

    tok = nlp("making $12 an hour? where am i going to go? i have no other financial assistance available and he certainly won't provide support.")
    svos = findSVOs(tok)
    printDeps(tok)
    assert set(svos) == {('i', '!have', 'assistance'), ('he', '!provide', 'support')}
    print(svos)
    
    tok = nlp("i don't have other assistance")
    svos = findSVOs(tok)
    printDeps(tok)
    assert set(svos) == {('i', '!have', 'assistance')}

    print("-----------------------------------------------")
    tok = nlp("They ate the pizza with anchovies.")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('they', 'ate', 'pizza')}

    print("--------------------------------------------------")
    tok = nlp("I have no other financial assistance available and he certainly won't provide support.")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('i', '!have', 'assistance'), ('he', '!provide', 'support')}

    print("--------------------------------------------------")
    tok = nlp("I have no other financial assistance available, and he certainly won't provide support.")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('i', '!have', 'assistance'), ('he', '!provide', 'support')}

    print("--------------------------------------------------")
    tok = nlp("he did not kill me")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', '!kill', 'me')}

    #print("--------------------------------------------------")
    #tok = nlp("he is an evil man that hurt my child and sister")
    #svos = findSVOs(tok)
    #printDeps(tok)
    #print(svos)
    #assert set(svos) == {('he', 'hurt', 'child'), ('he', 'hurt', 'sister'), ('man', 'hurt', 'child'), ('man', 'hurt', 'sister')}

    print("--------------------------------------------------")
    tok = nlp("he told me i would die alone with nothing but my career someday")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', 'told', 'me')}

    print("--------------------------------------------------")
    tok = nlp("I wanted to kill him with a hammer.")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('i', 'kill', 'him')}

    print("--------------------------------------------------")
    tok = nlp("because he hit me and also made me so angry i wanted to kill him with a hammer.")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', 'hit', 'me'), ('i', 'kill', 'him')}

    print("--------------------------------------------------")
    tok = nlp("he and his brother shot me")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', 'shot', 'me'), ('brother', 'shot', 'me')}

    print("--------------------------------------------------")
    tok = nlp("he and his brother shot me and my sister")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', 'shot', 'me'), ('he', 'shot', 'sister'), ('brother', 'shot', 'me'), ('brother', 'shot', 'sister')}

    print("--------------------------------------------------")
    tok = nlp("the boy raced the girl who had a hat that had spots.")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    #assert set(svos) == {('boy', 'raced', 'girl'), ('who', 'had', 'hat'), ('hat', 'had', 'spots')}

    print("--------------------------------------------------")
    tok = nlp("he spit on me")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', 'spit', 'me')}

    print("--------------------------------------------------")
    tok = nlp("he didn't spit on me")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', '!spit', 'me')}

    print("--------------------------------------------------")
    tok = nlp("the boy raced the girl who had a hat that didn't have spots.")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    #assert set(svos) == {('boy', 'raced', 'girl'), ('who', 'had', 'hat'), ('hat', '!have', 'spots')}

    print("--------------------------------------------------")
    tok = nlp("he is a nice man that didn't hurt my child and sister")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    #assert set(svos) == {('he', 'is', 'man'), ('man', '!hurt', 'child'), ('man', '!hurt', 'sister')}

    print("--------------------------------------------------")
    tok = nlp("he didn't spit on me and my child")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
    assert set(svos) == {('he', '!spit', 'me'), ('he', '!spit', 'child')}

    print("--------------------------------------------------")
    tok = nlp("he beat and hurt me")
    svos = findSVOs(tok)
    printDeps(tok)
    print(svos)
```

## Load the file


```python
import json

json_data = open('anly610.json').readlines()
file_feeds = []
for line in json_data:
    file_feeds.append(json.loads(line))

title1 = file_feeds[0]['title']
print(title1)
```

    Disney+ streaming: Launch date, subscription cost, which TV shows and films feature | The Independent
    


```python
file_feeds[0]
```




    {'thread': {'uuid': 'c395929128774c187c6087a91263c75eb75a854a',
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
     'crawled': '2019-08-07T17:45:03.007+03:00'}




```python
def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])

def testSVOs():
    #nlp = English()

    tok = nlp(file_feeds[0]['text'])
    svos = findSVOs(tok)
    printDeps(tok)
    #assert set(svos) == {('i', '!have', 'assistance'), ('he', '!provide', 'support')}
    print(svos)
```


```python
if __name__ == "__main__":
    testSVOs()
```

    Disney+ nsubj PROPN is [] []
    is ROOT AUX is ['Disney+'] ['service', 'be', '.']
    a det DET service [] []
    brand nmod NOUN service [] []
    new amod ADJ service [] []
    streaming amod VERB service [] []
    service attr NOUN is ['a', 'brand', 'new', 'streaming'] ['arriving']
    arriving acl VERB service [] ['in', 'rival']
    in prep ADP arriving [] ['2019']
    2019 pobj NUM in [] []
    to aux PART rival [] []
    rival advcl VERB arriving ['to'] ['likes']
    the det DET likes [] []
    likes dobj NOUN rival ['the'] ['of']
    of prep ADP likes [] ['Netflix']
    Netflix pobj PROPN of [] ['and', 'Prime']
    and cc CCONJ Netflix [] []
    Amazon compound PROPN Prime [] []
    Prime conj PROPN Netflix ['Amazon'] []
    .It nsubj PROPN be [] []
    will aux AUX be [] []
    be conj AUX is ['.It', 'will'] ['platform']
    a det DET platform [] []
    comprehensive amod ADJ platform [] []
    streaming amod NOUN platform [] []
    platform attr NOUN be ['a', 'comprehensive', 'streaming'] ['featuring', ',', 'as', 'content']
    featuring acl VERB platform [] ['shows']
    all predet DET shows [] []
    the det DET shows [] []
    TV compound NOUN shows [] []
    shows dobj NOUN featuring ['all', 'the', 'TV'] ['and', 'films']
    and cc CCONJ shows [] []
    films conj NOUN shows [] ['produced']
    produced acl VERB films [] ['by', 'since']
    by agent ADP produced [] ['Disney']
    Disney pobj PROPN by [] []
    since prep SCONJ produced [] ['1937']
    1937 pobj NUM since [] []
    , punct PUNCT platform [] []
    as advmod ADV as [] []
    well advmod ADV as [] []
    as cc SCONJ platform ['as', 'well'] []
    new amod ADJ content [] []
    exclusive amod ADJ content [] []
    content conj NOUN platform ['new', 'exclusive'] []
    . punct PUNCT is [] []
    Disney+ nsubj PROPN host [] []
    will aux AUX host [] []
    host ROOT VERB host ['Disney+', 'will'] ['hubs', '.']
    five nummod NUM hubs [] []
    hubs dobj NOUN host ['five'] ['dedicated', ':', 'Disney']
    dedicated acl ADJ hubs [] ['to']
    to prep ADP dedicated [] ['franchises']
    the det DET franchises [] []
    major amod ADJ franchises [] []
    franchises pobj NOUN to ['the', 'major'] ['owned']
    owned acl VERB franchises [] ['by']
    by agent ADP owned [] ['company']
    media compound NOUN company [] []
    company pobj NOUN by ['media'] []
    : punct PUNCT hubs [] []
    Disney appos PROPN hubs [] [',', 'Pixar']
    , punct PUNCT Disney [] []
    Pixar conj PROPN Disney [] [',', 'Marvel']
    , punct PUNCT Pixar [] []
    Marvel conj PROPN Pixar [] [',', 'Wars']
    , punct PUNCT Marvel [] []
    Star compound PROPN Wars [] []
    Wars conj PROPN Marvel ['Star'] ['and', 'Geographic']
    and cc CCONJ Wars [] []
    National compound PROPN Geographic [] []
    Geographic conj PROPN Wars ['National'] []
    . punct PUNCT host [] []
    From ROOT ADP From [] ['extras', '.']
    extras pobj NOUN From [] []
    . punct PUNCT From [] []
    Here advmod ADV ’s [] []
    ’s advcl VERB debut ['Here'] ['everything']
    everything dobj PRON ’s [] ['know']
    we nsubj PRON know [] []
    know relcl VERB everything ['we'] ['about', 'far']
    about prep ADP know [] ['Disney+']
    Disney+ pobj NOUN about [] []
    so advmod ADV far [] []
    far advmod ADV know ['so'] []
    … punct PUNCT debut [] []
    Disney+ nsubj NOUN debut [] []
    will aux AUX debut [] []
    debut ROOT VERB debut ['’s', '…', 'Disney+', 'will'] ['in', 'on', 'with', '.']
    in prep ADP debut [] ['US']
    the det DET US [] []
    US pobj PROPN in ['the'] []
    on prep ADP debut [] ['November']
    12 nummod NUM November [] []
    November pobj PROPN on ['12'] ['2019']
    2019 nummod NUM November [] []
    with prep ADP debut [] ['variety']
    a det DET variety [] []
    variety pobj NOUN with ['a'] ['of', 'as', 'some']
    of prep ADP variety [] ['content']
    classic amod ADJ content [] []
    Disney compound PROPN content [] []
    content pobj NOUN of ['classic', 'Disney'] []
    as advmod ADV as [] []
    well advmod ADV as [] []
    as cc SCONJ variety ['as', 'well'] []
    some conj DET variety [] ['of']
    of prep ADP some [] ['exclusives']
    its poss PRON exclusives [] []
    new amod ADJ exclusives [] []
    exclusives pobj NOUN of ['its', 'new'] []
    . punct PUNCT debut [] []
    While mark SCONJ is [] []
    there expl PRON is [] []
    is advcl AUX set ['While', 'there'] ['date']
    no det DET date [] []
    UK compound PROPN release [] []
    release compound VERB date ['UK'] []
    date attr NOUN is ['no', 'release'] ['yet']
    yet advmod ADV date [] []
    , punct PUNCT set [] []
    the det DET service [] []
    service nsubjpass NOUN set ['the'] []
    is auxpass AUX set [] []
    set ROOT VERB set ['is', ',', 'service', 'is'] ['launch', '.']
    to aux PART launch [] []
    launch xcomp VERB set ['to'] ['in', 'within']
    in prep ADP launch [] ['region']
    every det DET region [] []
    major amod ADJ region [] []
    region pobj NOUN in ['every', 'major'] ['around']
    around prep ADP region [] ['world']
    the det DET world [] []
    world pobj NOUN around ['the'] []
    within prep ADP launch [] ['years']
    the det DET years [] []
    next amod ADJ years [] []
    two nummod NUM years [] []
    years pobj NOUN within ['the', 'next', 'two'] []
    . punct PUNCT set [] []
    Disney+ nsubj NOUN cost [] []
    will aux AUX cost [] []
    cost ROOT VERB cost ['Disney+', 'will'] ['6.99', 'in', ',', 'or', '69.99', '–', 'making', '.']
    $ nmod SYM 6.99 [] []
    6.99 dobj NUM cost ['$'] ['per']
    per prep ADP 6.99 [] ['month']
    month pobj NOUN per [] []
    in prep ADP cost [] ['US']
    the det DET US [] []
    US pobj PROPN in ['the'] []
    , punct PUNCT cost [] []
    or cc CCONJ cost [] []
    $ nmod SYM 69.99 [] []
    69.99 conj NUM cost ['$'] ['per']
    per prep ADP 69.99 [] ['year']
    year pobj NOUN per [] []
    – punct PUNCT cost [] []
    making advcl VERB cost [] ['expensive']
    it nsubj PRON expensive [] []
    less advmod ADV expensive [] []
    expensive ccomp ADJ making ['it', 'less'] ['than']
    than prep SCONJ expensive [] ['Netflix']
    Netflix pobj PROPN than [] []
    . punct PUNCT cost [] []
    [('service', 'rival', 'likes'), ('platform', 'featuring', 'shows'), ('platform', 'featuring', 'films'), ('disney+', 'host', 'hubs'), ('disney+', 'cost', '6.99')]
    
