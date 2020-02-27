### Week 3: Webhose Crawl


```python
import json
import webhoseio
```

#### Entity       : Amazon, Site Type : News


```python
webhoseio.config(token="paste token here")
query_params = {
	"q": "organization:Amazon site_type:news",
	"ts": "1562542332603",
	"sort": "published"
    }
output = webhoseio.query("filterWebContent", query_params)
    #print output['posts'][0]['text'] # Print the text of the first post
    #print output['posts'][0]['published'] # Print the text of the first post publication date
```


```python
feeds = []
for i in output['posts']:
    feeds.append(i)
    print(i['title'], i['published'])
```

    Disney+ streaming: Launch date, subscription cost, which TV shows and films feature | The Independent 2019-08-07T23:26:00.000+03:00
    Bolsonaro rejects 'Captain Chainsaw' label as data shows deforestation 'exploded' | World news | The Guardian 2019-08-07T23:25:00.000+03:00
    FedEx Dumps Amazon Ground Shipping Business in Another Bold Move 2019-08-07T23:23:00.000+03:00
    Alphabet's DeepMind Losses Soared To $570 Million In 2018 2019-08-07T23:14:00.000+03:00
    FedEx severs ties with Amazon 2019-08-07T22:30:00.000+03:00
    Amazon: FedEx severs ties, will no longer make ground deliveries 2019-08-07T22:23:00.000+03:00
    Clarks Men's Touareg Vibe Oxford from $33.93 at Amazon 2019-08-07T22:12:00.000+03:00
    Sprint customers: Free $2 Amazon gift card in the Sprint Rewards App 2019-08-07T22:11:00.000+03:00
    Why Microsoft Poached Amazon’s Top Twitch Star 2019-08-07T21:55:00.000+03:00
    FedEx severs ties with Amazon 2019-08-07T21:48:00.000+03:00
    FedEx severs ties with Amazon - News 2019-08-07T21:30:00.000+03:00
    3 Stocks That Turned $1,000 Into $5,000 (or More) in 5 Years or Less 2019-08-07T21:23:00.000+03:00
    This cheap SIM-free Huawei Mate 20 X deal from Argos now includes a free smartwatch | TechRadar 2019-08-07T21:16:00.000+03:00
    FedEx severs ties with Amazon 2019-08-07T20:51:00.000+03:00
    Bose SoundSport in-ear headphones for Apple, Samsung, and Android devices. $39.00 at Amazon 2019-08-07T20:44:00.000+03:00
    PRONTO Cruz Polyester 49 cms Coffee Travel Duffle @ Rs.899 2019-08-07T20:30:00.000+03:00
    Hurry and you can save $10 on the new Nintendo Switch game everyone’s flipping out over 2019-08-07T20:11:00.000+03:00
    Hurry and you can save $10 on the new Nintendo Switch game everyone’s flipping out over 2019-08-07T20:11:00.000+03:00
    Up to 40% Off - Fastrack Watches Starts At Rs. 599 2019-08-07T20:02:00.000+03:00
    Amazon slices 35% off select Fossil Men’s Gen 4 Explorist HR smartwatches 2019-08-07T20:02:00.000+03:00
    MIT’s AlterEgo system gives wearers an A.I. assistant in their heads 2019-08-07T20:01:00.000+03:00
    Privo Achieves End User Computing (EUC) Competency Status​ 2019-08-07T20:00:00.000+03:00
    Now you can choose how fast Alexa talks on your Amazon Echo 2019-08-07T20:00:00.000+03:00
    This is what $120 billion-plus in net worth looks like on the deck of a boat 2019-08-07T19:51:00.000+03:00
    Amazon's delivery robots head to the Golden State| Latest News Videos Business 2019-08-07T19:37:00.000+03:00
    Cheap electric razor deals: Braun Series 9, Series 8, Philips Series 9000 and 7000 and a load more better than half price 2019-08-07T19:35:00.000+03:00
    The Xbox One India Story: Failure to Find an Audience | HuffPost India 2019-08-07T19:21:00.000+03:00
    FedEx Will No Longer Deliver Amazon Packages to Customers 2019-08-07T19:19:00.000+03:00
    I've been sleeping on these $20 AmazonBasics sheets that have more than 20,000 online reviews — here's how they feel 2019-08-07T19:05:00.000+03:00
    Wonder Woman: Bloodlines - Exclusive Official Trailer 2019-08-07T19:00:00.000+03:00
    Privo Achieves End User Computing (EUC) Competency Status​ 2019-08-07T19:00:00.000+03:00
    You can now adjust how quickly Alexa speaks in the U.S. 2019-08-07T19:00:00.000+03:00
    Wonder Woman: Bloodlines - Exclusive Official Trailer 2019-08-07T19:00:00.000+03:00
    The Big Shift: Why Record Companies Need to Pivot From a B2B to a B2C Model (Column) 2019-08-07T19:00:00.000+03:00
    The Big Shift: Why Record Companies Need to Pivot From a B2B to a B2C Model (Column) 2019-08-07T19:00:00.000+03:00
    People love these true wireless earbuds with touch control like AirPods, and they’re back down to $29 2019-08-07T18:56:00.000+03:00
    People love these true wireless earbuds with touch control like AirPods, and they’re back down to $29 2019-08-07T18:56:00.000+03:00
    People love these true wireless earbuds with touch control like AirPods, and they’re back down to $29 2019-08-07T18:56:00.000+03:00
    Under Armour Mens Charged Cotton 2.0 No Show Socks (6 Pack) – $12 (reg. $19.99), Best price 2019-08-07T18:53:00.000+03:00
    FedEx to let ground contract with Amazon expire at end of August 2019-08-07T18:50:00.000+03:00
    FedEx to let ground contract with Amazon expire at end of August 2019-08-07T18:50:00.000+03:00
    Charlie Brooker gets taste of Black Mirror as son accidentally calls him 'Alexa' 2019-08-07T18:48:00.000+03:00
    Carnival Row UK release date, cast, plot, trailer for Orlando Bloom, Cara Delevingne series 2019-08-07T18:47:00.000+03:00
    Sometimes I Like to Curl Up in a Ball Board Book Only $1.48! (Reg. $6) Great Reviews! 2019-08-07T18:40:00.000+03:00
    Massive spike in Amazon rainforest deforestation since Bolsonaro came to power 2019-08-07T18:39:00.000+03:00
    Tax dodging creates record-setting $4.8B in luxury home sales, report finds 2019-08-07T18:36:00.000+03:00
    Packing for your next trip: A few things that will making traveling easier 2019-08-07T18:32:00.000+03:00
    $70 Savings at Amazon with Chase, American Express, Discover and Citi 2019-08-07T18:32:00.000+03:00
    Amazon leaks Motorola One Action, reveals price and launch date 2019-08-07T18:30:00.000+03:00
    Thermos Funtainer 10 Ounce Food Jar, Teal *Low Price!* 2019-08-07T18:30:00.000+03:00
    In the New Carnival Row Trailer, Murder and Mystery Bring 2 Lovers Back Home 2019-08-07T18:30:00.000+03:00
    The Note 10 is finally here, but definitely get the Pixel 3 XL on sale for $300 off instead 2019-08-07T18:29:00.000+03:00
    Pears Pure And Gentle Body Wash, 250ml (Pack Of 2) @ Rs.199 2019-08-07T18:28:00.000+03:00
    FedEx will no longer deliver Amazon packages 2019-08-07T18:25:00.000+03:00
    FedEx To End Ground-delivery Contract With Amazon 2019-08-07T18:25:00.000+03:00
    FedEx is Ending Ground-Delivery Contract With Amazon 2019-08-07T18:21:00.000+03:00
    Amazon and Google remove listings for firearm accessories following 2 mass shootings in the US (GOOGL, AMZN) 2019-08-07T18:21:00.000+03:00
    FedEx is Ending Ground-Delivery Contract With Amazon 2019-08-07T18:21:00.000+03:00
    Sunsilk Thick and Long Shampoo 650 ML @ Rs.177 2019-08-07T18:20:00.000+03:00
    FedEx is ending its ground-shipping contract with Amazon 2019-08-07T18:17:00.000+03:00
    FedEx cuts ties with Amazon in sign of new rivalry 2019-08-07T18:14:00.000+03:00
    Flat 59% Off: Wipro Garnet 20-Watt Slim LED Batten 2 Packs at Rs. 479 2019-08-07T18:12:00.000+03:00
    FedEx slips after canceling its ground-shipping deal with Amazon (FDX) 2019-08-07T18:10:00.000+03:00
    FedEx severs ties with Amazon 2019-08-07T18:10:00.000+03:00
    Jeff Bezos Sells Almost $3 Billion Worth Of Amazon Shares 2019-08-07T18:08:00.000+03:00
    Brazilian Amazon deforestation surges, embattled institute says 2019-08-07T18:06:00.000+03:00
    CleverMade Collapsible Cooler Bag: Insulated Leakproof 50 Can Soft Sided Portable Beverage Tote with Bottle Opener & Storage Pockets, Charcoal/Black *Discounted* 2019-08-07T18:05:00.000+03:00
    What's the difference between PS4 Share Play and Play Together? 2019-08-07T18:03:00.000+03:00
    Inalsa Hand Blender Robot 2.5PS 200-Watt 58% Off @ Rs.549 2019-08-07T18:02:00.000+03:00
    Donate blood on Aug. 9 2019-08-07T18:02:00.000+03:00
    Amazon Is Developing An Event Horizon Series 2019-08-07T18:00:00.000+03:00
    17 Clever Products We Found Hiding on Amazon That'll Make Book-Lovers Excited 2019-08-07T18:00:00.000+03:00
    Hamilton Beach Nonstick Belgian Waffle Maker *Low Price!* 2019-08-07T18:00:00.000+03:00
    Cable operator Altice is making a $400 Alexa smart speaker with premium sound 2019-08-07T17:56:00.000+03:00
    Why Microsoft Poached Amazon’s Top Twitch Star 2019-08-07T17:55:00.000+03:00
    Disney sees box office gains, but earnings fall short 2019-08-07T17:52:00.000+03:00
    Snag Google’s Pixelbook for $359 less on Amazon before heading back to school 2019-08-07T17:50:00.000+03:00
    FedEx is ending its ground-shipping contract with Amazon 2019-08-07T17:49:00.000+03:00
    FedEx severs ties with Amazon 2019-08-07T17:49:00.000+03:00
    FedEx severs ties with Amazon 2019-08-07T17:48:00.000+03:00
    FedEx severs ties with Amazon 2019-08-07T17:48:00.000+03:00
    FedEx Will No Longer Offer Ground Delivery To Amazon 2019-08-07T17:48:00.000+03:00
    FedEx severs ties with Amazon 2019-08-07T17:48:00.000+03:00
    FedEx severs ties with Amazon 2019-08-07T17:48:00.000+03:00
    Virginia winemaker Eric Trump has a big grin on his face — and one of daddy’s signature issues may be the reason why 2019-08-07T17:45:00.000+03:00
    FedEx to end ground deliveries for Amazon - MarketWatch 2019-08-07T17:45:00.000+03:00
    Today’s best deals: True wireless earbuds, $28 wireless camera, surround sound, free money from Amazon, more 2019-08-07T17:43:00.000+03:00
    EE phone deals just got smarter for gamers and TV fans - adding Amazon Prime Video 2019-08-07T17:40:00.000+03:00
    FedEx Drops Amazon 2019-08-07T17:39:00.000+03:00
    FedEx Ends Its Ground-Shipping Contract with Amazon 2019-08-07T17:34:00.000+03:00
    Amazon deforestation increases by 278% in a year, institute warns climate skeptic president Bolsonaro 2019-08-07T17:34:00.000+03:00
    FedEx severs ties with Amazon 2019-08-07T17:30:00.000+03:00
    FedEx severs ties with Amazon 2019-08-07T17:30:00.000+03:00
    OXO Good Grips Smooth Edge Can Opener *Discounted* 2019-08-07T17:30:00.000+03:00
    FedEx Severs Ties With Amazon 2019-08-07T17:30:00.000+03:00
    I tried the $42 best-selling cream that promises a 'face lift in a jar' — here's what happened 2019-08-07T17:28:00.000+03:00
    Amazon announces two more renewable energy projects 2019-08-07T17:26:00.000+03:00
    FedEx ukončí doručování pro Amazon 2019-08-07T17:25:00.000+03:00
    FedEx to end ground-delivery partnership with Amazon 2019-08-07T17:24:00.000+03:00
    HONOR 20i Phantom Red Limited Edition to be available on Flipkart and Amazon 2019-08-07T17:21:00.000+03:00
    


```python
print(len(feeds))
```

    100
    

#### Crawl for remaining batches of posts:


```python
count = 99

while True:
    output = webhoseio.get_next()
    for i in output['posts']:
        feeds.append(i)
    count -= 1
    
    if (count < 1):
        break
print(len(feeds))
```

    10000
    

#### Store in a json file:


```python
with open('anly610.json', 'w') as myfile:
    for feed in feeds:
        line = json.dumps(feed)
        myfile.write(line)
        myfile.write('\n')
```

#### Read feeds back from the newly created json file:


```python
json_data = open('anly610.json').readlines()
file_feeds = []
for line in json_data:
    file_feeds.append(json.loads(line))

print(len(file_feeds))
```

    10000
    

#### Print Title, Text and URLs of first 10 posts


```python
for feed in file_feeds[:10]:
    print('\033[1m' + '\033[91m' + '\033[4m' + 'Title' + '\033[0m' + ': ')
    print(feed['title'])
    print('\033[1m' + '\033[91m' + '\033[4m' + 'Text' + '\033[0m' + ': ')
    print(feed['text'])
    print('\033[1m' + '\033[91m' + '\033[4m' + 'URL' + '\033[0m' + ': ')
    print(feed['url'])
    print('\n')
```

    [1m[91m[4mTitle[0m: 
    Disney+ streaming: Launch date, subscription cost, which TV shows and films feature | The Independent
    [1m[91m[4mText[0m: 
    Disney+ is a brand new streaming service arriving in 2019 to rival the likes of Netflix and Amazon Prime .It will be a comprehensive streaming platform featuring all the TV shows and films produced by Disney since 1937, as well as new exclusive content.Disney+ will host five hubs dedicated to the major franchises owned by media company: Disney, Pixar, Marvel, Star Wars and National Geographic.From extras.Here’s everything we know about Disney+ so far…Disney+ will debut in the US on 12 November 2019 with a variety of classic Disney content as well as some of its new exclusives.While there is no UK release date yet, the service is set to launch in every major region around the world within the next two years. Disney+ will cost $6.99 per month in the US, or $69.99 per year – making it less expensive than Netflix.
    [1m[91m[4mURL[0m: 
    https://www.independent.co.uk/arts-entertainment/tv/news/disney-plus-streaming-service-launch-date-cost-how-much-tv-shows-films-when-hulu-a9045506.html
    
    
    [1m[91m[4mTitle[0m: 
    Bolsonaro rejects 'Captain Chainsaw' label as data shows deforestation 'exploded' | World news | The Guardian
    [1m[91m[4mText[0m: 
    Data says 2,254 sq km cleared in July as president says Macron and Merkel ‘haven’t realized Brazil’s under new management’. Deforestation in the Brazilian Amazon “exploded” in July it has emerged as Jair Bolsonaro scoffed at his portrayal as Brazil’s “Captain Chainsaw” and mocked Emmanuel Macron and Angela Merkel for challenging him over the devastation.
    Speaking in São Paulo on Tuesday, Brazil’s president attacked the leaders of France and Germany – who have both voiced concern about the surge in destruction since Bolsonaro took office in January.
    “They still haven’t realized Brazil’s under new management,” Bolsonaro declared to cheers of approval from his audience. “Now we’ve got a blooming president.”
    Amazon deforestation accelerating towards unrecoverable 'tipping point' Read more
    The far-right populist repeated claims that his administration – which critics accuse of helping unleash a new wave of environmental destruction – was the victim of a mendacious international smear campaign based on “imprecise” satellite data showing a jump in deforestation.
    Bolsonaro ridiculed what he called his depiction as “ Capitão Motoserra ” (“ Captain Chainsaw”).
    But as he spoke, official data laid bare the scale of the environmental crisis currently unfolding in the world’s biggest rainforest, of which about 60% is in Brazil .
    According to a report in the Estado de São Paulo newspaper , Amazon destruction “exploded” in July with an estimated 2,254 sq km (870 sq miles) of forest cleared, according to preliminary data gathered by Brazil’s National Institute for Space Research, the government agency that monitors deforestation.
    That is an area about half the size of Philadelphia and reportedly represents a 278% rise on the 596.6 sq km destroyed in July last year.
    Rômulo Batista, an Greenpeace campaigner based in the Amazon city of Manaus, said the numbers – while preliminary – were troubling and showed a clear trend of rising deforestation under Bolsonaro. What was not yet clear was if the devastation was “going up, going up a lot, or skyrocketing”.
    Batista blamed Bolsonaro’s “anti-environmental” discourse and policies – such as slashing the budget of Brazil’s environmental agency, Ibama – for the surge.
    “It’s almost as if a licence to deforest illegally and with impunity has been given, now that you have the [environmental] inspection and control teams being attacked by no less than the president of the republic and the environment minister,” Batista added. “This is a very worrying moment.”
    The spike in destruction under Bolsonaro – who was elected with the support of powerful mining and agricultural sectors – has come as a shock to environmentalists, but not a surprise.
    During a visit to the Amazon last year Bolsonaro told the Guardian that as president he would target “cowardly” environmental NGOs who were “sticking their noses” into Brazil’s domestic affairs.
    “This tomfoolery stops right here!” Bolsonaro proclaimed, praising Donald Trump’s approval of the Dakota Access and Keystone XL oil pipelines.
    Bolsonaro returned to that theme on Tuesday during a gathering of car dealers in Brazil’s economic capital, São Paulo, complaining that “60% of our territory is rendered unusable by indigenous reserves and other environmental questions”.
    “You can’t imagine how much I enjoyed talking to Macron and Angela Merkel [about these issues during the recent G20 in Japan],” Bolsonaro added to guffaws from the crowd. “What a pleasure!”
    In June Merkel described the environmental situation in Bolsonaro’s Brazil as “dramatic”.
    In recent weeks the globally respected National Institute for Space Research has found itself at the eye of a political storm as a result of the inconvenient truths revealed by its data.
    Earlier this month, with alarm growing about the consequences of the intensifying assault on the Amazon, its director, Ricardo Galvão, was sacked after contesting Bolsonaro’s “pusillanimous” claims he was peddling lies about the state of the Amazon.
    Galvão’s successor, the air force colonel Darcton Policarpo Damião, looks set to follow a more Bolsonarian line. In an interview this week Damião said he was not convinced global heating was a manmade phenomenon and called such matters “not my cup of tea” .
    Pope Francis – who is preparing to host a special synod on the Amazon in October – has also incurred Bolsonaro’s wrath on the environment.
    In June the Argentinian leader of the Catholic church questioned “the blind and destructive mentality” of those seeking to profit from the world’s biggest rainforest. “What is happening in Amazonia will have repercussions at a global level,” he warned.
    Asked about those comments, Bolsonaro offered a characteristically unvarnished response, suggesting they reflected an international conspiracy to commandeer the Amazon.
    “Brazil is the virgin that every foreign pervert wants to get their hands on,” Bolsonaro said .
    Topics Brazil Jair Bolsonaro Amazon rainforest Deforestation Americas Conservation Trees and forests news
    [1m[91m[4mURL[0m: 
    https://www.theguardian.com/world/2019/aug/07/bolsonaro-amazon-deforestation-exploded-july-data
    
    
    [1m[91m[4mTitle[0m: 
    FedEx Dumps Amazon Ground Shipping Business in Another Bold Move
    [1m[91m[4mText[0m: 
    FedEx will end its ground contract with e-commerce giant Amazon.com . The move follows the June decision to walk away from Amazon’s express shipping business.
    Of course, Amazon.com (ticker: AMZN) can still ship with FedEx (FDX), but it no longer has contractually negotiated rates.
    “We think FedEx realized Amazon wasn’t really a frenemy,” Trip Miller, managing partner at Memphis-based hedge fund Gullane Capital Partners, told Barron’s . Miller believes FedEx’s business with Amazon is a small, low-margin part of the total.
    Even if it’s small, the move for FedEx is another pivot away from Amazon and toward other e-commerce players that are trying to match Amazon’s ever-shrinking delivery windows.
    “This change is consistent with our strategy to focus on the broader e-commerce market, which the recent announcements related to our FedEx Ground network have us positioned extraordinarily well,” a FedEx spokeswoman told Barron’s in an emailed statement.
    The break up appears amicable on the surface. “We are constantly innovating to improve the carrier experience and sometimes that means reevaluating our carrier relationships,” an Amazon spokeswoman told Barron’s in an emailed statement. “FedEx has been a great partner over the years and we appreciate all their work delivering packages to our customers.”
    The move comes in the summer, but it has bigger implications for Christmas. Both United Parcel Service (UPS) as well as FedEx have struggled in prior years to keep up with volume during peak shipping season, while complaining about below-average profitability for Amazon business. When Christmas comes, investors will want to see how Amazon responds and how FedEx profitability changes in the absence of some of its high-volume, low-margin business.
    “Amazon volume revolves around two seasons: Prime Day and Christmas,” Matthew White, iDrive Logistics strategist, told Barron’s . “Peak capacity is expensive, and it isn’t profitable business.”
    FedEx Ground is the second largest division at the company, generating more than $20 billion in annual sales. The Express division generates more than $37 billion in sales each year.
    FedEx stock was down 1.6% on Wednesday, amid a broader market slide. FedEx shares are flat year to date and down 35% over the past year, far worse than the comparable returns of the Dow Jones Industrial Average . The fears of new competition from Amazon have loomed large in investors’ minds.
    Barron’s recently wrote positively about FedEx, believing that fears were overblown and that the extensive FedEx network represents a deep and sustainable competitive moat.
    FedEx stock rose Tuesday after Pershing Square sold its position in United Technologies (UTX), saying it has added another investment to its portfolio. Some speculated that investment was FedEx .
    Miller’s fund holds both FedEx and Amazon shares.
    “We are long-term value investors, so when you see a 10 [price to earnings] multiple on a stock with better free cash flow in 18 months we understand why it looks attractive to other value investors,” he said.
    Write to Al Root at allen.root@dowjones.com
    [1m[91m[4mURL[0m: 
    https://www.barrons.com/articles/fedex-drops-amazon-ground-shipping-business-51565182609
    
    
    [1m[91m[4mTitle[0m: 
    Alphabet's DeepMind Losses Soared To $570 Million In 2018
    [1m[91m[4mText[0m: 
    DeepMind, the Google-owned artificial intelligence firm on a mission to create human-level AI, had an expensive year in 2018, according to documents filed with the U.K.'s Companies House registry on Wednesday. The London-based AI lab—founded in 2010 by Demis Hassabis, Mustafa Suleyman, and Shane Legg—saw its pre-tax losses grow to $570 million (£470 million), up from $341 million (£281 million) in 2017, and $154 million (£127 million) in 2016.
    DeepMind's losses are growing because it continues to hire hundreds of expensive researchers and data scientists without generating any significant revenues. Amazon, Apple, Facebook are locked in an expensive battle with DeepMind and Alphabet to hire the world's best AI experts with the goal of building self-learning algorithms that can transform industries.
    In 2018, DeepMind spent $483 million (£398 million) on around 700 employees, up from $243 million (£200 million) in 2017. Other significant costs included technical infrastructure and operating costs. In addition, DeepMind spent $17 million (£14 million) on academic donations and sponsorships.
    DeepMind also spent $12 million (£9 million) on construction and $1.2 million (£1 million) on furniture and fixtures. The company is planning to move out of Google's office in King's Cross and into a new property around the middle of 2020.
    While losses at DeepMind have grown, so to have the company's revenues. Turnover almost doubled in 2018 to £103 million, up from £48 million in 2017. The firm sold some of its software to Google, which has used DeepMind's AI systems to make the cooling units in its data centres more energy efficient, and improved battery life on Android devices. DeepMind does not make any money from its work with Britain's National Health Service.
    A DeepMind spokesperson provided Forbes with the following statement:
    "We're on a long-term mission to advance AI research and use it for positive benefit. We believe there's huge potential for AI to advance scientific discovery and we're really proud of the impact our work is already having in areas such as protein folding.
    "Our DeepMind for Google team continues to make great strides bringing our expertise and knowledge to real-world challenges at Google scale, nearly doubling revenues in the past year. We will continue to invest in fundamental research and our world-class, interdisciplinary team, and look forward to the breakthroughs that lie ahead."
    In 2018, DeepMind also passed its Streams application for clinicians to Google. However, this transaction had not been completed by the time the financial statements were filed.
    Yann LeCun, chief AI scientist at Facebook, said in an interview last year that he does not think DeepMind has yet proven its worth to Google, adding that DeepMind is too isolated to have a significant impact on the tech giant. "I wouldn't want to be in the situation Demis [the CEO] is in," he said.
    [1m[91m[4mURL[0m: 
    https://www.forbes.com/sites/samshead/2019/08/07/deepmind-losses-soared-to-570-million-in-2018/
    
    
    [1m[91m[4mTitle[0m: 
    FedEx severs ties with Amazon
    [1m[91m[4mText[0m: 
    Tennessee Gov. Bill Lee addresses reporters at a news conference announcing an investment by shipping giant FedEx Corp. of $450 billion to help modernize its Memphis hub on Friday. Aug.2, 2019 in Memphis, Tenn. (AP Photo/Adrian Sainz) More NEW YORK (AP) — FedEx is severing ties with Amazon as the online retailer builds out its own delivery fleet and becomes more of a threat.
    The announcement Wednesday that FedEx would no longer make ground deliveries for Amazon comes two months after the delivery company said it was terminating its air delivery contract with Amazon.
    Amazon.com Inc. is building up its own fleet of air and ground transportation to have more control of how its packages are delivered and cut its reliance on FedEx, UPS and the U.S. Postal Service. The Seattle-based company has been leasing jets, building several package-sorting hubs at airports and has launched a program that lets contractors start businesses delivering packages in vans stamped with the Amazon logo.
    Last month, FedEx warned for the first time in a government filing that Amazon's fledging delivery business could hurt its revenue and "negatively impact our financial condition and results of operations."
    At the same time, e-commerce has become a priority for retailers like Walmart and Target, meaning that FedEx can distance itself from Amazon.com without suffering the same competitive damage it might once have. FedEx said that Amazon made up just 1.3% of its total revenue in 2018, or about $850 million.
    [1m[91m[4mURL[0m: 
    https://sg.finance.yahoo.com/news/fedex-severs-ties-amazon-135119923.html
    
    
    [1m[91m[4mTitle[0m: 
    Amazon: FedEx severs ties, will no longer make ground deliveries
    [1m[91m[4mText[0m: 
    NEW YORK – FedEx is severing ties with Amazon as the online retailer builds out its own delivery fleet and becomes more of a threat. The announcement Wednesday that FedEx would no longer make ground deliveries for Amazon comes two months after the delivery company said it was terminating its air delivery contract with Amazon.
    Amazon.com Inc. is building up its own fleet of air and ground transportation to have more control of how its packages are delivered and cut its reliance on FedEx, UPS and the U.S. Postal Service. The Seattle-based company has been leasing jets, building several package-sorting hubs at airports and has launched a program that lets contractors start businesses delivering packages in vans stamped with the Amazon logo.
    Trade war: Dow ends more than 700 points lower as China pummels stock market
    Last month, FedEx warned for the first time in a government filing that Amazon’s fledging delivery business could hurt its revenue and “negatively impact our financial condition and results of operations.”
    At the same time, e-commerce has become a priority for retailers like Walmart and Target, meaning that FedEx can distance itself from Amazon.com without suffering the same competitive damage it might once have. FedEx said that Amazon made up just 1.3% of its total revenue in 2018, or about $850 million.
    [1m[91m[4mURL[0m: 
    https://www.usatoday.com/story/money/2019/08/07/amazon-fedex-severs-ties-no-longer-make-ground-deliveries/1942569001/
    
    
    [1m[91m[4mTitle[0m: 
    Clarks Men's Touareg Vibe Oxford from $33.93 at Amazon
    [1m[91m[4mText[0m: 
     "Clarks Men's Touareg Vibe Oxford from .93 at Amazon"
    5 Aug, 9:30 pm
    [1m[91m[4mURL[0m: 
    https://www.dealighted.com/main/page/comment/Clarks_Men_s_Touareg_Vibe_Oxford_from_33_93_at_Amazon_13293193
    
    
    [1m[91m[4mTitle[0m: 
    Sprint customers: Free $2 Amazon gift card in the Sprint Rewards App
    [1m[91m[4mText[0m: 
     "Sprint customers: Free Amazon gift card in the Sprint Rewards App"
    Today, 12:20 pm
    [1m[91m[4mURL[0m: 
    https://www.dealighted.com/main/page/comment/Sprint_customers_Free_2_Amazon_gift_card_in_the_Sprint_Rewards_App_13291384
    
    
    [1m[91m[4mTitle[0m: 
    Why Microsoft Poached Amazon’s Top Twitch Star
    [1m[91m[4mText[0m: 
    Tyler "Ninja" Blevins might be an unfamiliar name to investors who don't follow esports, but the 28-year-old is the world's most popular gamer. Blevins was the top streamer on Amazon 's (NASDAQ: AMZN) Twitch (a live video platform popular with gamers) with over 14 million followers and an average of 40,000 weekly viewers.
    Blevins, who previously played for several professional gaming teams, rose to fame after he started streaming Epic Games' Fortnite: Battle Royale two years ago. His massive fanbase, which includes over 22 million subscribers on his YouTube channel (which doesn't feature live streams), attracted major sponsorship deals -- including an appearance in an NFL ad and a $1 million payment for promoting Electronic Arts ' Apex Legends .
    That's why esports viewers were recently stunned by Blevins' decision to abandon Twitch and move his live streams to Microsoft 's (NASDAQ: MSFT) Mixer. The Verge claims that Microsoft paid Blevins over $50 million to make the switch -- a whopping amount for a single celebrity gamer. Let's see why Microsoft paid such a massive amount for Ninja's channel, and whether or not it will hurt Amazon's gaming ambitions.
    A gaming keyboard, mouse, and headset. More Image source: Getty Images.
    Mixer is still a tiny underdog Microsoft launched Mixer (originally called Beam) in January 2016, but the service never gained much momentum. Mixer accounted for just 3% of all game streaming hours in the second quarter of 2019, according to StreamElements, putting it in fourth place behind Twitch (72.2%), Alphabet 's YouTube (19.5%), and Facebook Gaming (5.3%).
    Mixer was integrated into Windows 10 and the Xbox One, but it struggled against Twitch and YouTube for two reasons. First, Twitch arrived in 2011, giving it a big head start and making it the preferred platform for top-tier gamers like Ninja. YouTube also leveraged its lead in streaming videos to expand into the gaming market. Second, Amazon started bundling Twitch memberships with Prime memberships after it acquired the company in 2014.
    Mixer makes money with subscriptions to individual channels, "embers" (bought with real money) for tipping streamers, and "sparks" (earned for passive viewing) which can also be used for tips. Twitch monetizes its streams in similar ways, with channel subscriptions and "bits" for tipping.
    A gamer plays a PC game. More
    [1m[91m[4mURL[0m: 
    https://sg.finance.yahoo.com/news/why-microsoft-poached-amazon-top-135512453.html
    
    
    [1m[91m[4mTitle[0m: 
    FedEx severs ties with Amazon
    [1m[91m[4mText[0m: 
    NEW YORK — FedEx is severing ties with Amazon as the online retailer builds out its own delivery fleet and becomes more of a threat.
    The decision by FedEx also illustrates how e-commerce has become universal as major retailers ramp up their online presence.
    The announcement Wednesday that FedEx would no longer make ground deliveries for Amazon comes two months after the delivery company said it was terminating its air delivery contract with Amazon.
    Amazon.com Inc. is building up its own fleet of air and ground transportation to cut its reliance on FedEx, UPS, and the U.S. Postal Service.
    RELATED: How to stop human review of your Alexa recordings
    RELATED: Justice Department launches antitrust probe of Big Tech
    At the same time, e-commerce has become a priority for retailers like Walmart and Target, meaning that FedEx can distance itself from Amazon.com without suffering the same competitive damage it might once have.
    [1m[91m[4mURL[0m: 
    https://www.cbs8.com/article/news/nation-world/fedex-severs-ties-with-amazon/509-1be289ac-7314-4b8c-98c9-ba9bd7646349
    
    
    
