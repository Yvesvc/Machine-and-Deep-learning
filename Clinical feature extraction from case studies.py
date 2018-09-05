
# coding: utf-8

# In[1]:


#In this project I will create a program that can extract relevant clinical features that are presented in medical case studies


# In[2]:


#Import the relevant packages or modules that will be needed for this task
#The library that I will mostly use for this task is NLTK, a Natural Language processing toolkit
import re
from nltk import RegexpParser
from nltk.tokenize import word_tokenize
from nltk import pos_tag


# In[3]:


#First, I gathered a few use cases (taken from https://www.aacc.org/publications/clinical-chemistry/clinical-case-studies)in a txt file, seperated by a new line.
#This will be used as a basis for how we will extract relevant information from the text

file = 'C:\\Users\\Yves Vc\\Desktop\\case_studies.txt'

cases = open(file, 'r').read()


# In[5]:


#Have a look at the cases

cases

"""A 31-year-old Chinese woman with bilateral upper extremity swelling presented to the emergency department at the 20th week of gestation.
\n\nA primary care physician telephoned to inquire about the clinical significance of low alkaline phosphatase (ALP) in a 54-year-old man, which led to an investigation for the cause of the low ALP, a biochemical abnormality that is known to be underappreciated.
\n\nA 69-year-old man was referred to the endocrine clinic with a 3-year history of erectile dysfunction, reduced libido, and lack of nocturnal tumescence with no response to phosphodiesterase type 5 inhibitors (sildenafil and tadalafil). The symptoms troubled him to such an extent that he asked his general practitioner to be referred to a specialist clinic.
\n\nAn 18-year-old male patient was transferred to our hospital for evaluation of recent-onset generalized weakness, difficulty walking, and a 3-week history of progressive numbness in his fingertips and distal extremities.
\n\nA 3-month-old boy was seen for routine follow-up at the pediatric nephrology outpatient clinic. He had been diagnosed as having Sotos syndrome manifesting with craniofacial dysmorphism, feeding difficulties, pulmonary artery stenosis, and atrial septal defect, as well as complex urological abnormalities.
\n\nA 12-year-old female patient presented with nonspecific symptoms including fatigue, fever, and headache persisting for approximately 4 months. Plasma ammonia was 230 µg/dL (135 µmol/L) (critical value >187 µg/dL or 110 µmol/L). The patient was referred to a pediatric emergency center for urgent evaluation of hyperammonemia."""


# In[6]:


#Based on reading these case studies, I found that the age of a patient is always structured in the same manner: 
#a(n) + space + number + hyphe + month/year + hyphen + old eg A 42-year-old
#Knowing this, we can extract the age of a patient (aka the number in that structure), using regular expressions
#What is a regular expression: Expresses a pattern of text that is to be located
def age(case_study):
    result = re.findall(r'[a,A]\s\d+[-][a-z]+[-]old', case_study)
    for i in result:
        return re.findall(r'\d+', i)


# In[7]:


#To find the gender of someone, we simply look up the gender-word in the text:

def gender(case_study):
    if 'woman' in cases:
        return ('woman')
    elif 'man' in cases:
        return('man')
    elif 'male' in cases:
        return('male')
    elif 'boy' in cases:
        return('boy')
    elif 'female' in cases:
        return('female')


# In[8]:


#This function firstly tokenizes the text (aka seperates it in words) 
#and then part-of-speech tags each word (aka labels these words eg: 32 is a number and happy is an adjective )
#It will be used in the cells below

def pos_tagger(case_study):
    cases_tokenized = word_tokenize(case_study)
    cases_tagged = pos_tag(cases_tokenized)
    return cases_tagged


# In[9]:


#Similarly to the age description, relevant clinical features in case studies are often structured in the same way.
#Therefore, we can combine the part-of-speech-tags of words together with regular expressions
#This function looks up the following structure: comma + 0 or 1 times any kind of verb + 1 or more times any kind of noun.
#Example: ,reduced libido
#I decided to remove the comma in the output of the code because it adds no information

def combo1(case_study):
    
    chunkgram = r"""Chunk: {<,><VB.?>?<NN.?>+}"""

    chunkparser = RegexpParser(chunkgram)
    chunked = chunkparser.parse(pos_tagger(case_study))
    clin_feat = ''
    for subtree in chunked.subtrees(filter = lambda t: t.label() == 'Chunk'):
        clin_feat += ' |feature: '
        for word, pos in subtree:
            clin_feat += ' '
            if word == ',':
                continue
            else:
                clin_feat += word
    return(clin_feat)


# In[10]:


#In the same way, we can do this using different regular expressions to locate other patterns of text
def combo2(case_study):
    
    chunkgram = r"""Chunk: {<IN><JJ>+<NN.?>+<VBG>?<NN>*(<IN><JJ>+<NN>)?}"""

    chunkparser = RegexpParser(chunkgram)
    chunked = chunkparser.parse(pos_tagger(case_study))
    clin_feat = ''
    for subtree in chunked.subtrees(filter = lambda t: t.label() == 'Chunk'):
        clin_feat += ' |feature: '
        for word, pos in subtree:
            clin_feat += ' '
            if word == 'with':
                continue
            elif word == 'of':
                continue
            else:
                clin_feat += word
    return(clin_feat)


# In[11]:


def combo3(case_study):
    
    chunkgram = r"""Chunk: {<,><CC><JJ>*<NN>{2}}"""

    chunkparser = RegexpParser(chunkgram)
    chunked = chunkparser.parse(pos_tagger(case_study))
    clin_feat = ''
    for subtree in chunked.subtrees(filter = lambda t: t.label() == 'Chunk'):
        clin_feat += ' |feature: '
        for word, pos in subtree:
            clin_feat += ' '
            if word == ',':
                continue
            elif word == 'and':
                continue
            else:
                clin_feat += word
    return(clin_feat)


# In[12]:


def combo4(case_study):    
    
    chunkgram = r"""Chunk: {<NN.?>+<VB.?><CD><NNP>}"""

    chunkparser = RegexpParser(chunkgram)
    chunked = chunkparser.parse(pos_tagger(case_study))
    clin_feat = ''
    for subtree in chunked.subtrees(filter = lambda t: t.label() == 'Chunk'):
        clin_feat += ' |feature: '
        for word, pos in subtree:
            clin_feat += ' '
            clin_feat += word
    return(clin_feat)


# In[13]:


def combo5(case_study):    
    
    chunkgram = r"""Chunk: {<VB.?><NN.?><JJ>}"""

    chunkparser = RegexpParser(chunkgram)
    chunked = chunkparser.parse(pos_tagger(case_study))
    clin_feat = ''
    for subtree in chunked.subtrees(filter = lambda t: t.label() == 'Chunk'):
        clin_feat += ' |feature: '
        for word, pos in subtree:
            clin_feat +=  ' '
            clin_feat += word
    return(clin_feat)


# In[14]:


def combo6(case_study):    
    
    chunkgram = r"""Chunk: {<,><CC><VBG>}"""

    chunkparser = RegexpParser(chunkgram)
    chunked = chunkparser.parse(pos_tagger(case_study))
    clin_feat = ''
    for subtree in chunked.subtrees(filter = lambda t: t.label() == 'Chunk'):
        clin_feat += ' |feature: '
        for word, pos in subtree:
            if word == ',':
                continue
            elif word == 'and':
                continue
            else:
                clin_feat +=  ' '
                clin_feat += word
    return(clin_feat)


# In[15]:


def combo7(case_study):    
    
    chunkgram = r"""Chunk: {<CC><NN.?><IN><JJ>*<NN>}"""

    chunkparser = RegexpParser(chunkgram)
    chunked = chunkparser.parse(pos_tagger(case_study))
    clin_feat = ''
    for subtree in chunked.subtrees(filter = lambda t: t.label() == 'Chunk'):
        clin_feat += ' |feature: '
        for word, pos in subtree:
            if word == 'and':
                continue
            else:
                clin_feat +=  ' '
                clin_feat += word
    return(clin_feat)


# In[16]:


def combo8(case_study):    
    
    chunkgram = r"""Chunk: {<,><CC><DT>?<JJ><NN.?><IN><JJ>*<NN>*<IN>?<PRP.?>?<NN.?>*(<CC><JJ><NN.?>)*}"""

    chunkparser = RegexpParser(chunkgram)
    chunked = chunkparser.parse(pos_tagger(case_study))
    clin_feat = ''
    for subtree in chunked.subtrees(filter = lambda t: t.label() == 'Chunk'):
        clin_feat += ' |feature: '
        for word, pos in subtree:
            if word == ',':
                continue
            elif word == 'and':
                continue
            else:
                clin_feat +=  ' '
                clin_feat += word
    return(clin_feat)


# In[17]:


#This function combines the previous functions and outputs 1 summary of all extracted clinical features from a text


def all_features(case_study):
    features = ''
    if int(age(case_study)[0]) > 0:
        features += 'age: '
        features += str(age(case_study)[0])
    features += ' '
    if gender(case_study).isalpha():
        features += 'gender: '
        features += gender(case_study)
       
    features += combo1(case_study)
    features += combo2(case_study)
    features += combo3(case_study)
    features += combo4(case_study)
    features += combo5(case_study)
    features += combo6(case_study)
    features += combo7(case_study)
    features += combo8(case_study)  
        
        
        
        
        
    return features


# In[19]:


#Let's try it out on unseen text:

text = 'A 42-year-old woman with chronic HIV infection presented with sudden onset of progressive limb weakness, leading to immobility within 4 days. This was preceded by severe abdominal pain, nausea, and vomiting for 2 days and episodes of confusion and agitation.'
all_features(text)

""""Output:'age: 42 gender: woman |feature:   nausea |feature:   chronic HIV infection 
|feature:   sudden onset  progressive limb |feature:  by severe abdominal pain |feature:  vomiting 
|feature:  episodes of confusion' """


# In[20]:


"""Although this test on unseen case studies is promising, it didn't always work well for other cases.
The problem is that if you start creating more rules, the amount of false positives increases (aka generates text that are not clinical features)
One solution could be to start working with named entity recognitions that recognize medical terms and include those in the rules"""

