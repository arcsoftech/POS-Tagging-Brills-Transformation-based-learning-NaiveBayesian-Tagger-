import re
import sys
import timeit
from collections import Counter, defaultdict, OrderedDict
import os
import pandas as pd

# Split word and Tag
def __token_splitter__(x):
    return tuple(x.split("_"))

# Get sentence token
def __get_sentence_token(sentence):
    return sentence.split(" ")

# Get tokens and required parmeter from dataset
def __get_corpus_tokens__(sentences):
    taggedTokens = [x.strip().split(" ") for x in sentences]
    
    sentenceTokens = []
    for x in taggedTokens:
        splittedToken = [__token_splitter__(y) for y in x]

        sentenceTokens.append(splittedToken)
        
    tagWordToken = []
    for x in sentenceTokens:
        for y in x:
            tagWordToken.append((y[0].lower(),y[1].upper()))
    correctSentTags = []
    for sent in sentenceTokens:
        sent_token = " ".join([y[1].upper() for y in sent])
        correctSentTags.append(sent_token)
    tagTokens = []
    for x in sentenceTokens:
        for y in x:
            tagTokens.append(y[1].upper())
    wordTokens = []
    for x in sentenceTokens:
        for y in x:
            wordTokens.append(y[0].lower())

 
    return tagWordToken, correctSentTags ,wordTokens,tagTokens

# Count Bigrams
def __bigram_count_model__(sentences, bModel=None):
    bigrams = []
    if(bModel is None):
        sentence_token = [x.split(" ") for x in sentences]
        flat_list = []
        for x in sentence_token:
            sequences = [x[i:] for i in range(2)]
            bigrams = zip(*sequences)
            flat_list += (bigrams)

        bigrams = Counter(flat_list)
    else:
        token_words = sentences.split(" ")
        sequences = [token_words[i:] for i in range(2)]
        sentbigrams = zip(*sequences)
        bigrams = OrderedDict({token: bModel.get(token) if bModel.get(
            token) is not None else 0 for token in sentbigrams})
    return bigrams

#Unigrams
def __unigram_count_model__(tokens, unigramModel=None):
    unigram = {}
    if(unigramModel is None):
        unigram = Counter(tokens)
    else:
        unigram = OrderedDict({token: unigramModel.get(token)
                               for token in tokens})
    return unigram

# Get bigram probability 
def __get_bigram_probability__(bigram, unigram,TT):
    prob = {}
    i=1
    if(TT):
        i=0
    for key in bigram:
        prob[key]= bigram[key]/unigram[key[i]]
    return prob


def main():
    pd.set_option('expand_frame_repr', False)
    start = timeit.default_timer()
    if len(sys.argv) != 2:
        print(
            "Incorrect Arguments! Correct form as: python NBTagger.py <corpus.txt>")
    else:
        corpus_file_name = sys.argv[1]
        dataset = open(corpus_file_name, "r").read().lower()
######################Training##############################################
        
        sentences = dataset.strip().split("\n")     
        tagWordToken, correctSentTags ,wordTokens , tagTokens = __get_corpus_tokens__(sentences)
        tagBigramCounts = __bigram_count_model__(correctSentTags)
        tagUnigramCount = __unigram_count_model__(tagTokens)
        tagWordUnigramCount = __unigram_count_model__(tagWordToken)
        probWT=__get_bigram_probability__(tagWordUnigramCount,tagUnigramCount,TT=False)
        probTT=__get_bigram_probability__(tagBigramCounts,tagUnigramCount,TT=True)
        WT_Columns= ["P({}|{})".format(key[0],key[1]) for key in probWT]
        TT_Columns= ["P({}|{})".format(key[1],key[0]) for key in probTT]
        wordTagProbDataFrame = pd.DataFrame.from_dict({"Bigram": WT_Columns, "Probability": probWT.values()},
                                                      orient='index').transpose()
    
        TagTagDataFrame = pd.DataFrame.from_dict({"Bigram": TT_Columns, "Probability": probTT.values()},
                                                      orient='index').transpose()
        os.makedirs("output/NaiveBayesian/", exist_ok=True)
        wordTagProbDataFrame.to_csv(r'output/NaiveBayesian/wordTagProbability.csv')
        TagTagDataFrame.to_csv(r'output/NaiveBayesian/tagTagProbability.csv')
        print("\nModels are saved in output folder in its respected folder. Zero probabilities and counts are ignored while computation.")
        
###########################Testing############################################
#        for word in wordToFind:
#            print(word)
#            tagProbability = {k:v for k,
#                                       v in probWT.items() if k[0] == word}
#            probableTags[word]=(tagProbability)
#        seq={}
#        for key in probableTags["standard"].keys():
#            x = "DT,{},NN".format(key[1])
#            probability = probTT[("DT",key[1])]*probWT[key]*probTT[(key[1],"NN")]
#            seq[x] =probability
#        print(seq)
#        print(max(seq, key=seq.get))
#        print("\n For work")
#        seq={}
#        for key in probableTags["work"].keys():
#            x = "JJ,TO,{}".format(key[1])
#            probability = seq[("JJ,TO,{}".format(key[1]))]= probTT[("VBZ","JJ")]*probTT[("JJ","TO")]*probWT[key]
#            seq[x] =probability
#        print(seq)
#        print(max(seq, key=seq.get))            
    stop = timeit.default_timer()
    print('Execution Time: ', stop - start)


if __name__ == '__main__':
    main()
