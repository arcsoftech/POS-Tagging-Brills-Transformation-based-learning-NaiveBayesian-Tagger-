import re
import sys
import timeit
import os
from collections import Counter, defaultdict

import pandas as pd

# Split Word and Tag
def __token_splitter__(x):
    return tuple(x.split("_"))

# Split sentence into token
def __get_sentence_token(sentence):
    return sentence.split(" ")

# Generate required tokens from the given dataset
def __get_corpus_tokens__(sentences):
    taggedTokens = [x.strip().split(" ") for x in sentences]
    
    sentenceTokens = []
    for x in taggedTokens:
        splittedToken = [__token_splitter__(y) for y in x]

        sentenceTokens.append(splittedToken)
    correctSentWords = []
    for sent in sentenceTokens:
        sent_token = [y[0].lower() for y in sent]
        correctSentWords.append(sent_token)
    correctSentTags = []
    for sent in sentenceTokens:
        sent_token = [y[1].upper() for y in sent]
        correctSentTags.append(sent_token)
    combinedUnigram = [item for sublist in taggedTokens for item in sublist]
          
    unigram_pos_list = defaultdict(list)
    for token in combinedUnigram:
        unigram = re.split("_",token)
        unigram_pos_list[unigram[0]].append(unigram[1].upper())
    return correctSentWords, correctSentTags, unigram_pos_list


# Get tag probability
def __get_tag_probability__(tokenCount, taggedTokenCount):
    prob = {}
    for key in taggedTokenCount:
        prob[key] = taggedTokenCount[key] / tokenCount[key[0]]
    return prob

# Find most probable tag for each word
def __get_most_probable_tag(unigram_pos_list):
    probableTags = defaultdict(str)
    for unigram in unigram_pos_list.keys():
        tag = max(unigram_pos_list[unigram],key=unigram_pos_list[unigram].count)
        probableTags[unigram] = tag
    return probableTags


# Brills Tagger- Transformation based learning 
def __tbl__(probableTagsForWords, correctSentWords, correctSentTags,epoch):
    
    currentSentTags = __get_first_currentTags__(probableTagsForWords,correctSentWords)
    transforms_queue = []
    join = [item for sublist in correctSentTags for item in sublist]
    uniqueTags=list(set(join))
    i =10000
    a=0
    try:
        i = int(epoch)
    except:
        pass
    if  i == 10000:
        print("No epoch entered. Rules generation will take approximately 29 min . It will terminate whenever no positive improvements are found.")
    while i>0:
        a+=1
        print("Iteration {}\n".format(a))
        i-=1
        bestRule = __get_best_instance__(uniqueTags,currentSentTags, correctSentTags)
        if (bestRule == -1):
            return transforms_queue
        transforms_queue.append(bestRule)
        currentSentTags = __apply_transformation__(bestRule, currentSentTags)

    return transforms_queue

# Replace word tag with their apropriate tags
def __get_first_currentTags__(probableTagsForWords, correctSentWords):
    currentModifiedSentTags =[]
    for sent in correctSentWords:
        modTags=[probableTagsForWords[word] for word in sent]
        currentModifiedSentTags.append(modTags)
    return currentModifiedSentTags

# Apply transformations on the dataset
def __apply_transformation__(bestRule, currentSentTag):
    for i in range(currentSentTag.__len__()):
        for j in range(1,currentSentTag[i].__len__()):
            if currentSentTag[i][j-1] == bestRule["prev"] and currentSentTag[i][j] == bestRule["from"]:
                currentSentTag[i][j] = bestRule["to"]
    return currentSentTag

#Get best instance
def __get_best_instance__(uniqueTags,currentSentTags, correctSentTags):
    
    
    bestScore=0
    bestRule={}
    for fr in uniqueTags:
        for to in uniqueTags:
            score = defaultdict(lambda:0)
            if fr == to:
                continue
            for i in range(correctSentTags.__len__()):
                for pos in range(1,correctSentTags[i].__len__()):
                    if (correctSentTags[i][pos] == to and currentSentTags[i][pos] == fr):
                        rule = currentSentTags[i][pos - 1]
                        score[rule] += 1
                    if (correctSentTags[i][pos] == fr and currentSentTags[i][pos] == fr):
                        rule = currentSentTags[i][pos - 1]
                        score[rule] -= 1
            if bool(score):
                argmax=max(score, key=score.get)
                if score[argmax] > bestScore:
                    bestScore = score[argmax]
                    bestRule={"prev":argmax,"from":fr,"to":to,"score":bestScore}
  
    if bool(bestRule):
        return bestRule
    else:
        return -1

def main():
    pd.set_option('expand_frame_repr', False)
    start = timeit.default_timer()
    if len(sys.argv) != 3:
        print("Incorrect Arguments! Correct form as: python BrillsTagger.py <corpus.txt> <Epoc>")
    else:
        corpus_file_name = sys.argv[1]
        epoch = sys.argv[2]
        dataset = open(corpus_file_name, "r").read().lower()
        sentences = dataset.strip().split("\n")     
        correctSentWords, correctSentTags,unigram_pos_list = __get_corpus_tokens__(sentences)
        probableTagsForWords = __get_most_probable_tag(unigram_pos_list)
        transforms_queue = __tbl__(probableTagsForWords, correctSentWords, correctSentTags,epoch)
        transformsDataFrame = pd.DataFrame.from_dict(transforms_queue).sort_values(by=['score'],ascending=False)[["prev","from","to","score"]]
        print(transformsDataFrame)
        os.makedirs("output/BrillsTagger/", exist_ok=True)
        transformsDataFrame.to_csv("output/BrillsTagger/transformationRules.csv")

    stop = timeit.default_timer()
    print('Execution Time: ', stop - start)


if __name__ == '__main__':
    main()
