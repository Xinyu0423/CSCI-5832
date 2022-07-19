import random
import collections
import re
import math
import numpy as np

def gengerateText():
    inputFile = open("generated_training_sets.txt",'r')
    myStr=""
    countLine=0
    countWhen=0
    for line in inputFile:
        if not line.startswith('\n'):
            #line="<s> "+line+"<\s>"
            #print(line)
            #line = re.sub(r'[^\w\s]', '', line).lower()
            #remove all punctuation and lower all words
            #line = "<s> " + line.rstrip('\n') + '</s>' + "\n"
            myStr=myStr+line
            countLine=countLine+1
            '''if "<s> when" in line and "<s> whenever" not in line:
                print(line)
                countWhen=countWhen+1'''
    #print(countWhen)
    counterStr=collections.Counter(myStr.split())
    #print(counterStr)
    print("There are ",countLine," lines in the text")
    countToken=0
    for i in counterStr.values():
        countToken=countToken+i
    print("There are ", countToken , " tokens in the text")
    countType=len(counterStr)
    print("There are ", countType , " types in the text")
    #for line in inputFile:
        #if "when" in line:
    #    print(line)
    #print(123)
    return counterStr,countLine,countToken,countType


def trainWords(counterStr):
    inputTraFile=open("generated_training_sets.txt",'r')
    myStr=""
    for line in inputTraFile:
        myStr=myStr+line
    words=myStr.split()
    two_words = [' '.join(ws) for ws in zip(words, words[1:])]
    wordscount = {w: f for w, f in collections.Counter(two_words).most_common()}
    #get from https://stackoverflow.com/questions/51949681/counting-the-frequency-of-three-words
    return wordscount

def sampleSent(wordscount):
    sentence="<s>"
    curWrod="<s>"

    twoWordsList=[]
    for i in wordscount.keys():
        tempList=i.split()
        twoWordsList.append(tempList)
    #print(twoWordsList)
    while curWrod!="</s>":
        possibleWords=[]
        for i in twoWordsList:
            #print(firstWord)
            #print(i[0])
            if i[0]==curWrod:
                possibleWords.append(i[1])
        addedWord=np.random.choice(np.array(possibleWords))
        if addedWord!="</s>":
            sentence=sentence+" "+addedWord
        elif addedWord=="</s>":
            sentence=sentence+" </s>"
        curWrod=addedWord
    #print(wordscount)
    print(sentence)
    return sentence

def computeProb(sentence,wordscount):
    inputTraFile = open("generated_training_sets.txt", 'r')
    sentenceList=sentence.split()
    sentenceLen=len(sentence)-2
    twoWordList=[]
    two_words =list(map(' '.join, zip(sentenceList[:-1], sentenceList[1:])))
    twoWordList.append(two_words)
    #print(twoWordList)
    prob=0
    for i in twoWordList:
        tempProb = []
        for j in i:
            if j in wordscount.keys():
                firstEle = j.split()[0]
                tempProb.append((wordscount[j]+1) / (counterStr[firstEle]+sentenceLen))
            #print(wordscount[j] / counterStr[firstEle])
            else:
                firstEle = j.split()[0]
                tempProb.append(1 / (counterStr[firstEle]+sentenceLen[0]))
        tempResult = 1.0
        for x in tempProb:
            tempResult = tempResult * x
        prob=tempResult
    #print(prob)
    probNorm=math.exp(math.log(prob) / sentenceLen)
    probPerplexity=1/probNorm
    print("The probability for the sample sentence is "+str(prob))
    print("The probability normalized by sentence length is "+str(probNorm))
    print("The perplexity for the sample sentence is "+str(probPerplexity))

counterStr,countLine,countToken,countType=gengerateText()

for i in range(0,5):
    wordscount = trainWords(counterStr)
    sentence=sampleSent(wordscount)
    computeProb(sentence,wordscount)