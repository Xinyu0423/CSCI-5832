import collections
import re
import math

def addS():
    inputFile = open("hw2_training_sets.txt", 'r')
    generatedFile=open("generated_training_sets.txt",'w')
    for line in inputFile:
        if not line.startswith('\n'):
            line = re.sub(r'[^\w\s]', '', line).lower()
            # remove all punctuation and lower all words
            line = "<s> "+line.rstrip('\n') + '</s>'+"\n"
            generatedFile.write(line)

    inputTestFile = open("test_set.txt", 'r')
    generatedTestFile = open("generated_test_sets.txt", 'w')
    for line in inputTestFile:
        if not line.startswith('\n'):
            line = re.sub(r'[^\w\s]', '', line).lower()
            # remove all punctuation and lower all words
            line = "<s> " + line.rstrip('\n') + '</s>' + "\n"
            generatedTestFile.write(line)
    inputFile.close()
    generatedTestFile.close()

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


def computeProb(counterStr):
    inputTraFile=open("generated_training_sets.txt",'r')
    inputTestFile=open("generated_test_sets.txt",'r')
    sentence1=[]
    sentence2=[]
    sentence3=[]
    sentence4=[]
    sentence5=[]
    count=0
    sentenceLen=[]
    for line in inputTestFile:
        if count==0:
            lineList=line.split()
            sentenceLen.append(len(lineList)-2)
            two_words =list(map(' '.join, zip(lineList[:-1], lineList[1:])))
            sentence1.append(two_words)
        if count==1:
            lineList=line.split()
            sentenceLen.append(len(lineList)-2 )
            two_words =list(map(' '.join, zip(lineList[:-1], lineList[1:])))
            sentence2.append(two_words)
        if count==2:
            lineList=line.split()
            sentenceLen.append(len(lineList)-2)
            two_words =list(map(' '.join, zip(lineList[:-1], lineList[1:])))
            sentence3.append(two_words)
        if count==3:
            lineList=line.split()
            sentenceLen.append(len(lineList)-2 )
            two_words =list(map(' '.join, zip(lineList[:-1], lineList[1:])))
            sentence4.append(two_words)
        if count==4:
            lineList=line.split()
            sentenceLen.append(len(lineList)-2)
            two_words =list(map(' '.join, zip(lineList[:-1], lineList[1:])))
            sentence5.append(two_words)
        count=count+1
    #print(sentence1)
    #print(sentence4)
    myStr=""
    for line in inputTraFile:
        myStr=myStr+line
    #words = re.findall(r'\w+',myStr)
    words=myStr.split()
    two_words = [' '.join(ws) for ws in zip(words, words[1:])]
    #print(two_words)
    wordscount = {w: f for w, f in collections.Counter(two_words).most_common()}
    #print(wordscount)
    #get from https://stackoverflow.com/questions/51949681/counting-the-frequency-of-three-words
    probSentence=[]

    for i in sentence1:
        tempProb = []
        for j in i:
            if j in wordscount.keys():
                firstEle = j.split()[0]
                tempProb.append((wordscount[j]+1) / (counterStr[firstEle]+sentenceLen[0]))
            #print(wordscount[j] / counterStr[firstEle])
            else:
                firstEle = j.split()[0]
                tempProb.append(1 / (counterStr[firstEle]+sentenceLen[0]))
        tempResult = 1.0
        for x in tempProb:
            tempResult = tempResult * x
        probSentence.append(tempResult)

    for i in sentence2:
        tempProb = []
        for j in i:
            if j in wordscount.keys():
                firstEle = j.split()[0]
                tempProb.append((wordscount[j]+1) / (counterStr[firstEle]+sentenceLen[1]))
            #print(wordscount[j] / counterStr[firstEle])
            else:
                firstEle = j.split()[0]
                tempProb.append(1 / (counterStr[firstEle]+sentenceLen[1]))
        tempResult = 1.0
        for x in tempProb:
            tempResult = tempResult * x
        probSentence.append(tempResult)

    for i in sentence3:
        tempProb = []
        for j in i:
            if j in wordscount.keys():
                firstEle = j.split()[0]
                tempProb.append((wordscount[j]+1) / (counterStr[firstEle]+sentenceLen[2]))
            #print(wordscount[j] / counterStr[firstEle])
            else:
                firstEle = j.split()[0]
                tempProb.append(1 / (counterStr[firstEle]+sentenceLen[2]))
        tempResult = 1.0
        for x in tempProb:
            tempResult = tempResult * x
        probSentence.append(tempResult)

    for i in sentence4:
        tempProb = []
        for j in i:
            if j in wordscount.keys():
                firstEle = j.split()[0]
                tempProb.append((wordscount[j]+1) / (counterStr[firstEle]+sentenceLen[3]))
            #print(wordscount[j] / counterStr[firstEle])
            else:
                firstEle = j.split()[0]
                tempProb.append(1 / (counterStr[firstEle]+sentenceLen[3]))
        tempResult = 1.0
        for x in tempProb:
            tempResult = tempResult * x
        probSentence.append(tempResult)

    for i in sentence5:
        tempProb = []
        for j in i:
            if j in wordscount.keys():
                firstEle = j.split()[0]
                tempProb.append((wordscount[j]+1) / (counterStr[firstEle]+sentenceLen[4]))
            #print(wordscount[j] / counterStr[firstEle])
            else:
                firstEle = j.split()[0]
                tempProb.append(1 /(counterStr[firstEle]+sentenceLen[4]))
        tempResult = 1.0
        for x in tempProb:
            tempResult = tempResult * x
        probSentence.append(tempResult)

    print(probSentence)

    normList=[]
    perpleList=[]
    for i in range(0,len(probSentence)):
        tempNorm=math.exp(math.log(probSentence[i]) / sentenceLen[i])
        normList.append(tempNorm)

    print(normList)
    for i in normList:
        tempPerple=1/i
        perpleList.append(tempPerple)
    print(perpleList)
    #print(counterStr)
    #print(wordscount)
    return probSentence,normList,perpleList

def writeOutput(probSentence,normList,perpleList):
    outputFile=open("output.txt",'w')
    for i in range(0,len(probSentence)):
        outputFile.write(str(probSentence[i])+", "+str(normList[i])+", "+str(perpleList[i])+"\n")




addS()
counterStr,countLine,countToken,countType=gengerateText()
#print(type(countToken))

probSentence,normList,perpleList=computeProb(counterStr)
writeOutput(probSentence,normList,perpleList)
