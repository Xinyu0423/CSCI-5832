from nltk.tokenize import word_tokenize
import collections
def countTokenType():
    inputFile=open("hw1_training_sets.txt").read()
    token=word_tokenize(inputFile)
    #print(len(token))
    countToken=len(token)
    countType = collections.Counter(token)
    #print(len(countType))

    outputFile=open("output.txt",'w')
    outputFile.write(str(countToken))
    outputFile.write("\n")
    outputFile.write(str(len(countType)))
    outputFile.write("\n")
    outputFile.write("\n")
    commFive=countType.most_common(5)
    keyList=[]
    valueList=[]
    for i in range(0,5):
        keyList.append(commFive[i][0])
        valueList.append(commFive[i][1])
    #print(keyList)
    #print(valueList)
    for i in range(0,5):
        outputFile.write(keyList[i]+" "+str(valueList[i]))
        if i==4:
            break
        outputFile.write("\n")
    outputFile.close()

countTokenType()