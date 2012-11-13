# -*- coding: utf-8 -*-
# Name:Sumaiya Hashmi
# Date: 10/1/12
# Description: a *new and improved* Naive Bayes text sentiment classifier!
# training data should be in a folder titled 'reviews'
# in the same parent directory as sentimentalanyzer.py
# positive reviews should contain movies-5 in file name; negative should contain movies-1 
#


import math, os, pickle, re, random
from pprint import *

class Bayes_Classifier:

   
   def __init__(self):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text."""


      self.numPosFeatures = 0
      self.numNegFeatures = 0
      self.priorPPos = 0
      self.priorPNeg = 0
      
      # set file names for the dictionaries
      self.posFile = "posDictFile.txt"
      self.negFile = "negDictFile.txt"
      self.posBiFile = "posBiFile.txt"
      self.negBiFile = "negBiFile.txt"
      self.posTriFile = "posTriFile.txt"
      self.negTriFile = "negTriFile.txt"
      
      # initialize the two dictionaries
      self.posDict = {}
      self.negDict = {}
      # initialize the bigram dictionaries
      self.posBi = {}
      self.negBi = {}
      # initialize the trigram dictionaries
      self.posTri = {}
      self.negTri = {}

      # get names of all files in review directory
      # this is used in both train and test, so let's keep it in init.
      self.lFileList = [] 
      for fFileObj in os.walk("reviews2/"): 
         self.lFileList = fFileObj[2] 
         break

      
      # if pickled files already exist, load them into memory
      if os.path.exists(self.posFile) and os.path.exists(self.negFile) and \
         os.path.exists(self.posBiFile) and os.path.exists(self.negBiFile) and \
         os.path.exists(self.posTriFile) and os.path.exists(self.negTriFile):
         self.posDict = self.load(self.posFile)
         self.negDict = self.load(self.negFile)
         self.posBi = self.load(self.posBiFile)
         self.negBi = self.load(self.negBiFile)
         self.posTri = self.load(self.posTriFile)
         self.negTri = self.load(self.negTriFile)
         
      else: # otherwise, create them
         self.train()

   
   def train(self):   
      """Trains the Naive Bayes Sentiment Classifier."""

      for fileName in self.lFileList:

         self.trainHelper(fileName)
                  
      # save dictionaries using pickle     
      self.save(self.negDict, self.negFile)
      self.save(self.posDict, self.posFile)
      self.save(self.negBi, self.negBiFile)
      self.save(self.posBi, self.posBiFile)
      self.save(self.negTri, self.negTriFile)
      self.save(self.posTri, self.posTriFile)

                              
   def trainHelper(self, fileName):
      """ does the training """
         # get the contents of the file as list of tokens in the string
      text = self.loadFile("reviews/" + fileName) 
      tokens = self.tokenize(text)
      num = len(tokens) # number of words in the review
      # determine if negative or positive review, based on filename
      if "movies-5" in fileName:
         for token in tokens:
            # add tokens to the correct unigram dictionary
            # increment to get word frequency
            if not token in self.posDict:
               self.posDict[token] = 1
            else:
               self.posDict[token] +=1

         # add tokens to the bigram dictionary
         for x in range(num-1):
            bigram = tokens[x] + " " + tokens[x+1]
            if not bigram in self.posBi:
               self.posBi[bigram] = 1
            else:
               self.posBi[bigram] += 1
               
         # add tokens to the trigram dictionary
         for x in range(num-2):
            trigram = tokens[x] + " " + tokens[x+1] + " " + tokens[x+2]
            if not trigram in self.posTri:
               self.posTri[trigram] = 1
            else:
               self.posTri[trigram] += 1

      if "movies-1" in fileName:
         for token in tokens:
            if not token in self.negDict:
               self.negDict[token] = 1
            else:
               self.negDict[token] += 1
         for x in range(num-1):
            bigram = tokens[x] + " " + tokens[x+1]
            if not bigram in self.negBi:
               self.negBi[bigram] = 1
            else:
               self.negBi[bigram] += 1
         for x in range(num-2):
            trigram = tokens[x] + " " + tokens[x+1] + " " + tokens[x+2]
            if not trigram in self.negTri:
               self.negTri[trigram] = 1
            else:
               self.negTri[trigram] += 1

   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """
      # to get prior probabilities of a document being neg or pos,
      # count the numbers of pos, neg, and total docs
      numDocs = 0
      numPos = 0
      numNeg = 0

      # need to go through the files to count numbers of docs
      lFileList = [] 
      for fFileObj in os.walk("reviews/"): 
         lFileList = fFileObj[2] 
         break

      for fileName in lFileList:
         numDocs += 1
         if "movies-5" in fileName:
            numPos += 1
         if "movies-1" in fileName:
            numNeg +=1
                  
      # calculate prior probabilities of doc being pos or neg
      self.priorPPos = numPos/float(numDocs) 
      self.priorPNeg = numNeg/float(numDocs)

      # count how many pos and neg features there are
      # first count the unigram features
      for key in self.posDict.keys():
         self.numPosFeatures += self.posDict[key]
      for key in self.negDict.keys():
         self.numNegFeatures += self.negDict[key]
      # then count the bigram features
      for key in self.posBi.keys():
         self.numPosFeatures += self.posBi[key]
      for key in self.negBi.keys():
         self.numNegFeatures += self.negBi[key]
      # then count the trigram features
      for key in self.posTri.keys():
         self.numPosFeatures += self.posTri[key]
      for key in self.negTri.keys():
         self.numNegFeatures += self.negTri[key]   

      
      getUnigrams = self.unigram(sText)
      unigramSummationPos = getUnigrams[0]
      unigramSummationNeg = getUnigrams[1]

      getBigrams = self.bigram(sText)
      bigramSummationPos = getBigrams[0]
      bigramSummationNeg = getBigrams[1]

      getTrigrams = self.trigram(sText)
      trigramSummationPos = getTrigrams[0]
      trigramSummationNeg = getTrigrams[1]

      
      # add log of prior probability
      pDocPos = math.log(self.priorPPos) + unigramSummationPos + \
                bigramSummationPos + trigramSummationPos
      pDocNeg = math.log(self.priorPNeg) + unigramSummationNeg + \
                bigramSummationNeg + trigramSummationPos
 
      # compare probabilities of input string being pos or neg
      if pDocPos > pDocNeg:
         return "positive"
      elif pDocPos < pDocNeg:
         return "negative"
      else: # (keeping it simple)
         return "neutral"

   def unigram(self, sText):
      """ given target text, return tuple
         (probability that it is positive, probability that it is negative),
         considering unigrams as features """
      # need to sum the logs of the feature conditional probabilities
      pSummationPos = 0
      pSummationNeg = 0

      # go through each word in the input string
      tokens = self.tokenize(sText)

      for token in tokens:
         if not token in self.posDict:
            # make unknown words equally likely to be pos or neg:
            pFeaturePos = 1/(float((self.numPosFeatures + self.numNegFeatures)/2))
         
         if not token in self.negDict:
            pFeatureNeg = 1/(float((self.numPosFeatures + self.numNegFeatures)/2))
            # pFeatureNeg = math.pow(float(self.priorPPos), 1/float(unknownN))
  
         if token in self.posDict:
            # p(token | positive)
            pFeaturePos = (self.posDict[token] + 1)/float(self.numPosFeatures)
            
         if token in self.negDict:
            # p(token | negative)
            pFeatureNeg = (self.negDict[token] + 1)/float(self.numNegFeatures)
  

         # add log of the conditional probability to the summation
         pSummationPos += math.log(pFeaturePos)
         pSummationNeg += math.log(pFeatureNeg)

      return pSummationPos, pSummationNeg

   def bigram(self, sText):
      """ given target text, return tuple
         (probability that it is positive, probability that it is negative),
         considering bigrams as features """

      # need to sum the logs of the feature conditional probabilities
      pSummationPos = 0
      pSummationNeg = 0

      # go through each word in the input string
      tokens = self.tokenize(sText)
      num = len(tokens)
      for x in range(num-1):
         bigram = tokens[x] + " " + tokens[x+1]
         if not bigram in self.posBi:
            pFeaturePos = 1/(float((self.numPosFeatures + self.numNegFeatures)/2))
          
         if not bigram in self.negBi:
            pFeatureNeg = 1/(float((self.numPosFeatures + self.numNegFeatures)/2))

         if bigram in self.posBi:
            # p(token | positive)
            pFeaturePos = (self.posBi[bigram] + 1)/float(self.numPosFeatures)

         if bigram in self.negBi:
            # p(token | negative)
            pFeatureNeg = (self.negBi[bigram] + 1)/float(self.numNegFeatures)

         # add log of the conditional probability to the summation
         pSummationPos += math.log(pFeaturePos)

         pSummationNeg += math.log(pFeatureNeg)

      return pSummationPos, pSummationNeg

   def trigram(self, sText):
      """ given target text, return tuple
         (probability that it is positive, probability that it is negative),
         considering trigrams as features """
      # need to sum the logs of the feature conditional probabilities
      pSummationPos = 0
      pSummationNeg = 0

      # go through each word in the input string
      tokens = self.tokenize(sText)
      num = len(tokens)
      for x in range(num-2):
         trigram = tokens[x] + " " + tokens[x+1] + " " + tokens[x+2]
         if not trigram in self.posTri:
            # add-one smoothing
            pFeaturePos = 1/(float((self.numPosFeatures + self.numNegFeatures)/2))
         if not trigram in self.negTri:
            # add-one smoothing
            pFeatureNeg = 1/(float((self.numPosFeatures + self.numNegFeatures)/2))
         if trigram in self.posTri:
            # p(token | positive)
            pFeaturePos = (self.posTri[trigram] + 1)/float(self.numPosFeatures)
         if trigram in self.negTri:
            # p(token | negative)
            pFeatureNeg = (self.negTri[trigram] + 1)/float(self.numNegFeatures)
    
         # add log of the conditional probability to the summation
         pSummationPos += math.log(pFeaturePos)

         pSummationNeg += math.log(pFeatureNeg)

      return pSummationPos, pSummationNeg

   def loadFile(self, sFilename):
      """Given a file name, return the contents of the file as a string."""

      f = open(sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt
   
   def save(self, dObj, sFilename):
      """Given an object and a file name, write the object to the file using pickle."""

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()
   
   def load(self, sFilename):
      """Given a file name, load and return the object stored in the file."""

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText): 
      """Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order)."""

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))
               
      if sToken != "":
         lTokens.append(sToken)

      return lTokens

   def test(self):
       """ runs a 10 fold cross validation test of the classifier """
       # initialize precision, recall, f measure values
       count = 0
       ppTotal = 0
       prTotal = 0
       pfTotal = 0
       npTotal = 0
       nrTotal = 0
       nfTotal = 0
              
       end = len(self.lFileList)
       increment = end/10

       #shuffle the files
       random.shuffle(self.lFileList)

       # iterate through the 10 buckets  
       for n in range(0, end, increment):
           print n, end
           # reset dictionaries
           self.posDict = {}
           self.negDict = {}
           self.posBi = {}
           self.negBi = {}
           self.posTri = {}
           self.negTri = {}
           # train on 9/10 of the files
           for fileName in (self.lFileList[0:n] + self.lFileList[n+increment:end]):
               self.trainHelper(fileName)
           # save the dictionaries
           self.save(self.negDict, self.negFile)
           self.save(self.posDict, self.posFile)
           self.save(self.negBi, self.negBiFile)
           self.save(self.posBi, self.posBiFile)
           self.save(self.negTri, self.negTriFile)
           self.save(self.posTri, self.posTriFile)
           # test on 1/10 of the files
           # true/false positives/negatives, from perspective of Pos and Neg reviews
           # add one to avoid divide by zero errors  
           TPP = 1
           TNP = 1
           FPP = 1
           FNP = 1
           TPN = 1
           TNN = 1
           FPN = 1
           FNN = 1
           for fileName in self.lFileList[n:n+increment]:
               # only test pos or neg review files
               if ('movies-1' in fileName) or ('movies-5' in fileName):
                  # get numbers of true/false negatives/positives
                  results = self.testHelper(fileName)
                  TPP += results[0]
                  TNP += results[1]
                  FPP += results[2]
                  FNP += results[3]
                  TPN += results[4]
                  TNN += results[5]
                  FPN += results[6]
                  FNN += results[7]
           #print 'tp', TPP, 'tn', TNP, 'fp', FPP, 'fn', FNP

           # calculate accuracy, precision, recall, f measure for each train/test combo
           pAccuracy = (TPP + TNP)/float(TPP + TNP + FPP + FNP)
           pPrecision = TPP/float(TPP + FPP)
           pRecall = TPP/float(TPP + FNP)
           pF = (2*pPrecision*pRecall)/float(pPrecision + pRecall)

           nAccuracy = (TPN + TNN)/float(TPN + TNN + FPN + FNN)
           nPrecision = TPN/float(TPN + FPN)
           nRecall = TPN/float(TPN + FNN)
           nF = (2*nPrecision*nRecall)/float(nPrecision + nRecall)

           #print 'acc', accuracy, 'pre', precision, 'rec', recall, 'f', f

           # add it to the running total
           ppTotal += pPrecision
           prTotal += pRecall
           pfTotal += pF
           npTotal += nPrecision
           nrTotal += nRecall
           nfTotal += nF
           count += 1
               
       # get averages of precision, recall, f score        
       ppAvg = ppTotal/float(count)
       prAvg = prTotal/float(count)
       pfAvg = pfTotal/float(count)
       npAvg = npTotal/float(count)
       nrAvg = nrTotal/float(count)
       nfAvg = nfTotal/float(count)
       
       print "Positive: pAvg, rAvg, fAvg", ppAvg, prAvg, pfAvg
       print "Negative: pAvg, rAvg, fAvg", npAvg, nrAvg, nfAvg
       
   def testHelper(self, fileName):
      """ helper for the test function"""
      # everything starts at 0
      FPP = 0
      TPP = 0
      FNP = 0
      TNP = 0
      FPN = 0
      TPN = 0
      FNN = 0
      TNN = 0
      expected = 'null'
      if 'movies-1' in fileName:
        expected = 'negative'
      if 'movies-5' in fileName:
        expected = 'positive'
      text = self.loadFile("reviews/" + fileName)
      actual = self.classify(text)
      if actual == expected:
        if actual == 'positive': # expected P, got P
            TPP += 1
            TNN += 1
        if actual == 'negative': # expected N, got N        
            TNP += 1
            TPN += 1
      elif actual == 'positive': # expected N, got P
        FNN += 1
        FPP += 1
      elif actual == 'negative': # expected P, got N
        FNP += 1
        FPN += 1
      return TPP, TNP, FPP, FNP, TPN, TNN, FPN, FNN

   def countTrigrams(self):
      singleCountN = 0
      totalN = 0
      singleCountP = 0
      totalP = 0
      for key in self.posTri.keys():
         totalP += 1
         if self.posTri[key] == 1:
            singleCountP += 1
      for key in self.negTri.keys():
         totalN += 1
         if self.negTri[key] == 1:
            singleCountN += 1
      print "N single ", singleCountN/float(totalN)
      print "P single", singleCountP/float(totalP)

   def countBigrams(self):
      singleCountN = 0
      totalN = 0
      singleCountP = 0
      totalP = 0
      for key in self.posBi.keys():
         totalP += 1
         if self.posBi[key] == 1:
            singleCountP += 1
      for key in self.negBi.keys():
         totalN += 1
         if self.negBi[key] == 1:
            singleCountN += 1
      print "N single ", singleCountN/float(totalN)
      print "P single", singleCountP/float(totalP)
   def countUnigrams(self):
      singleCountN = 0
      totalN = 0
      singleCountP = 0
      totalP = 0
      for key in self.posDict.keys():
         totalP += 1
         if self.posDict[key] == 1:
            singleCountP += 1
      for key in self.negDict.keys():
         totalN += 1
         if self.negDict[key] == 1:
            singleCountN += 1
      print "N single ", singleCountN/float(totalN)
      print "P single", singleCountP/float(totalP)
