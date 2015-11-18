import nltk
import numpy as np
from collections import defaultdict
import os
import pickle
import random
import math

class EPParser():
    
    def __init__(self, filename):        
        self.START = ['***START***', '***START2***']
        self.END = ['***END***', '***END2***']
        self.model = EPModel()
        self.trainSentences = []
        self.testSentences = []
        
        if os.path.exists('resources/Saved' + filename):
            with open('resources/Saved' + filename, 'r') as f:
                self.sentences = pickle.load(f)
            with open('resources/EP' + filename, 'r') as f:
                self.allEPtags = pickle.load(f)
        else:
            os.makedirs('resources')
            print 'Saved resource file for ' + filename + ' not found.\nCreating new files'
            f = open(filename, 'r')
            sentences = []
            allEPtags = []
            for line in f:
                words = line.split()
                if words:
                    taggedwords = [w.split('_') for w in words if '_' in w]
                    wordlist= []
                    EPtags = []
                    for w in words:
                        if '_' in w:
                            ws = w.split('_')
                            wordlist.append(ws[0])
                            EPtags.append(ws[1])
                    wordpostags = nltk.pos_tag(wordlist)
                    
                    fullannotations = [['***START***', 'None', 'None'], ['***START2***', 'None', 'None']] + [w + [wordpostags[wordlist.index(w[0])][1]] for w in taggedwords] + [['***END***', 'None', 'None'], ['***END2***', 'None', 'None']]
                    sentences.append(fullannotations)
                    allEPtags.append(EPtags)
            with open('resources/Saved' + filename, 'w') as f:
                 pickle.dump(sentences, f)
            with open('resources/EP' + filename, 'w') as f:
                 pickle.dump(allEPtags, f)
            self.sentences = sentences
            self.allEPtags = allEPtags
        self.shuffle()
        
    def shuffle(self):
        random.shuffle(self.sentences)
        self.trainSentences = self.sentences[:int(math.floor(len(self.sentences) * .8))]
        self.testSentences = self.sentences[int(math.floor(len(self.sentences) * .8)):]
        
    #https://github.com/sloria/textblob-aptagger/blob/master/textblob_aptagger/taggers.py
    def _get_features(self, i, word, context, EPtag, EPtag2, postag0, postag1, postag2):
        '''Map tokens into a feature representation, implemented as a
        {hashable: float} dict. If the features change, a new model must be
        trained.
        '''
        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)
        # It's useful to have a constant feature, which acts sort of like a prior
        add('bias')
        add('i suffix', word[-3:])
        add('i pref1', word[0])
        add('i-1 EPtag', EPtag)
        add('i-2 EPtag', EPtag2)
        add('i-1 EPtag+i-2 EPtag', EPtag, EPtag2)
        add('i word', context[i])
        add('i-1 EPtag+i word', EPtag, context[i])
        add('i-1 postag', postag1)
        add('i-2 postag', postag2)
        add('i postag+i-2 postag', postag1, postag2)
        add('i word', context[i])
        add('i-1 postag+i word', postag1, context[i])
        add('i postag', postag0)
        add('i postag+i word', postag0, context[i])
        add('i postag+i-1 EPtag', postag0, EPtag)
        add('i-1 word', context[i-1])
        add('i-1 suffix', context[i-1][-3:])
        add('i-2 word', context[i-2])
        add('i+1 word', context[i+1])
        add('i+1 suffix', context[i+1][-3:])
        add('i+2 word', context[i+2])
        return features


    def getWordFeatures(self, words):
        for i in words:
            word.append(words[i][0])
            EPtag.append(words[i][1])
            postag.append(words[i][2])
        isWasOrWere = (word[0].lower() == 'was' or word[0].lower() == 'were')
        endsInEDAndIsVBN = (len(word[0]) > 3 and word[0][-2:] == 'ed' and postag[0] == 'VBN')
        previousPOSTagIsDT = (postag[1] == 'DT')

        boolFeature = [isWasOrWere, endsInEDAndIsVBN]
        feature = [int(b) for b in boolFeatures]        
        
    def getFeatures(self, sentences):
        features = []
        for s in sentences:
            context = [w[0] for w in s]
            EPtags = [w[1] for w in s]
            postags = [w[2] for w in s]
            for i in range(len(context) - 4):
                feature = self._get_features(i, context[i+2], context, EPtags[i+1], EPtags[i], postags[i+2], postags[i+1], postags[i])

                features.append(feature)
        return features
        #TODO

    def evaluate(self):
        print 'Evaluating'
        correct = 0
        counter = 0
        sentences = self.testSentences
        outputSentences = []
        for s in sentences:
            output = []
            wordlist = [w[0] for w in s[2:-2]]
            EPtags = [w[1] for w in s[2:-2]]
            postags = [w[2] for w in s[2:-2]]
            prev = "None"
            prev2 = "None"
            context = self.START + wordlist + self.END
            print 'New Sentence:'
            for i in range(len(wordlist)):
                tags = []
                for j in range(3):
                    if i- j >= 0:
                        tags.append(postags[i-j])
                    else:
                        tags.append("None")
                features = self._get_features(i, wordlist[i], context, prev, prev2, tags[0], tags[1], tags[2])
                #print features
                scores = self.model.predict(features)
                #print scores
                maxval = -1000000
                label = 'N'
                for l in scores.keys():
                    if scores[l] > maxval:
                        maxval = scores[l]
                        label = l
                prev2 = prev
                prev = label
                output.append([wordlist[i], label])
                
                print 'Word: ' + wordlist[i] + '-' + 'Predicted/Expected: ' + label + '/' + EPtags[i]
                if label == EPtags[i]:
                    correct = correct + 1
                counter = counter + 1
            outputSentences.append(output)
        print "correct: " + str(float(correct)/counter)
        return outputSentences
        #TODO implement Viterbi

    def train(self, n = 10):
        print 'Training'
        for it in range(n):
            self.shuffle()
            sentences = self.trainSentences
            alltags = []
            for i in sentences:
                for j in i[2:-2]:
                    alltags.append(j[1])
            features = self.getFeatures(sentences)
            for i in range(len(features)):
                feat = features[i]
                pred = self.model.predict(feat)
                if not pred:
                    self.model.update(alltags[i], "None", feat)
                else:
                    maxval = -1000000
                    label = ''
                    for l in pred.keys():
                        if pred[l] > maxval:
                            maxval = pred[l]
                            label = l
                    self.model.update(alltags[i], label, feat)

class EPModel():
    def __init__(self):
        self.weights = {}
        self.labels = ['N', 'I', 'E', 'A', 'P', 'PR', 'R']
        
    def predict(self, features):
        scores = defaultdict(float)
        for f, v in features.items():
            if f not in self.weights or v == 0:
                continue
            weights = self.weights[f]
            for l, w in weights.items():
                scores[l] = scores[l] + v*w

        return scores

    def update(self, real, pred, features):
        if real == pred:
            return None
        else:
            for f in features:
                if f in self.weights:
                    for c in self.labels:
                        if c == real:
                            self.weights[f][c] = self.weights[f][c]+1
                        else:
                            self.weights[f][c] = self.weights[f][c]-1
                else:
                    self.weights[f] = {}
                    for c in self.labels:
                        if c == real:
                            self.weights[f][c] = 1
                        else:
                            self.weights[f][c] = 0
    
    def showWeights(self):
        for f in self.weights.keys():
            for l in self.weights[f].keys():
                print f + '-' + l + ': ' + str(self.weights[f][l])

class Recipe():

    def __init__(self, filename):
        self.parser = EPParser(filename)
        self.parser.train(3)
        self.taggedSentences = self.parser.evaluate()
        self.createRecipe()
        
    def createRecipe(self):
        recipe = []
        for s in self.taggedSentences:
            r = defaultdict(list)
            for w, t in s:
                if t != 'N':
                    r[t].append(w)
            if not r['A']:
                continue
            recipe.append(r)
        print 'Recipe: '
        print recipe
