from os import listdir
import nltk
import numpy as np
from collections import defaultdict
import os
import pickle
import random
import math
import codecs

class EPParser():
    
    def __init__(self):        
        self.START = ['***START***', '***START2***']
        self.END = ['***END***', '***END2***']
        self.labels = ['N', 'I', 'E', 'A', 'P', 'PR', 'R']
        self.model = EPModel(self.labels)
        self.samplefiles = [filename for filename in listdir('samples') if 'train' in filename]

        self.trainSentences = []
        self.testSentences = []
        self.Docs = []
        self.trainDocs = []
        self.testDocs = []
        self.trainFilenames = []
        self.testFilenames = []
        
        for filename in self.samplefiles:
            if os.path.exists('resources/Saved' + filename):
                print 'Saved resource file for ' + filename + ' found.'
                with open('resources/Saved' + filename, 'r') as f:
                    sentences = pickle.load(f)
                with open('resources/EP' + filename, 'r') as f:
                    allEPtags = pickle.load(f)
            else:
                if not os.path.exists('resources'):
                    os.makedirs('resources')
                print 'Saved resource file for ' + filename + ' not found.\nCreating new files'
                f = codecs.open('samples/' + filename, encoding = 'utf-8')
                sentences = []
                allEPtags = []
                for line in f:
                    words = line.split()
                    if words:
                        taggedwords = []
                        wordlist= []
                        EPtags = []
                        for w in words:
                            if '_' in w:
                                ws = w.split('_')
                                if ws[1] not in self.labels:
                                    for i in self.labels:
                                        if i in ws[1]:
                                            ws[1] = i
                                    if ws[1] not in self.labels:
                                        ws[1] = 'N'
                                wordlist.append(ws[0].encode('ascii', 'ignore'))
                                EPtags.append(ws[1].encode('ascii', 'ignore'))
                                taggedwords.append([ws[0].encode('ascii', 'ignore'), ws[1].encode('ascii', 'ignore')])
                        print taggedwords
                        wordpostags = nltk.pos_tag(wordlist)
                        
                        fullannotations = [['***START***', 'None', 'None'], ['***START2***', 'None', 'None']] + [w + [wordpostags[wordlist.index(w[0])][1]] for w in taggedwords] + [['***END***', 'None', 'None'], ['***END2***', 'None', 'None']]
                        sentences.append(fullannotations)
                        allEPtags.append(EPtags)
                with open('resources/Saved' + filename, 'w') as f:
                     pickle.dump(sentences, f)
                with open('resources/EP' + filename, 'w') as f:
                     pickle.dump(allEPtags, f)
            self.Docs.append(sentences)
        self.trainDocs = self.Docs[:int(math.floor(len(self.Docs) * .85))]
        self.testDocs = self.Docs[int(math.floor(len(self.Docs) * .85)):]
        self.trainFilenames = self.samplefiles[:int(math.floor(len(self.Docs) * .85))]
        self.testFilenames = self.samplefiles[int(math.floor(len(self.Docs) * .85)):]
        
    def shuffle(self):
        print 'Shuffling data sets'
        idx = range(len(self.Docs))
        random.shuffle(idx)

        tempDocs = []
        tempfiles = []
        
        for i in idx:
            tempDocs.append(self.Docs[i])
            tempfiles.append(self.samplefiles[i])

        self.Docs = tempDocs
        self.samplefiles = tempfiles

        self.trainDocs = self.Docs[:int(math.floor(len(self.Docs) * .85))]
        self.testDocs = self.Docs[int(math.floor(len(self.Docs) * .85)):]
        self.trainFilenames = self.samplefiles[:int(math.floor(len(self.Docs) * .85))]
        self.testFilenames = self.samplefiles[int(math.floor(len(self.Docs) * .85)):]
        
    #https://github.com/sloria/textblob-aptagger/blob/master/textblob_aptagger/taggers.py
    def _get_features(self, i, word, context, EPtag, EPtag2, postag0, postag1, postag2):
        '''Map tokens into a feature representation, implemented as a
        {hashable: float} dict. If the features change, a new model must be
        trained.
        '''
        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1
        c = context[i]
        cm1 = context[i-1]
        cm2 = context[i-2]
        cp1 = context[i+1]
        cp2 = context[i+2]
        
        if word[0].isdigit():
            word = '!NUM'
        if context[i][0].isdigit():
            c = '!NUM'
        if context[i-1][0].isdigit():
            cm1 = '!NUM'
        if context[i-2][0].isdigit():
            cm2 = '!NUM'
        if context[i+1][0].isdigit():
            cp1 = '!NUM'
        if context[i+2][0].isdigit():
            cp2 = '!NUM'
        i += len(self.START)
        features = defaultdict(int)
        # It's useful to have a constant feature, which acts sort of like a prior
        add('bias')
        #add('i suffix', word[-3:])
        #add('i pref1', word[0])
        add('i-1 EPtag', EPtag)
        add('i-2 EPtag', EPtag2)
        add('i-1 EPtag+i-2 EPtag', EPtag, EPtag2)
        add('i word', c)
        add('i-1 EPtag+i word', EPtag, c)
        add('i postag', postag0)
        add('i-1 postag', postag1)
        add('i-2 postag', postag2)
        add('i postag+i-1 postag', postag0, postag1)
        #add('i postag+i-2 postag', postag1, postag2)
        add('i word', c)
        #add('i-1 postag+i word', postag1, context[i])
        add('i postag+i word', postag0, c)
        add('i postag+i-1 EPtag', postag0, EPtag)
        add('i-1 word', cm1)
        #add('i-1 suffix', context[i-1][-3:])
        add('i-2 word', cm2)
        add('i+1 word', cp1)
        #add('i+1 suffix', context[i+1][-3:])
        add('i+2 word', cp2)
        return features

    #Obsolete for now
##    def getWordFeatures(self, words):
##        for i in words:
##            word.append(words[i][0])
##            EPtag.append(words[i][1])
##            postag.append(words[i][2])
##        isWasOrWere = (word[0].lower() == 'was' or word[0].lower() == 'were')
##        endsInEDAndIsVBN = (len(word[0]) > 3 and word[0][-2:] == 'ed' and postag[0] == 'VBN')
##        previousPOSTagIsDT = (postag[1] == 'DT')
##
##        boolFeature = [isWasOrWere, endsInEDAndIsVBN]
##        feature = [int(b) for b in boolFeatures]        
        
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
        fp = 0
        fn = 0
        counter = 0
        outputDoc = []
        for doc in self.testDocs:
            sentences = doc
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
                    output.append([wordlist[i], label, postags[i]])

                    if label != 'N' or EPtags[i] != 'N':
                        print 'Word: ' + wordlist[i] + '-' + 'Predicted/Expected: ' + label + '/' + EPtags[i]
                    if label == EPtags[i] and EPtags[i] != 'N':
                        correct = correct + 1
                    elif label == 'N':
                        fn = fn + 1
                    elif EPtags[i] == 'N':
                        fp = fp + 1
                    counter = counter + 1
                outputSentences.append(output)
            outputDoc.append(outputSentences)
        print "correct: " + str(correct)
        print "fp: " + str(fp)
        print "fn: " + str(fn)
        precision = float(correct)/(correct + fp)
        recall = float(correct)/(correct + fn)
        print 'Precision: ' + str(precision)
        print 'Recall: ' + str(recall)
        print 'F1: ' + str(2.0/((1.0/recall) + (1.0/precision)))
        return outputSentences
        #TODO implement Viterbi

    def train(self, n = 10):
        print 'Training'
        alldocs = self.trainDocs
        for it in range(n):
            random.shuffle(alldocs)
            for doc in alldocs:
                sentences = doc
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
    def __init__(self, labels):
        self.weights = {}
        self.labels = labels
        
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
                            self.weights[f][c] = self.weights[f][c]-float(1.0)/(len(self.labels) - 1)
                else:
                    self.weights[f] = {}
                    for c in self.labels:
                        if c == real:
                            self.weights[f][c] = 1.0
                        else:
                            self.weights[f][c] = -1.0/(len(self.labels) - 1)
    
    def showWeights(self):
        for f in self.weights.keys():
            for l in self.weights[f].keys():
                print f + '-' + l + ': ' + str(self.weights[f][l])

class EPDependencyParser():
    def __init__(self):
        self.SHIFT = 0
        self.LEFT = 1
        self.RIGHT = 2
        self.MOVES = [self.SHIFT, self.RIGHT, self.LEFT]
        self.model = EPModel(self.MOVES)
        
    def transition(self, move, head, stack, parse):
        if move == self.SHIFT:
            stack.append(head)
            return head + 1
        elif move == self.RIGHT:
            parse.add_arc(stack[-2], stack.pop(-1))
            return head
        elif move == self.LEFT:
            parse.add_arc(stack[-1], stack.pop(-2))
            return head

    def get_valid_moves(self, loc, n, stack_depth):
        moves = []
        if loc < n:
            moves.append(self.SHIFT)
        if stack_depth > 2:
            moves.append(self.RIGHT)
        if stack_depth > 2:
            moves.append(self.LEFT)
        return moves

    def _get_features(self, words, tags, n0, n, stack, parse):
        def get_stack_context(depth, stack, data):
            if depth >= 3:
                return data[stack[-1]], data[stack[-2]], data[stack[-3]]
            elif depth >= 2:
                return data[stack[-1]], data[stack[-2]], ''
            elif depth == 1:
                return data[stack[-1]], '', ''
            else:
                return '', '', ''
     
        def get_buffer_context(i, n, data):
            if i + 1 >= n:
                return data[i], '', ''
            elif i + 2 >= n:
                return data[i], data[i + 1], ''
            else:
                return data[i], data[i + 1], data[i + 2]
     
        def get_parse_context(word, deps, data):
            if word == -1:
                return 0, '', ''
            deps = deps[word]
            valency = len(deps)
            if not valency:
                return 0, '', ''
            elif valency == 1:
                return 1, data[deps[-1]], ''
            else:
                return valency, data[deps[-1]], data[deps[-2]]
     
        features = {}
        # Set up the context pieces --- the word, W, and tag, T, of:
        # S0-2: Top three words on the stack
        # N0-2: First three words of the buffer
        # n0b1, n0b2: Two leftmost children of the first word of the buffer
        # s0b1, s0b2: Two leftmost children of the top word of the stack
        # s0f1, s0f2: Two rightmost children of the top word of the stack
     
        depth = len(stack)
        s0 = stack[-1] if depth else -1
     
        Ws0, Ws1, Ws2 = get_stack_context(depth, stack, words)
        Ts0, Ts1, Ts2 = get_stack_context(depth, stack, tags)
     
        Wn0, Wn1, Wn2 = get_buffer_context(n0, n, words)
        Tn0, Tn1, Tn2 = get_buffer_context(n0, n, tags)
     
        Vn0b, Wn0b1, Wn0b2 = get_parse_context(n0, parse.lefts, words)
        Vn0b, Tn0b1, Tn0b2 = get_parse_context(n0, parse.lefts, tags)
     
        Vn0f, Wn0f1, Wn0f2 = get_parse_context(n0, parse.rights, words)
        _, Tn0f1, Tn0f2 = get_parse_context(n0, parse.rights, tags)
     
        Vs0b, Ws0b1, Ws0b2 = get_parse_context(s0, parse.lefts, words)
        _, Ts0b1, Ts0b2 = get_parse_context(s0, parse.lefts, tags)
     
        Vs0f, Ws0f1, Ws0f2 = get_parse_context(s0, parse.rights, words)
        _, Ts0f1, Ts0f2 = get_parse_context(s0, parse.rights, tags)
     
        # Cap numeric features at 5? 
        # String-distance
        Ds0n0 = min((n0 - s0, 5)) if s0 != 0 else 0
     
        features['bias'] = 1
        # Add word and tag unigrams
        for w in (Wn0, Wn1, Wn2, Ws0, Ws1, Ws2, Wn0b1, Wn0b2, Ws0b1, Ws0b2, Ws0f1, Ws0f2):
            if w:
                features['w=%s' % w] = 1
        for t in (Tn0, Tn1, Tn2, Ts0, Ts1, Ts2, Tn0b1, Tn0b2, Ts0b1, Ts0b2, Ts0f1, Ts0f2):
            if t:
                features['t=%s' % t] = 1
     
        # Add word/tag pairs
        for i, (w, t) in enumerate(((Wn0, Tn0), (Wn1, Tn1), (Wn2, Tn2), (Ws0, Ts0))):
            if w or t:
                features['%d w=%s, t=%s' % (i, w, t)] = 1
     
        # Add some bigrams
        features['s0w=%s,  n0w=%s' % (Ws0, Wn0)] = 1
        features['wn0tn0-ws0 %s/%s %s' % (Wn0, Tn0, Ws0)] = 1
        features['wn0tn0-ts0 %s/%s %s' % (Wn0, Tn0, Ts0)] = 1
        features['ws0ts0-wn0 %s/%s %s' % (Ws0, Ts0, Wn0)] = 1
        features['ws0-ts0 tn0 %s/%s %s' % (Ws0, Ts0, Tn0)] = 1
        features['wt-wt %s/%s %s/%s' % (Ws0, Ts0, Wn0, Tn0)] = 1
        features['tt s0=%s n0=%s' % (Ts0, Tn0)] = 1
        features['tt n0=%s n1=%s' % (Tn0, Tn1)] = 1
     
        # Add some tag trigrams
        trigrams = ((Tn0, Tn1, Tn2), (Ts0, Tn0, Tn1), (Ts0, Ts1, Tn0), 
                    (Ts0, Ts0f1, Tn0), (Ts0, Ts0f1, Tn0), (Ts0, Tn0, Tn0b1),
                    (Ts0, Ts0b1, Ts0b2), (Ts0, Ts0f1, Ts0f2), (Tn0, Tn0b1, Tn0b2),
                    (Ts0, Ts1, Ts1))
        for i, (t1, t2, t3) in enumerate(trigrams):
            if t1 or t2 or t3:
                features['ttt-%d %s %s %s' % (i, t1, t2, t3)] = 1
     
        # Add some valency and distance features
        vw = ((Ws0, Vs0f), (Ws0, Vs0b), (Wn0, Vn0b))
        vt = ((Ts0, Vs0f), (Ts0, Vs0b), (Tn0, Vn0b))
        d = ((Ws0, Ds0n0), (Wn0, Ds0n0), (Ts0, Ds0n0), (Tn0, Ds0n0),
            ('t' + Tn0+Ts0, Ds0n0), ('w' + Wn0+Ws0, Ds0n0))
        for i, (w_t, v_d) in enumerate(vw + vt + d):
            if w_t or v_d:
                features['val/d-%d %s %d' % (i, w_t, v_d)] = 1
        return features

    def getWordsTags(self, s):
        words = []
        EPtags = []
        postags = []
        for i in s:
            words.append(i[0])
            EPtags.append(i[1])
            if len(i) == 3:
                postags.append(i[2])
        return words, EPtags, postags

    def train_single(self, words, tags, truemoves):
        words.append('**STOP**')
        tags.append('None')
        n = len(words)
        i = 0
        stack = [-1]
        parse = DependencyParser(n)
        while truemoves:
            features = self._get_features(words, tags, i, n, stack, parse)
            scores = self.model.predict(features)
            true = int(truemoves.pop(0))
            if len(scores) == 0:
                self.model.update(true, 'None', features)
            else:
                valid_moves = self.get_valid_moves(i, n, len(stack))
                guess = max(valid_moves, key=lambda move: scores[move])
                #print guess
                self.model.update(true, guess, features)
            #print true, i, stack
            i = self.transition(true, i, stack, parse)

    def train(self, sentencelist, moveslist):
        for s in range(len(sentencelist)):
            wordlist = sentencelist[s]
            moves = moveslist[s]
            words, EPtags, postags = self.getWordsTags(wordlist)
            self.train_single(words, EPtags, moves)

    def evaluate(self, sentencelist, headslist = None):
        totalheads = 0
        correct = 0
        outputHeads = []
        for s in range(len(sentencelist)):
            wordlist = sentencelist[s]
            if headslist is not None:
                heads = headslist[s]
            words, EPtags, postags = self.getWordsTags(wordlist)
            words.append('**STOP**')
            EPtags.append('None') 
            n = len(words)
            i = 0
            stack = [-1]
            parse = DependencyParser(n)
            print words
            while stack and (i + 1) <= n:
                features = self._get_features(words, EPtags, i, n, stack, parse)
                scores = self.model.predict(features)
                valid_moves = self.get_valid_moves(i, n, len(stack))
                guess = max(valid_moves, key=lambda move: scores[move])
                print guess, i, stack
                i = self.transition(guess, i, stack, parse)
            print parse.heads
            if headslist is not None:
                for j in range(len(parse.heads)):
                    if parse.heads[j] == heads[j]:
                        correct = correct + 1
            totalheads = totalheads + len(parse.heads)
            outputHeads.append(parse.heads)
        if headslist is not None:
            print 'Score: ' + str(float(correct)/totalheads)
        return outputHeads
            
        
class DependencyParser():
    def __init__(self, n):
        self.n = n
        self.heads = [None] * (n-1)
        self.lefts = []
        self.rights = []
        for i in range(n + 1):
            self.lefts.append([])
            self.rights.append([])
    def add_arc(self, head, child):
        self.heads[child] = head
        if child < head:
            self.lefts[head].append(child)
        else:
            self.rights[head].append(child)
    
def groupWords(s):
    previoustag = 'N'
    currentGroup = ''
    output = []
    containsA = False
    currentOutput = []
    for idx in range(len(s)):
        w, EPt, post = s[idx]
        if w == ',':
            if idx + 1 < len(s):
                w1, EPt1, post1 = s[idx + 1]
                if EPt1 == previoustag:
                    currentGroup = currentGroup + ' (+)'
                elif post1 == 'RB':
                    if EPt != 'N' and EPt != 'None':
                        currentOutput.append([currentGroup, previoustag])
                    previoustag = 'N'
                    currentGroup = ''                    
            continue
        
        if EPt != 'N' and EPt != 'None':
            if EPt == 'A':
                if containsA:
                    if previoustag != 'A':
                        if previoustag != 'N':
                            currentOutput.append([currentGroup, previoustag])
                        output.append(currentOutput)
                        currentOutput = []
                        containsA = False
                        currentGroup = w
                        previoustag = 'A'
                        continue
                else:
                    containsA = True
            if previoustag == 'N':
                previoustag = EPt
                currentGroup = w
            else:
                if EPt == previoustag:
                    currentGroup = currentGroup + ' ' + w
                else:
                    currentOutput.append([currentGroup, previoustag])
                    currentGroup = w
                    previoustag = EPt

        else:
            if idx + 1 < len(s):
                w1, EPt1, post1 = s[idx + 1]
                if EPt1 == previoustag and previoustag != 'N':
                    currentGroup = currentGroup + ' (+)'
                    continue
                        
            if previoustag != 'N':
                currentOutput.append([currentGroup, previoustag])
                currentGroup = '' 
                previoustag = 'N'
    output.append(currentOutput)
    return output, containsA     

class Recipe():

    def __init__(self):
        self.parser = EPParser()
        self.originalDocs = self.getGroups()
        self.groupedDocs = []
        for doc in self.originalDocs:
            groupDoc = []
            for s in doc:
                gs, a = groupWords(s)
                for g in gs:
                    groupDoc.append(g)
            self.groupedDocs.append(groupDoc)
        for gd in self.groupedDocs:
            print '----------NEW DOC------------'
            for s in gd:
                print s
        self.originalmoves = []
        for fn in self.parser.trainFilenames:
            self.originalmoves.append(self.getMoves(fn))
        self.parser.shuffle()
        self.parser.train(10)
        self.taggedSentences = self.parser.evaluate()
        self.DPparser = EPDependencyParser()
        #self.DPparser.train(self.groupedDocs, self.originalmoves)
        #self.createRecipe()
        
    def createRecipe(self):
        steps = []
        for s in self.taggedSentences:
            words = [w[0] for w in s]
            groups, containsA = groupWords(s)
            #self.DPparser.evaluate(g)
            for group in groups:
                if containsA:
                    steps.append(group)

        heads = self.DPparser.evaluate(steps)

        print 'Recipe: '
        #print recipe
        print heads

    def getGroups(self):
        s1 = self.parser.trainDocs
        s2 = self.parser.testDocs
        return s1 + s2
    
    def getMoves(self, filename):
        f = open('resources/Moves' + filename, 'r')
        moveslist = []
        for line in f:
            moves = line.split('|')
            for move in moves:
                moveslist.append(move.split())
        return moveslist
                
