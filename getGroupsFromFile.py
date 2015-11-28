import argparse
import os
import codecs
from os import listdir
import pickle

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

def main(filename):
    print 'Groups for ' + filename
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
    for s in sentences:
        output, containsA = groupWords(s)    
        for g in output:
            print g

def doAll():
    samplefiles = [filename for filename in listdir('samples') if 'Ftrain' in filename]
    for f in samplefiles:
        main(f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="file")
    args = parser.parse_args()
    if args.file:
        if args.file == 'all':
            doAll()
        else:
            main(args.file)
        
    else:
        doAll()
