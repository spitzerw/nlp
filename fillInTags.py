import argparse
import os
import codecs
from os import listdir

def fillInTags(filename):
    f = codecs.open('samples/' + filename, encoding = 'utf-8')
    sentences = []
    allEPtags = []
    for line in f:
        words = []
        tags = []
        w = line.split()
        if w:
            for word in w:
                if '_' in word:
                    ww = word.split('_')
                    words.append(ww[0])
                    tags.append(ww[1])
                else:
                    words.append(word)
                    tags.append('N')
        sentences.append(words)
        allEPtags.append(tags)
    f.close()
    ff = codecs.open('samples/F' + filename, 'w', encoding = 'utf-8')
    for i in range(len(sentences)):
        sentence = sentences[i]
        tags = allEPtags[i]
        strSent = ''
        for j in range(len(sentence)):
            strSent = strSent + sentence[j] + '_' + tags[j]
            if j != len(sentence) - 1:
                strSent = strSent + ' '
            else:
                strSent = strSent + '\n'
        ##print strSent
        ff.write(strSent)
    ff.close()

def doAll():
    samplefiles = [filename for filename in listdir('samples') if 'train' in filename and 'Ftrain' not in filename]
    for f in samplefiles:
        fillInTags(f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="file")
    args = parser.parse_args()
    if args.file:
        if args.file == 'all':
            doAll()
        else:
            fillInTags(args.file)
        
    else:
        doAll()
