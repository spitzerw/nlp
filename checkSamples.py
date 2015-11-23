from os import listdir

samplefiles = [filename for filename in listdir('samples') if 'train' in filename]
labels = ['N', 'I', 'E', 'A', 'P', 'PR', 'R']
for fn in samplefiles:
    print 'Document: ' + str(fn)
    with open('samples/' + fn, 'r') as f:
        counter = 0
        for line in f:
            counter = counter + 1
            words = line.split()
            for idx in range(len(words)):
                w = words[idx]
                if '_' in w:
                    t = w.split('_')[1]
                    if t not in labels:
                        txt = '>>>' + w + '<<<'
                        for j in range(1,3):
                            if idx - j >= 0:
                                txt = words[idx-j] + ' '  + txt
                            else:
                                txt = '***START*** ' + txt
                            if idx + j < len(words):
                                txt = txt + ' ' + words[idx-j]
                            else:
                                txt = txt + ' ***STOP***'
                                
                        print 'Error in line ' + str(counter) + ': ' + txt
                else:
                    txt = '>>>' + w + '<<<'
                    for j in range(1,3):
                        if idx - j >= 0:
                            txt = words[idx-j] + ' '  + txt
                        else:
                            txt = '***START*** ' + txt
                        if idx + j < len(words):
                            txt = txt + ' ' + words[idx-j]
                        else:
                            txt = txt + ' ***STOP***'
                            
                    print 'Error in line ' + str(counter) + ': ' + txt
