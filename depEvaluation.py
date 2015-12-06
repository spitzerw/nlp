"""
Evalaute dependency 
input:
parsed dependency: 0 0 1 0 2 0 0 2 2 2|0 0 2 0 2 2 
actual dependency:

output: number of correct arc/ total number of arcs
"""

SHIFT = '0'
LEFT = '1'
RIGHT = '2'
START = '|'
# @input: dependency
# @list of arcs where arc is (i, j)
def getArcs( deps):
	movements = deps.replace(' ', '')
	rootIndx = 0
	bufferTop = 1
	indxInStack = [rootIndx]
	arcs = []
	for move in movements:
		if move == SHIFT:
			indxInStack.append(bufferTop)
			bufferTop += 1
		elif move == LEFT:
			arcs.append( ( indxInStack[-1], indxInStack[-2]))
			indxInStack.pop(-2)
		elif move == RIGHT:
			arcs.append( ( indxInStack[-2], indxInStack[-1]))
			indxInStack.pop(-1)
		elif move == START:
			indxInStack = [rootIndx]
	return arcs


def evaluate(trueDeps, parsedDeps):
	trueArcs = getArcs(trueDeps)
	parsedArcs = getArcs(parsedDeps)
	numCorrectArcs = 0.0
	for arc in parsedArcs:
		if arc in trueArcs:
			numCorrectArcs +=1
	print( numCorrectArcs, len(parsedArcs))

	print( numCorrectArcs/len(parsedArcs))
	return numCorrectArcs/len(parsedArcs)

#TEST works
#getArcs("0 0 1 0 2 0 0 2 2 2|0 0 2 0 2 2")
#getArcs('0 0 1 0 0 1 2 0 2 2')
evaluate('0 0 1 0 0 1 2 0 2 2', '0 0 1 0 0 1 2 0 2 1')
