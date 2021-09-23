import xml.etree.ElementTree as ET
import re
import spacy
import scispacy
import spacy_transformers
import numpy as np
import networkx as nx

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score
from nltk.tag import pos_tag
from spacy import displacy

def stringReplace(object):
    temp = object.group(0)
    temp = temp[0] + " " + temp[1]
    return temp

def dataPreprocess(root):
    docs = []
    entities = []
    vocabulary = []
    sentences = []
    entVals = []
    yLabel = []
    entStorages = []
    for x in root.findall("./document/sentence"):
        count = 0
        yCount = 0
        flag = 0
        sentence = x.attrib["text"]
        sentence = re.sub('[a-z]\(', stringReplace, sentence)
        sentence = re.sub('[a-z]\)', stringReplace, sentence)
        sentence = re.sub('\([a-z]', stringReplace, sentence)
        sentence = re.sub('\)[a-z]', stringReplace, sentence)
        sentence = re.sub('[a-z]\/', stringReplace, sentence)
        sentence = re.sub('\/[a-z]', stringReplace, sentence)
        sentence = re.sub('[a-z]\.', stringReplace, sentence)
        sentence = re.sub('[a-z]\,', stringReplace, sentence)
        sentence = re.sub('[a-z]\-', stringReplace, sentence)
        sentence = re.sub('\-[a-z]', stringReplace, sentence)
        sentence = re.sub('[A-Z]\-', stringReplace, sentence)
        sentence = re.sub('[A-Z]\(', stringReplace, sentence)
        sentence = re.sub('[A-Z]\)', stringReplace, sentence)
        sentence = re.sub('\([A-Z]', stringReplace, sentence)
        sentence = re.sub('\)[A-Z]', stringReplace, sentence)
        sentence = re.sub('\-[A-Z]', stringReplace, sentence)
        sentence = re.sub('[A-Z]\/', stringReplace, sentence)
        sentence = re.sub('\/[A-Z]', stringReplace, sentence)
        sentence = re.sub('[0-9]\-', stringReplace, sentence)
        sentence = re.sub('\-[0-9]', stringReplace, sentence)
        sentence = re.sub('[0-9]\(', stringReplace, sentence)
        temp = sentence
        entCount = 0
        entStorage = []
        pairStorage = []
        for i in x.findall("./entity"):
            entCount += 1
        if entCount < 2:
            continue
        for i in x.findall("./entity"):
            entDraft = i.attrib["text"]
            entDraft = re.sub('[a-z]\(', stringReplace, entDraft)
            entDraft = re.sub('[a-z]\)', stringReplace, entDraft)
            entDraft = re.sub('\([a-z]', stringReplace, entDraft)
            entDraft = re.sub('\)[a-z]', stringReplace, entDraft)
            entDraft = re.sub('[a-z]\/', stringReplace, entDraft)
            entDraft = re.sub('\/[a-z]', stringReplace, entDraft)
            entDraft = re.sub('[a-z]\.', stringReplace, entDraft)
            entDraft = re.sub('[a-z]\,', stringReplace, entDraft)
            entDraft = re.sub('[a-z]\-', stringReplace, entDraft)
            entDraft = re.sub('\-[a-z]', stringReplace, entDraft)
            entDraft = re.sub('[A-Z]\-', stringReplace, entDraft)
            entDraft = re.sub('[A-Z]\(', stringReplace, entDraft)
            entDraft = re.sub('[A-Z]\)', stringReplace, entDraft)
            entDraft = re.sub('\([A-Z]', stringReplace, entDraft)
            entDraft = re.sub('\)[A-Z]', stringReplace, entDraft)
            entDraft = re.sub('\-[A-Z]', stringReplace, entDraft)
            entDraft = re.sub('[A-Z]\/', stringReplace, entDraft)
            entDraft = re.sub('\/[A-Z]', stringReplace, entDraft)
            entDraft = re.sub('[0-9]\-', stringReplace, entDraft)
            entDraft = re.sub('\-[0-9]', stringReplace, entDraft)
            entDraft = re.sub('[0-9]\(', stringReplace, entDraft)
            if entDraft in entStorage:
                flag = 1
            entStorage.append(entDraft)
        if flag == 1:
            continue
        entStorages.append(entStorage)
        for i in x.findall("./pair"):
            if (entStorage[int(i.attrib["e1"][-1])] == entStorage[int(i.attrib["e2"][-1])]):
                continue
            elif (entStorage[int(i.attrib["e2"][-1])].find(entStorage[int(i.attrib["e1"][-1])]) != -1):
                continue
            elif (entStorage[int(i.attrib["e1"][-1])].find(entStorage[int(i.attrib["e2"][-1])]) != -1):
                continue
            if i.attrib["interaction"] == "False":
                interVal = 0
            else:
                interVal = 1
            tup = (int(i.attrib["e1"][-1]), int(i.attrib["e2"][-1]), interVal)
            pairStorage.append(tup)
        for (a, b, c) in pairStorage:
            vals = {}
            ents = {}
            tempSen = sentence
            tempSen = tempSen.replace(entStorage[a], "ENTITY0")
            tempSen = tempSen.replace(entStorage[b], "ENTITY1")
            ents[entStorage[a]] = ["BRAIN_REGION"]
            ents[entStorage[b]] = ["BRAIN_REGION"]
            vals["ENTITY0"] = entStorage[a]
            vals["ENTITY1"] = entStorage[b]
            yLabel.append(c)
            docs.append(tempSen)
            entVals.append(vals)
            sentences.append(sentence)
            entities.append(ents)
    return yLabel, docs, entVals, sentences, entities

def shortestPathsCalculator(yLabel, docs, entVals, sentences, entities):
    shortestPaths = []
    dependecies = []
    nums = []
    _docs = docs
    for sr, i in enumerate(_docs):
        # print(i)
        document = nlp(i)
        edges = []
        deps = []
        for token in document:
            for child in token.children:
                edges.append(('{0}'.format(token.lower_), '{0}'.format(child.lower_)))
                deps.append(('{0}'.format(token.dep_), '{0}'.format(child.dep_)))
        graph = nx.Graph(edges)
        entity1 = 'entity0'
        entity2 = 'entity1'
        # print(sr)
        # print(edges)
        try:
            path = nx.shortest_path(graph, source=entity1, target=entity2)
            directions = []
            dirDeps = []
            for xi, x in enumerate(edges):
                for y in range(len(path) - 1):
                    if path[y] in x:
                        if path[y] == x[0]:
                            if path[y + 1] == x[1]:
                                directions.append("->")
                                dirDeps.append([deps[xi][0], deps[xi][1]])
                            else:
                                continue
                        else:
                            if path[y + 1] == x[0]:
                                directions.append("<-")
                                dirDeps.append([deps[xi][1], deps[xi][0]])
                            else:
                                continue
                    else:
                        continue
            finalPath = []
            for x in range(len(path)):
                finalPath.append(path[x])
                if len(directions) > x:
                    finalPath.append(directions[x])
        except nx.NetworkXNoPath:
            del yLabel[sr]
            del docs[sr]
            del sentences[sr]
            del entVals[sr]
            del entities[sr]
            continue   
        # print(path)
        nums.append(sr)
        shortestPaths.append(finalPath)
        dependecies.append(dirDeps)
    print(len(nums), len(shortestPaths), len(dependecies), len(yLabel), len(docs), len(sentences), len(entVals), len(entities))
    return shortestPaths, dependecies

def entityLabels(sentences, entities):
    for sr, x in enumerate(sentences):
        document = nlp(x)
        for d in document.ents:
            for i in entities[sr].keys():
                if d.text in i:
                    if d.label_ not in entities[sr][i]:
                        entities[sr][i].append(d.label_)

class Tree:
    def __init__ (self):
        self.nodes = []
    
    def addNodes(self, nodeList, n, entVals, entities, dependencies):
        if len(nodeList) == 3:
            flag = 0
            for i in range(3):
                if nodeList[i] == "<-":
                    self.nodes.append([nodeList[i], dependencies[n][0][0]])
                if nodeList[i] == "->":
                    self.nodes.append([nodeList[i], dependencies[n][0][1]])
                else:
                    ent = entVals[n]["ENTITY" + str(flag)]
                    self.nodes.append([ent, "NN", "NOUN", entities[n][ent]])
                    flag = 1
        else:
            relCount = 0
            for i in range(len(nodeList)):
                if (nodeList[i] == "<-"):
                    self.nodes.append([nodeList[i], dependencies[n][relCount][0]])
                    relCount += 1
                elif ( nodeList[i] == "->"):
                    self.nodes.append([nodeList[i], dependencies[n][relCount][1]])

                # if (nodeList[i] == "<-" or nodeList[i] == "->"):
                #     self.nodes.append([nodeList[i]])

                # if (nodeList[i] == "<-" or nodeList[i] == "->"):
                #     continue
                
                elif (nodeList[i] == "entity0"):
                    ent = entVals[n]["ENTITY0"]
                    self.nodes.append([ent, "NN", "NOUN", entities[n][ent]])
                elif (nodeList[i] == "entity1"):
                    ent = entVals[n]["ENTITY1"]
                    self.nodes.append([ent, "NN", "NOUN", entities[n][ent]])
                else:
                    test = nlp(nodeList[i])
                    for token in test:
                        self.nodes.append([token.text, token.tag_, token.pos_])
            
    def displayNodes(self):
        print(self.nodes)

def similarityFunction(x, y):
    if len(x) != len(y):
        return 0
    else:
        simScore = 1
        for (xVal, yVal) in zip(x, y):
            tempSimScore = 0
            if len(xVal) == len(yVal):
                for i in range(len(xVal)):
                    if isinstance(xVal[i], list):
                        for j in xVal[i]:
                            if j in yVal[i]:
                                tempSimScore += 1
                    elif xVal[i] == yVal[i]:
                        tempSimScore += 1
            simScore *= tempSimScore
        return simScore

def treeKernel(X1, X2):
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = similarityFunction(x1.nodes, x2.nodes)
    return gram_matrix

nlp = spacy.load("en_ner_bionlp13cg_md")

tree = ET.parse('data/train/train.xml')
root = tree.getroot()
print(root.tag)

yLabel, docs, entVals, sentences, entities = dataPreprocess(root)
print(len(docs), len(yLabel))

entityLabels(sentences, entities)

for i in range(2):
    shortestPaths, dependencies = shortestPathsCalculator(yLabel, docs, entVals, sentences, entities)

for i in range(10):
    print(docs[i])
    print(shortestPaths[i], dependencies[i], yLabel[i])

xObjects = []
for i in range(len(shortestPaths)):
    obj = Tree()
    obj.addNodes(shortestPaths[i], i, entVals, entities, dependencies)
    xObjects.append(obj)

for i in range(10):
    xObjects[i].displayNodes()

xObjects = np.array(xObjects)
yLabel = np.array(yLabel)
print(len(xObjects), len(yLabel))

def SVM(count, C):
    print(str(count) + ":" + str(C))
    classifier = SVC(kernel = "precomputed", C = C)
    model = classifier.fit(treeKernel(xObjectsTrain, xObjectsTrain), yLabelTrain)
    pred = model.predict(treeKernel(xObjectsValid, xObjectsTrain))
    results[count][0] = C
    results[count][1] = model.score(treeKernel(xObjectsValid, xObjectsTrain), yLabelValid)
    results[count][2] = f1_score(yLabelValid, pred)
    results[count][3] = precision_score(yLabelValid, pred)
    results[count][4] = recall_score(yLabelValid, pred)


xObjectsArray = np.array_split(xObjects, 10)
yLabelArray = np.array_split(yLabel, 10)
fold = 0
finalC = 0
finalF1 = 0
for z in range(10):
    xObjectsValid = xObjectsArray[z]
    yLabelValid = yLabelArray[z]
    xObjectsTrain = np.array([])
    yLabelTrain = np.array([])
    for j in range(10):
        if j == z:
            continue
        else:
            xObjectsTrain = np.concatenate((xObjectsTrain, xObjectsArray[j]))
            yLabelTrain = np.concatenate((yLabelTrain, yLabelArray[j]))
    c = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
    results = np.zeros((7, 5))
    count = 0
    for i in c:
        # SVM(count, i)
        classifier = SVC(kernel = "precomputed", C = i)
        model = classifier.fit(treeKernel(xObjectsTrain, xObjectsTrain), yLabelTrain)
        pred = model.predict(treeKernel(xObjectsValid, xObjectsTrain))
        results[count][0] = i
        results[count][1] = model.score(treeKernel(xObjectsValid, xObjectsTrain), yLabelValid)
        results[count][2] = f1_score(yLabelValid, pred)
        results[count][3] = precision_score(yLabelValid, pred)
        results[count][4] = recall_score(yLabelValid, pred)
        count += 1
    bestC = 0
    bestF1 = 0
    for i in range(7):
        if results[i][2] > bestF1:
            bestC = results[i][0]
            bestF1 = results[i][2]
    if bestF1 > finalF1:
        finalC = bestC
        fold = z
        finalF1 = bestF1
    print("Fold: {}\t\tC: {}\t\tF1: {}".format(z, bestC, bestF1))
print("Best Fold: {}\t\tC: {}\t\tF1: {}".format(fold, finalC, finalF1))

xObjectsValid = xObjectsArray[fold]
yLabelValid = yLabelArray[fold]
xObjectsTrain = np.array([])
yLabelTrain = np.array([])
for j in range(10):
    if j == fold:
        continue
    else:
        xObjectsTrain = np.concatenate((xObjectsTrain, xObjectsArray[j]))
        yLabelTrain = np.concatenate((yLabelTrain, yLabelArray[j]))

classifier = SVC(kernel = "precomputed", C = finalC, gamma = "scale")
model = classifier.fit(treeKernel(xObjectsTrain, xObjectsTrain), yLabelTrain)

treeTest = ET.parse('data/test/WhiteTextUnseenEval.xml')
rootTest = treeTest.getroot()
print(rootTest.tag)

yLabelTest, docsTest, entValsTest, sentencesTest, entitiesTest = dataPreprocess(rootTest)

print(len(docsTest), len(yLabelTest))

entityLabels(sentencesTest, entitiesTest)

for i in range(2):
    shortestPathsTest, dependenciesTest = shortestPathsCalculator(yLabelTest, docsTest, entValsTest, sentencesTest, entitiesTest)

xObjectsTest = []
for i in range(len(shortestPathsTest)):
    obj = Tree()
    obj.addNodes(shortestPathsTest[i], i, entValsTest, entitiesTest, dependenciesTest)
    xObjectsTest.append(obj)

xObjectsTest = np.array(xObjectsTest)
yLabelTest = np.array(yLabelTest)

pred = model.predict(treeKernel(xObjectsTest, xObjectsTrain))
accuracy = model.score(treeKernel(xObjectsTest, xObjectsTrain), yLabelTest)
f1 = f1_score(yLabelTest, pred)
f2 = fbeta_score(yLabelTest, pred, beta = 2)
precision = precision_score(yLabelTest, pred)
recall = recall_score(yLabelTest, pred)

print("    Accuracy     F1     Precision    Recall     F2")
print("{:10.4f} {:10.4f} {:10.4f} {:10.4f}".format(accuracy, f1, precision, recall, f2))