import random

class Brain:
    def __init__(self):
        self.maxNodeSize = 10
        self.nodeCount = -1
        self.nodes = []

        for i in range(self.maxNodeSize):
            self.nodes.append(basicNode())

    def getNodes(self):
        nodeList = []
        for node in self.nodes:
            nodeList.append(node.getData())
        return nodeList

    def getTensorflowData(self):
        data = []
        for node in self.nodes:
            for source in node.source:
                data.append(source)
            for expect in node.expect:
                data.append(expect)
            data.append(node.weight)
            data.append(node.output)
        return data

    def setTensorflowData(self):
        pass

    def printNodes(self):
        for node in self.nodes:
            node.printData()

    def createRandNode(self):
        random.seed()
        if self.nodeCount < self.maxNodeSize-1:
            self.nodeCount += 1
            node = self.nodes[self.nodeCount]
            node.source[0] = random.randint(1, 5)
            node.expect[0] = random.randint(0, 50)
            node.weight = random.randint(1, 20) / 10
            node.output = random.randint(0, 5)
            node.numOfSources += 1
        else:
            print("Error creating new Random Node!")

    def randMutate(self):
        choice = random.randint(1, 5)
        if choice <= 2:  # Change Node
            self.changeNode()
        elif choice <= 3:  # Add Node
            self.createRandNode()
        elif choice <= 4:  # Remove Node
            self.removeNode()

    def changeNode(self, index=-1):
        index = random.randint(0, self.nodeCount) if index == -1 else index  # if index == -1, index = rand # between 0 - # of nodes
        node = self.nodes[index]
        num = random.randint(0, 4)
        if num <= 0:  # make changes to source
            tempIndex = 0
            node.source[tempIndex] = random.randint(1, 5)
        elif num <= 1:  # make changes to expected
            tempIndex = 0
            node.expect[tempIndex] = random.randint(0, 50)
        elif num <= 2:  # make changes to weight
            node.weight = random.randint(1, 20)/10
        elif num <= 3:  # makes changes to output
            node.output = random.randint(0, 5)
        elif num <= 4:
            self.createNewSource()

    def createNewSource(self):
        pass

    def removeNode(self, index=-1):
        index = random.randint(0, self.nodeCount) if index == -1 else index  # if index == -1, index = rand # between 0 - # of nodes
        self.nodes.pop(index)
        self.nodeCount -= 1
        self.nodes.append(basicNode())
        if self.nodeCount == 0:
            self.createRandNode()

    def pickAction(self, scanPoints):
        bestNode = 0
        bestScore = 0
        currentScore = 0
        for i, node in enumerate(self.nodes):
            if node.source[0] != 0:
                sourceNum = 0
                for source in node.source:
                    if source != 0:
                        currentScore += abs(scanPoints[source - 1] - node.expect[sourceNum]) * 10
                        sourceNum += 1
                currentScore = int((currentScore/sourceNum) * node.weight)
                if currentScore > bestScore:
                    bestNode = i
                    bestScore = currentScore
                currentScore = 0
        return self.nodes[bestNode].output

class basicNode:
    def __init__(self):
        self.source = [0, 0, 0]
        self.expect = [25, 25, 25]
        self.weight = 1
        self.output = 0

        self.numOfSources = -1

    def getData(self):
        return self.source, self.expect, self.weight, self.output

    def printData(self):
        print(self.source, end=", ")
        print(self.expect, end=", ")
        print(self.weight, end=", ")
        print(self.output)
