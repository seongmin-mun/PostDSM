class SBEs:

    def __init__(self, fold, window, function):
        self.fold = fold
        self.window = window
        self.function = function

    def Processing(self):
        import numpy as np
        import pandas as pd

        functions = self.function

        functionDicDic = {}

        for function in functions:

            functionDic = {}
            functionDir = "../../Data/Output/PPMI_SVD/Lo/" + str(self.fold) + "Fold/Lo_" + function + "_window_" + str(self.window) + ".csv"

            dfFunction = pd.read_csv(functionDir)
            words = dfFunction['word'].tolist()
            sims = dfFunction['similarity'].tolist()
            for k in range(0, len(words)):
                functionDic[words[k]] = sims[k]

            functionDicDic[function] = functionDic

        testDir = "../../Data/Input/Fold/" + str(self.fold) + "Fold/Lo_test_" + str(self.fold) + ".csv"

        df = pd.read_csv(testDir)

        headlines = df['Sentence'].tolist()

        countDic = {}  # corect
        frequencyDic = {}

        countDic["Total"] = 0
        frequencyDic["Total"] = 0

        for function in functions:
            countDic[function] = 0
            frequencyDic[function] = 0


        for sentence in headlines:
            originClass = ""

            token = sentence.split(" ")
            for eachToken in token:
                if ("(으)로/") in eachToken and "JKB_" in eachToken:
                    originClass = eachToken.replace(("(으)로/"), "").replace("JKB_", "")

            classifiedFunc = {}

            funcScore = {}
            matchNum = 0

            for eachToken in token:

                if functionDicDic.get("LOC").get(eachToken.strip()) == None or (("(으)로/") in eachToken and "JKB_" in eachToken):
                    pass
                else:

                    matchNum = matchNum + 1
                    for function in functions:
                        if funcScore.get(function) == None:
                            funcScore[function] = functionDicDic.get(function).get(eachToken.strip())
                        else:
                            funcScore[function] = funcScore.get(function) + functionDicDic.get(function).get(eachToken.strip())


            for function in functions:
                classifiedFunc[function] = funcScore.get(function) / matchNum


            dic_max = max(classifiedFunc.values())

            for x, y in classifiedFunc.items():
                if y == dic_max:
                    for function in functions:
                        if originClass == function and originClass == x:
                            countDic[function] = countDic.get(function) + 1
                    if originClass == x:
                        countDic["Total"] = countDic.get("Total") + 1

            frequencyDic["Total"] = frequencyDic.get("Total") + 1

            for function in functions:
                if originClass == function:
                    frequencyDic[function] = frequencyDic.get(function) + 1

        finalResult = {}

        totalAccuracy = countDic.get("Total") / frequencyDic.get("Total")
        finalResult["Total"] = totalAccuracy

        for function in functions:
            funcAccuracy = countDic.get(function) / frequencyDic.get(function)
            finalResult[function] = funcAccuracy

        averageAccuracy = 0
        for function in functions:
            averageAccuracy = averageAccuracy+finalResult.get(function)

        finalResult["TotalAverage"] = averageAccuracy / len(functions)

        return finalResult







