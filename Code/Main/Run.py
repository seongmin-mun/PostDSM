

#
# from Code.Algorithm.PPMI_SVD import PPMI_SVD_Algorithm
#
foldNum = [1,2,3,4,5,6,7,8,9,10]
windowNum = [1,2,3,4,5,6,7,8,9,10]
functionLo = ["FNS","INS","DIR","EFF","CRT","LOC"]
functionLoR = ["Total","TotalAverage","FNS","INS","DIR","EFF","CRT","LOC"]
#
# for fold in foldNum:
#     ppmi_svd = PPMI_SVD_Algorithm(fold, 11)
#     ppmi_svd.PPMI_SVD_Calculation()
#
#


from Code.Algorithm.PPMI_SVD_tSNE import PPMI_SVD_tSNE_Algorithm

ppmi_svd = PPMI_SVD_tSNE_Algorithm(11)
ppmi_svd.PPMI_SVD_tSNE_Calculation()


#
# from Code.Evaluation.SimilarityBasedEstimation import SBEs
#
#
#
# for window in windowNum:
#     # print("")
#     # print("Window :", window)
#     foldAverage = {}
#     for function in functionLoR:
#         foldAverage[function] = 0;
#     for fold in foldNum:
#         #print("Fold :", fold)
#         sbm = SBEs(fold, window, functionLo)
#         result = sbm.Processing()
#         #print("TotalAverage : ",result.get("TotalAverage"))
#         #print(result)
#         for function in functionLoR:
#             foldAverage[function] = foldAverage.get(function) + result.get(function)
#     for function in functionLoR:
#         foldAverage[function] = foldAverage.get(function) / 10
#     print(foldAverage.get("LOC"), ",", end=' ')


