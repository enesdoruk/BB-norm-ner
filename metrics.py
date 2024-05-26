

def accuracy(dd_pred, dd_resp):
    totalScore = 0.0

    for id in dd_resp.keys():
        score = 0.0
        l_cuiPred = dd_pred[id]["pred_cui"]
        l_cuiResp = dd_resp[id]["cui"]
        if len(l_cuiPred) > 0: 
            for cuiPred in l_cuiPred:
                if cuiPred in l_cuiResp:
                    score += 1
            score = score / max(len(l_cuiResp), len(l_cuiPred))  

        totalScore += score 

    totalScore = totalScore / len(dd_resp.keys())

    return totalScore