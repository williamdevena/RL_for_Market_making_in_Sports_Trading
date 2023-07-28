## calculates probability of winning a tennis match from any given score dependent on the skill levels
## of the two players


import random

import matplotlib.pyplot as plt
from tennisGameProbability import gameProb
from tennisSetProbability import setGeneral
from tennisTiebreakProbability import tiebreakProb


def fact(x):
    if x in [0, 1]:  return 1
    r = 1
    for a in range(1, (x+1)):  r = r*a
    return r

def ch(a, b):
    return fact(a)/(fact(b)*fact(a-b))

def matchGeneral(e, v=0, w=0, s=3):
    ## calculates probability of winning the match
    ## from the beginning of a set
    ## e is p(winning a set)
    ## v and w is current set score
    ## s is total number of sets ("best of")
    towin = (s+1)/2
    left = towin - v
    if left == 0:   return 1
    remain = s - v - w
    if left > remain:   return 0
    win = 0
    #print(left, remain)
    for i in range(int(left), (remain+1)):

        #print(i, left, e)

        add = ch((i-1), (int(left)-1))*(e[0]**(int(left)-1))*((1-e[0])**(i-int(left)))*e[0]
        win += add
    return win

def matchProb(s, t, gv=0, gw=0, sv=0, sw=0, mv=0, mw=0, sets=3):
    ## calculates probability of winning a match from any given score,
    ## given:
    ## s, t: p(server wins a service point), p(server wins return point)
    ## gv, gw: current score within the game. e.g. 30-15 is 2, 1
    ## sv, sw: current score within the set. e.g. 5, 4
    ## mv, mw: current score within the match (number of sets for each player)
    ## v's are serving player; w's are returning player
    ## sets: "best of", so default is best of 3

    # a = gameProb(s)
    # b = gameProb(t)
    print(gv, gw, sv, sw, mv, mw)
    c = setGeneral(s, t)
    print("-------------")

    if gv == 0 and gw == 0: ## no point score
        if sv == 0 and sw == 0: ## no game score
            #print(c[0])
            return matchGeneral(c, v=mv, w=mw, s=sets)
        else:   ## we're in mid-set, no point score
            #sWin = setGeneral(a, b, s, t, v=sv, w=sw)
            sWin = setGeneral(s, t, v=sv, w=sw)[0]
            sLoss = 1 - sWin
    elif sv == 6 and sw == 6:
        sWin = tiebreakProb(s, t, v=gv, w=gw)
        sLoss = 1 - sWin
    else:
        gWin = gameProb(s, v=gv, w=gw)
        gLoss = 1 - gWin
        #sWin = gWin*(1 - setGeneral((1-b), (1-a), (1-t), (1-s), v=sw, w=(sv+1)))

        result = setGeneral((1-t), (1-s), v=sw, w=(sv+1))
        if isinstance(result, tuple):
            sWin = gWin*(1 - result[0])
        else:
            sWin = gWin*(1 - result)


        # sWin = gWin*(1 - setGeneral((1-t), (1-s), v=sw, w=(sv+1))[0])
        # sWin += gLoss*(1 - setGeneral((1-b), (1-a), (1-t), (1-s), v=(sw+1), w=sv))
        result = setGeneral((1-t), (1-s), v=(sw+1), w=sv)
        if isinstance(result, tuple):
            sWin += gLoss*(1 - result[0])
        else:
            sWin += gLoss*(1 - result)
        sLoss = 1 - sWin

    mWin = sWin * matchGeneral(c, v=(mv+1), w=mw, s=sets)
    mWin += sLoss * matchGeneral(c, v=mv, w=(mw+1), s=sets)

    return mWin





if __name__=="__main__":
    # match_win = matchProb(s=0.85, t=0.2)
    # print(match_win)

    s = 0.5
    t = 0.5

    for x in range(10):
        #print(x)

        prob_list = []
        odds_list = []
        games_idx = []

        mv = 0
        mw = 0

        while mv<2 and mw<2:
            sv = 0
            sw = 0
            while sv<6 and sw<6:
                gv = 0
                gw = 0
                while gv<4 and gw<4:

                    # print(matchProb(s=s, t=t, gv=gv, gw=gw, sv=sv, sw=sw, mv=mv, mw=mw))
                    prob = matchProb(s=s, t=t, gv=gv, gw=gw, sv=sv, sw=sw, mv=mv, mw=mw)
                    #print(prob)
                    prob_list.append(prob)
                    odds_list.append(1/prob)
                    if prob>1:
                        print(prob)
                        print(gv, gw, sv, sw, mv, mw)
                    rand_num = random.random()
                    if rand_num<=0.5:
                        gv += 1
                    else:
                        gw += 1

                games_idx.append(len(prob_list)-1)
                if gv==4:
                    sv +=1
                else:
                    sw += 1
            if sv==6:
                mv += 1
            else:
                mw += 1


        plt.plot(prob_list)

        # for idx in games_idx:
        #     plt.axvline(idx, color='red')

        #plt.plot(odds_list)
        #plt.legend()

    #plt.ylim(0, 10)
    plt.show()
    plt.close()


    #print(gv, gw, sv, sw, mv, mw)


    # print(matchProb(s=s, t=t, gv=0, gw=0, sv=0, sw=0))
    # print(matchProb(s=s, t=t, gv=1, gw=0, sv=0, sw=0))
    # print(matchProb(s=s, t=t, gv=2, gw=0, sv=0, sw=0))
    # print(matchProb(s=s, t=t, gv=3, gw=0, sv=0, sw=0))
    # print(matchProb(s=s, t=t, gv=0, gw=0, sv=1, sw=0))

    # print(matchProb(s=s, t=t, gv=0, gw=0, sv=4, sw=0))

    # print(matchProb(s=s, t=t, gv=0, gw=0, sv=2, sw=2, mv=1, mw=0))

    #print(matchProb(s=s, t=t, gv=5, gw=0, sv=0, sw=0))
    #print(matchProb(s=s, t=t, gv=1, gw=0, sv=0, sw=0))

