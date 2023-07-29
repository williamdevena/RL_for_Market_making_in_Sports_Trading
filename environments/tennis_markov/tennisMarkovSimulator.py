import random

from environments.tennis_markov import match_probability


class TennisMarkovSimulator():

    def __init__(self,
                 s,
                 t,
                 gv=0,
                 gw=0,
                 sv=0,
                 sw=0,
                 mv=0,
                 mw=0,
                 sets=3):
        self.s = s
        self.t = t
        self.gv = gv
        self.gw = gw
        self.sv = sv
        self.sw = sw
        self.mv= mv
        self.mw = mw
        self.set = sets


    def restart(self):
        self.gv = 0
        self.gw = 0
        self.sv = 0
        self.sw = 0
        self.mv = 0
        self.mw = 0


    def simulate(self):
        prob_list = []
        odds_list = []
        games_idx = []
        ## A PLAYER WINS THE MATCH
        while self.mv<2 and self.mw<2:
            self.sv = 0
            self.sw = 0
            ## A PLAYER WINS A SET
            while self.sv<6 and self.sw<6:
                self.gv = 0
                self.gw = 0
                ## A PLAYER WINS A GAME
                while self.gv<4 and self.gw<4:
                    print("1")
                    prob = match_probability.matchProb(s=self.s, t=self.t,
                                                       gv=self.gv, gw=self.gw,
                                                       sv=self.sv, sw=self.sw,
                                                       mv=self.mv, mw=self.mw)
                    prob_list.append(prob)
                    odds_list.append(1/prob)
                    ## for test (prob should never be >1)
                    # if prob>1:
                    #     print(prob)
                    rand_num = random.random()
                    if rand_num<=0.5:
                        self.gv += 1
                    else:
                        self.gw += 1
                    print(self.gv, self.gw, self.sv, self.sw, self.mv, self.mw)
                games_idx.append(len(prob_list)-1)
                if self.gv==4:
                    self.sv += 1
                else:
                    self.sw += 1
                print(self.gv, self.gw, self.sv, self.sw, self.mv, self.mw)
            if self.sv==6:
                self.mv += 1
            else:
                self.mw += 1
            print(self.gv, self.gw, self.sv, self.sw, self.mv, self.mw)

        return prob_list, odds_list, games_idx




if __name__=="__main__":
    import matplotlib.pyplot as plt

    s = 0.5
    t = 0.5
    simulator = TennisMarkovSimulator(s=s, t=t)


    prob_list, odds_list, games_idx = simulator.simulate()

    plt.plot(prob_list)
    plt.show()






