#Artificial Intelligence Bayes Network Implementation

class BayesNetwork(object):

    #-----------RANDOM STARTER WEIGHTS---------------
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        def initTable(self):

    def forward(self, X, Y):
        if len(X) is not self.input_size or len(Y) is not self.output_size:
            print "Incorrect Data Dimensions"
            return
        for (y_index, y_value) in [(Y.index(y), y) for y in Y]:
            if y_value in self.table['Y'][y_index]:
                self.table['Y'][y_index][y_value]['freq'] += 1
                for (x_index, x_value) in [(X.index(x), x) for x in X]:
                    if x_value in self.table['Y'][y_index][y_value]['X'][x_index]:
                        self.table['Y'][y_index][y_value]['X'][x_index][x_value]['freq'] += 1
                    else:
                        self.table['Y'][y_index][y_value]['X'][x_index][x_value]['freq'] = 1
            else:
                self.table['Y'][y_index][y_value] = {'freq': 1, 'X':[ {x:{'freq':1}} for x in X]}

    def initTable(self):
        self.table = {'freq':0, 'Y': [{} for table in range(0,self.output_size)]}



#-------------------END OF BAYES NETWORK----------------------
#from BayesNetwork import BayesNetwork
#import random as r
def main():
    print "Welcome"
    # table = {'frequency': 0, 'Y':[ { y: { 'frequency': 1, 'X':[ { x:{'frequency':1} } for x in X] }} for y in Y]}
    #
    # inputs = [[a,b,c] for a in range(3) for b in range(3) for c in range(3) ]
    # outputs = [[a,b] for a in range(2) for b in range(2)]
    # X = [r.choice(inputs) for n in range(100)]
    # Y = [r.choice(outputs) for n in range(100)]
    # baynet = BayesNetwork(len(X[0]), len(Y[0]))


if __name__ == "__main__":
    main()
