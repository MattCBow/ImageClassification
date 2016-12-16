#Artificial Intelligence Bayes Network Implementation

class BayesNetwork(object):

    #-----------RANDOM STARTER WEIGHTS---------------
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.table = {}
        self.table['Y'] = [{} for table in range(0,self.output_size)]

    def forward(self, Y, X):
        if not hasattr(self, 'table'):
            self.table = {
                'frequency': 1,
                'Y':[ {
                    y: {
                        'frequency': 1,
                        'X':[ {
                            x:{'frequency':1}
                        } for x in X]
                    }} for y in Y]
            }


        for (index, value) in [(Y.index(y), y) for o in O]:
            if value in self.table['Y'][index]:
                self.table['Y'][index][value]['freq'] += 1
                #Input frequencyTable
            else:
                self.table['Y'][index][value] = {'freq': 1, 'X':[ {x:{'frequency':1}} for x in X] }


table = {
    'frequency': 1,
    'Y':[ {
        y: {
            'frequency': 1,
            'X':[ {
                x:{'frequency':1}
            } for x in X]
        }} for y in Y]
}
