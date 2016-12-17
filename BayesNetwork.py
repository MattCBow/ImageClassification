#Artificial Intelligence Bayes Network Implementation

class BayesNetwork(object):

    #-----------RANDOM STARTER WEIGHTS---------------

    def probability_y(self, X):
        p_y = [{} for y in range(0,self.output_size)]
        for y_index in range(len(p_y)):
            for y_value in self.model['Y'][y_index]:
                p_y[y_index][y_value] = (1.0*self.model['Y'][y_index][y_value])/self.model['samples']
                for (x_index, x_value) in [(x_index, X[x_index]) for x_index in range(self.input_size)]:
                    p_y[y_index][y_value] *= (1.0*self.model['X|Y'][y_index][y_value][x_index][x_value])/self.model['X'][x_index][x_value]
        return p_y

    def forward_X(self, X):
        for (x_index, x_value) in [(x_index, X[x_index]) for x_index in range(self.input_size)]:
            if x_value in self.model['X'][x_index]:
                self.model['X'][x_index][x_value] += 1
            else:
                self.model['X'][x_index][x_value] = 1

    def forward_Y(self, Y):
        for (y_index, y_value) in [(y_index, Y[y_index]) for y_index in range(self.output_size)]:
            if y_value in self.model['Y'][y_index]:
                self.model['Y'][y_index][y_value] += 1
            else:
                self.model['Y'][y_index][y_value] = 1

    def forward_X_Y(self, X, Y):
        for (y_index, y_value) in [(y_index, Y[y_index]) for y_index in range(self.output_size)]:
            if y_value not in self.model['X|Y'][y_index]:
                self.model['X|Y'][y_index][y_value] = [ {} for x in range(0,self.input_size)]

            for (x_index, x_value) in [(x_index, X[x_index]) for x_index in range(self.input_size)]:
                if x_value in self.model['X|Y'][y_index][y_value][x_index]:
                    self.model['X|Y'][y_index][y_value][x_index][x_value] += 1
                else:
                    self.model['X|Y'][y_index][y_value][x_index][x_value] = 1

    def forward(self, X, Y):
        if len(X) is not self.input_size or len(Y) is not self.output_size:
            print "Incorrect Data Dimensions"
            return
        self.model['samples'] += 1
        self.forward_X(X)
        self.forward_Y(Y)
        self.forward_X_Y(X,Y)

    def new_model(self):
        self.model = {
            'samples':0,
            'X': [{} for x in range(0,self.input_size)],
            'Y': [{} for y in range(0,self.output_size)],
            'X|Y': [{} for y in range(0,self.output_size)],
            }

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.new_model()


#-------------------END OF BAYES NETWORK----------------------

def main():
    print "Welcome"

"""
from ImageClassifier import *
from BayesNetwork import *
input = [[a,b,c] for a in range(3) for b in range(3) for c in range(3) for i in range(2*2)]
output = [[a,b] for i in range(3*3*3) for a in range(2) for b in range(2)]
baynet = BayesNetwork(len(input[0]), len(output[0]))
for i in range(len(input)):
    print str(input[i]) +' ---> ' +str(output[i])
    baynet.forward(input[i], output[i])

print_struct(baynet.model)
print_struct(baynet.probability_y([0,0,0]))


"""

def p_y(baynet, X):
    p_y = [{} for y in range(0,baynet.output_size)]
    for y_index in range(len(p_y)):
        for y_value in baynet.model['Y'][y_index]:
            p_y = (1.0*baynet.model['Y'][y_index][y_value])/baynet.model['samples']
            p_x = 1.0
            p_x_y = 1.0
            for (x_index, x_value) in [(x_index, X[x_index]) for x_index in range(baynet.input_size)]:
                p_x *= (1.0*baynet.model['X'][x_index][x_value])/baynet.model['samples']
                p_x_y *= (1.0*baynet.model['X|Y'][y_index][y_value][x_index][x_value])/baynet.model['Y'][y_index][y_value]
            p_y[y_index][y_value] = {
                            'p_y': p_y,
                            'p_x': p_x,
                            'p_x_y': p_x_y,
                            'value': (p_x_y*p_y/p_x),
            }
    return p_y


if __name__ == "__main__":
    main()
