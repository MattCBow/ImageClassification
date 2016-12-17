#Artificial Intelligence Bayes Network Implementation

"""
from ImageClassifier import *
from BayesNetwork import *
import numpy as np
classifier = ImageClassifier()
input = classifier.data['face']['test']['features']
output = classifier.data['face']['test']['classification']
#input = np.array([[a,b,c] for a in range(3) for b in range(3) for c in range(3) for i in range(2*2)], dtype=int)
#output = np.array([[a,b] for i in range(3*3*3) for a in range(2) for b in range(2)], dtype=int)

X = np.array(input, dtype=int)
Y = np.array(output, dtype=int)

baynet = BayesNetwork(len(X[0]), len(Y[0]))
baynet.train(X,Y)

baynet.prediction_Y(X[0])
[baynet.prediction_Y(X[i]) for i in range(10)]

print_struct(baynet.model)
print_struct(baynet.probability_Y([0,0,0]))
print str(baynet.prediction_Y([0,0,0]))

classifier.data['face']['test']['image_file'][0]
"""

class BayesNetwork(object):

    #-----------RANDOM STARTER WEIGHTS--------------

    def prediction_Y(self, X):
        return self.p_Y(X)[0]

    def probability_Y(self, X):
        return self.p_Y(X)[1]

    def p_Y(self, X):
        if len(X) != self.input_size:
            print "Incorrect Data Dimensions"
            return
        probability_Y = [{} for y in range(self.output_size)]
        prediction_Y = [0 for y in range(self.output_size)]
        for y_index in range(len(probability_Y)):
            prediction_Y[y_index] = self.model['Y'][y_index].keys()[0]
            for y_value in self.model['Y'][y_index].keys():
                p_Y = (1.0*self.model['Y'][y_index][y_value]) / (self.model['samples'])
                p_X = 1.0
                p_X_Y = 1.0
                p = 1.0
                s_y = (1.0 * self.model['Y'][y_index][y_value])
                p *= p_Y #/s_y


                for (x_index, x_value) in [(x_index, X[x_index]) for x_index in range(self.input_size)]:
                    '''
                    p_x_i = (1.0*self.model['X'][x_index][x_value]) / (self.model['samples'])
                    p_x_i_Y = (1.0)  / (self.model['Y'][y_index][y_value])
                    if x_index in self.model['X|Y'][y_index][y_value][x_index]:
                        p_x_i_Y *= (1.0 * self.model['X|Y'][y_index][y_value][x_index][x_value])  / (self.model['Y'][y_index][y_value])
                    p_X *= p_x_i
                    p_X_Y *= p_x_i_Y
                    p *= p_x_i_Y / p_x_i
                    '''
                    s_x_y = 1.0
                    if x_index in self.model['X|Y'][y_index][y_value][x_index]:
                        s_x_y = (1.0 * self.model['X|Y'][y_index][y_value][x_index][x_value])
                    s_u = (self.model['samples'])
                    s_x = (self.model['X'][x_index][x_value])
                    s_y = (1.0 * self.model['Y'][y_index][y_value])
                    p *= (s_x_y * s_u) / (s_x * s_y)
                    #print 'For y_value:'+str(y_value) +'\tx_index:'+str(x_index)+'\t s_u:'+str(s_u) +'\t s_x_y:'+str(s_x_y) +'\t s_x:'+str(s_x)+'\t p:'+str(p)


                #print 'For y_index:'+str(y_index) +'\ty_value:'+str(y_value)+'\tp:'+str(p)# +'\tp_Y:'+str(p_Y) +'\tp_X:'+str(p_X) +'\tp_X_Y:'+str(p_X_Y) +'\tp:'+str(p)

                probability_Y[y_index][y_value] = p #(p_X_Y*p_Y/p_X)

                if probability_Y[y_index][y_value] > probability_Y[y_index][prediction_Y[y_index]]:
                    prediction_Y[y_index] = y_value
        return (prediction_Y, probability_Y)

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
        if len(X) == self.input_size and len(Y) == self.output_size:
            self.model['samples'] += 1
            self.forward_X(X)
            self.forward_Y(Y)
            self.forward_X_Y(X,Y)
        else:
            print "Incorrect Data Dimensions: "+str(len(X))+"x"+str(len(Y))+" --> "+str(self.input_size)+"x"+str(self.output_size)


    def train(self, X, Y):
        for i in range(len(X)): self.forward(X[i], Y[i])

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
