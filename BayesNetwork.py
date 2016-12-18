#Artificial Intelligence Bayes Network Implementation

"""
from ImageClassifier import *
from BayesNetwork import *
import numpy as np
classifier = ImageClassifier()
input = classifier.data['face']['train']['features']
output = classifier.data['face']['train']['classification']
X = np.array(input, dtype=int)
Y = np.array(output, dtype=int)
baynet = BayesNetwork(len(X[0]), len(Y[0]))
baynet.train(X,Y)

p_hat = baynet.forward(X[0])
print_struct(p_hat)

right = 0
wrong = 0
for i in range(len(Y)):
    if baynet.forward(X[i])[0]['prediction'] == Y[i][0]:
        right +=1
    else:
        wrong +=1

"""
import numpy as np
import operator
class BayesNetwork(object):

    #-----------RANDOM STARTER WEIGHTS--------------

    def forward(self, X):
        model = self.samples.copy()
        p_table = [{} for y in range(self.output_size)]
        p_hat = [{} for y in range(self.output_size)]

        O = [[] for i in range(len(model['Y']))]
        O[0] = [[v] for v in model['Y'][0]]
        for i in range(1,len(model['Y'])):
            O[i] = [o[:]+[v] for o in O[i-1] for v in model['Y'][i].keys()]

        possible_outcomes = O[len(model['Y'])-1]
        for Y in possible_outcomes:
            model = self.insert_sample(model, X, Y)

        for y_index in range(self.output_size):
            for y_value in model['Y'][y_index]:
                p_table[y_index][y_value] =  [{} for x in range(self.input_size)]
                for (x_index, x_value) in [(i, X[i])  for i in range(len(X))]:
                    p_table[y_index][y_value][x_index][x_value] = {}
                    s_x = (model['X'][x_index][x_value])
                    s_x_y = model['X|Y'][y_index][y_value][x_index][x_value]
                    p_hat_y_x = (1.0*s_x_y) / (s_x)
                    p_table[y_index][y_value][x_index][x_value]['s_x'] = s_x
                    p_table[y_index][y_value][x_index][x_value]['s_x_y'] = s_x_y
                    p_table[y_index][y_value][x_index][x_value]['p_hat'] = p_hat_y_x
                S_p_hat = [p_table[y_index][y_value][x_index][x_value]['p_hat'] for (x_index, x_value) in [(i, X[i])  for i in range(len(X))]]
                p_hat[y_index][y_value] = sum(S_p_hat)/len(S_p_hat)
            prediction = max(p_hat[y_index].iteritems(), key=operator.itemgetter(1))[0]
            p_hat[y_index]['prediction'] = prediction
        return p_hat


    def insert_sample_X(self, model, X):
        for (x_index, x_value) in [(x_index, X[x_index]) for x_index in range(self.input_size)]:
            if x_value in model['X'][x_index]:
                model['X'][x_index][x_value] += 1
            else:
                model['X'][x_index][x_value] = 1
        return model

    def insert_sample_Y(self, model, Y):
        for (y_index, y_value) in [(y_index, Y[y_index]) for y_index in range(self.output_size)]:
            if y_value in model['Y'][y_index]:
                model['Y'][y_index][y_value] += 1
            else:
                model['Y'][y_index][y_value] = 1
        return model

    def insert_sample_X_Y(self, model, X, Y):
        for (y_index, y_value) in [(y_index, Y[y_index]) for y_index in range(self.output_size)]:
            if y_value not in model['X|Y'][y_index]:
                model['X|Y'][y_index][y_value] = [ {} for x in range(0,self.input_size)]

            for (x_index, x_value) in [(x_index, X[x_index]) for x_index in range(self.input_size)]:
                if x_value in model['X|Y'][y_index][y_value][x_index]:
                    model['X|Y'][y_index][y_value][x_index][x_value] += 1
                else:
                    model['X|Y'][y_index][y_value][x_index][x_value] = 1
        return model

    def insert_sample(self, model, X, Y):
        if len(X) == self.input_size and len(Y) == self.output_size:
            model['total'] += 1
            model = self.insert_sample_X(model, X)
            model = self.insert_sample_Y(model, Y)
            model = self.insert_sample_X_Y(model, X,Y)
        else:
            print "Incorrect Data Dimensions: "+str(len(X))+"x"+str(len(Y))+" --> "+str(self.input_size)+"x"+str(self.output_size)
        return model

    def init_samples(self):
        self.samples = {
            'total':0,
            'X': [{} for x in range(0,self.input_size)],
            'Y': [{} for y in range(0,self.output_size)],
            'X|Y': [{} for y in range(0,self.output_size)],
            }

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.init_samples()

    def train(self, X, Y, percentage):
        for i in range(int(percentage*len(X))):
            self.samples = self.insert_sample(self.samples, X[i], Y[i])
