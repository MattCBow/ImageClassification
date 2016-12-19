from ImageClassifier import *
from BayesNetwork import *
import numpy as np
from Perceptron import *
from ModelTrainer import *
import matplotlib.pyplot as plt
from scipy import optimize
from NueralNetwork import *
import time
from scipy.stats import norm
import gc

#---------------------------Main Classifier File--------------------------

class ImageClassifier(object):

    def init_data(self):
        self.data = {
            'digit':{
                'train':{
                    'image_file':{'path':'data/digitdata/trainingimages'},
                    'label_file':{'path':'data/digitdata/traininglabels'}},
                'test':{
                    'image_file':{ 'path':'data/digitdata/testimages'},
                    'label_file':{'path':'data/digitdata/testlabels'}},
                'valid':{
                    'image_file':{'path':'data/digitdata/validationimages'},
                    'label_file':{'path':'data/digitdata/traininglabels'}}},
            'face':{
                'train':{
                    'image_file':{'path':'data/facedata/facedatatrain'},
                    'label_file':{'path':'data/facedata/facedatatrainlabels'}},
                'test':{
                    'image_file':{'path':'data/facedata/facedatatest'},
                    'label_file':{'path':'data/facedata/facedatatestlabels'}},
                'valid':{
                    'image_file':{'path':'data/facedata/facedatavalidation'},
                    'label_file':{'path':'data/facedata/facedatavalidationlabels'}}}}

    def load_data(self):
        for data_class in self.data.values():
            for data_set in data_class.values():
                labels = []
                images = []
                labels_file =  open(data_set['label_file']['path']).readlines()
                image_file = open(data_set['image_file']['path']).readlines()
                for line in labels_file:
                        if not line.isspace(): labels.append(int(line))
                image_height = len(image_file)/len(labels)
                for i in range(0,len(labels)): images.append(image_file[i*image_height:(i+1)*image_height])
                data_set['label_file']['data'] = labels
                data_set['image_file']['data'] = images


    def format_data(self):
        for data_class in self.data.values():
            for data_set in data_class.values():
                images = data_set['image_file']['data']
                labels = data_set['label_file']['data']
                classifications = []
                features = []
                for label in labels:
                    encoded_classification = []
                    encoded_classification.append(label)
                    classifications.append(encoded_classification)
                for image in images:
                    encoded_feature = []
                    for line in image:
                        for char in line:
                            encoded_feature.append(ord(char))
                    features.append(encoded_feature)
                data_set['classification'] = classifications
                data_set['features'] = features

    def __init__(self):
        self.init_data()
        self.load_data()
        self.format_data()

def print_structure(structure, depth):
    ret = ""
    if isinstance(structure, type('a')):
        ret += ('\t'*depth) + (structure) + ('\n')
    if isinstance(structure, type(1)):
        ret += ('\t'*depth) + (str(structure)) + ('\n')
    if isinstance(structure, type(1.0)):
        ret += ('\t'*depth) + (str(structure)) + ('\n')
    if isinstance(structure, type((1,2))):
        ret += ('\t'*depth) + str(structure[0]) + (": ") + print_structure(structure[1], depth)[depth:]
    if isinstance(structure, type([])):
        ret += ('\t'*depth) + ('[') + ('\n')
        for i in range(0,len(structure)): ret += print_structure((i,structure[i]), depth+1)
        ret = ret[:len(ret)-1] + (']') + ('\n')
    if isinstance(structure, type({})) and len(structure):
        ret += ('\t'*depth) + ('{') + ('\n')
        for (k,v) in structure.iteritems(): ret += print_structure((k,v), depth+1)
        ret = ret[:len(ret)-1] + ('}') + ('\n')
    return ret

def print_struct(structure):
    print print_structure(structure, 0)

def main():
    print "Welcome\n"

    classifier = ImageClassifier()

#---------------------------------BAYES-----------------------------------------
    print "\n\n----------BAYES----------"

    data_types = classifier.data.keys()
    data_types.reverse()
    
    for data_type in data_types:
        print "\n",data_type, ":"
        #performance['']
        input = classifier.data[data_type]['train']['features']
        output = classifier.data[data_type]['train']['classification']

        X_train = np.array(input, dtype=int)
        Y_train = np.array(output, dtype=int)
        baynet = BayesNetwork(len(X_train[0]), len(Y_train[0]))

        X_test = np.array(classifier.data[data_type]['test']['features'], dtype=int)
        Y_test = np.array(classifier.data[data_type]['test']['classification'], dtype=int)

        for p in np.arange(0.1,1.01,0.1):
            right = 0
            wrong = 0
            print "\nTraining Bayesian Network with ", int((100*p)), "% of training data."
            train_start = time.time()
            datapoints = len(X_train) * (p)
            gc.collect()
            baynet.train( X_train[:datapoints] , Y_train[:datapoints] )
            predictions = [baynet.forward(X_test[i])[0]['prediction'] for i in range(len(Y_test))]
            costs = predictions - Y_test
            error, std = norm.fit(costs)
            train_end = time.time()
            time_elapsed = (train_end - train_start)

            classifier.data[data_type]['test'][datapoints] = {}
            classifier.data[data_type]['test'][datapoints]['error'] = error
            classifier.data[data_type]['test'][datapoints]['std'] = std
            classifier.data[data_type]['test'][datapoints]['time'] = time_elapsed

            print_struct(classifier.data[data_type]['test'][datapoints])


            '''
            train_start = time.time()
            datapoints = len(X_train) * (p)
            baynet.train( X_train[:datapoints] , Y_train[:datapoints] )
            train_end = time.time()
            print "Training time:", (train_end - train_start), "seconds"
            print "Testing Bayesian Network..."
            #classifier.data[data_type]['test'][]
            for i in range(len(Y_test)):
                if baynet.forward(X_test[i])[0]['prediction'] == Y_test[i][0]:
                    right +=1
                else:
                    wrong +=1
            '''
            #print "Bayesian Network accuracy:" , 100*(float(right)/float(right+wrong)), "%."

#----------------------------PERCEPTRON-----------------------------------------
    print "\n----------PERCEPTRON----------"

    for data_type in classifier.data.keys():
        print "\n",data_type, ":"
        X_train = np.array(classifier.data[data_type]['train']['features'], dtype=float)
        Y_train = np.array(classifier.data[data_type]['train']['classification'], dtype=float)
        X_train = X_train/100
        Y_train = Y_train/10

        X_test = np.array(classifier.data[data_type]['test']['features'], dtype=float)
        Y_test = np.array(classifier.data[data_type]['test']['classification'], dtype=float)
        X_test = X_test/100
        Y_test = Y_test/10
        percep_net = Perceptron(X_train.shape[1],Y_train.shape[1])
        trainer = ModelTrainer(percep_net)

        for p in np.arange(0.1,1.01,0.1):
            right = 0
            wrong = 0
            print "\nTraining Perceptron Network with ", int((100*p)), "% of training data."
            train_start = time.time()
            datapoints = len(X_train) * (p)
            trainer.train( X_train[:datapoints] , Y_train[:datapoints] )
            predictions =  percep_net.forward(X_test)#[percep_net.forward(X_test[i])[0]['prediction'] for i in range(len(Y_test))]
            costs = predictions - Y_test
            error, std = norm.fit(costs)
            train_end = time.time()
            time_elapsed = (train_end - train_start)

            classifier.data[data_type]['test'][datapoints] = {}
            classifier.data[data_type]['test'][datapoints]['error'] = error
            classifier.data[data_type]['test'][datapoints]['std'] = std
            classifier.data[data_type]['test'][datapoints]['time'] = time_elapsed

            print_struct(classifier.data[data_type]['test'][datapoints])

            '''
            train_start = time.time()
            datapoints = len(X_train) * (p)
            trainer.train(X_train[:datapoints],Y_train[:datapoints])
            train_end = time.time()
            print "Training time:", (train_end - train_start), "seconds"
            print "Testing Perceptron Network..."
            for i in range(len(Y_test)):
                ret = percep_net.forward(X_test[i])[0]
                predict = round(10*percep_net.forward(X_test[i])[0])
                expected = round(10*Y_test[i][0])
                if predict == expected:
                    right +=1
                else:
                    wrong +=1
            print "Perceptron Network accuracy:" , 100*(float(right)/float(right+wrong)), "%."
            '''

#-----------------------------NEURAL NET----------------------------------------
    print "\n\n----------NEURAL NET----------"
    for data_type in classifier.data.keys():
        print "\n",data_type, ":"
        X_train = np.array(classifier.data[data_type]['train']['features'], dtype=float)
        Y_train = np.array(classifier.data[data_type]['train']['classification'], dtype=float)
        X_train = X_train/100
        Y_train = Y_train/10

        X_test = np.array(classifier.data[data_type]['test']['features'], dtype=float)
        Y_test = np.array(classifier.data[data_type]['test']['classification'], dtype=float)
        X_test = X_test/100
        Y_test = Y_test/10

        nueral_net = NueralNetwork(input_size=X_train.shape[1],output_size=Y_train.shape[1])
        trainer = ModelTrainer(nueral_net)

        for p in np.arange(0.1,1.01,0.1):
            right = 0
            wrong = 0
            print "\nTraining Neural Network with ", int((100*p)), "% of training data."
            train_start = time.time()
            datapoints = len(X_train) * (p)
            trainer.train( X_train[:datapoints] , Y_train[:datapoints] )
            predictions = nueral_net.forward(X_test)#[nueral_net.forward(X_test[i])[0]['prediction'] for i in range(len(Y_test))]
            costs = predictions - Y_test
            error, std = norm.fit(costs)
            train_end = time.time()
            time_elapsed = (train_end - train_start)

            classifier.data[data_type]['test'][datapoints] = {}
            classifier.data[data_type]['test'][datapoints]['error'] = error
            classifier.data[data_type]['test'][datapoints]['std'] = std
            classifier.data[data_type]['test'][datapoints]['time'] = time_elapsed

            print_struct(classifier.data[data_type]['test'][datapoints])


            '''
            train_start = time.time()
            datapoints = len(X_train) * (p)
            trainer.train(X_train[:datapoints],Y_train[:datapoints])
            train_end = time.time()
            print "Training time:", (train_end - train_start), "seconds"
            print "Testing Neural Network..."
            for i in range(len(Y_test)):
                if nueral_net.forward(X_test[i])[0] == Y_test[i][0]:
                    right +=1
                else:
                    wrong +=1
            print "Neural Network accuracy:" , 100*(float(right)/float(right+wrong)), "%."
            '''




if __name__ == "__main__":
    main()
