    # bayes classifier
    # input: array of pixels, 0's and 1's, representing a digit
import numpy
import math
from ImageClassification import ImageClassifier

class LearnBayes():
    global images
    global image
    global feature_prob_table
    global numDigs
    numDigs = 10
    global rows
    rows = 20;#more for facial
    global cols
    cols = 30;
    global digit
    global n0, n1, n2, n3, n4, n5, n6, n7, n8, n9
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = 0,0,0,0,0,0,0,0,0,0 # times digit appears

    #------------------COMPUTE P(D = d | E)-----------------------
    def compP():
        top = 0.1 * topfn(1, 0, digit);
        # 0.1 bc all have same probability. bottfn returns one product
        # representing P(e|D=d)
        # probably wont call this since it takes forever
        # bottom = bottfn(1, 0, 0) + bottfn(1, 0, 1) + bottfn(1, 0, 2)
        # bottom += bottfn(1, 0, 3) + bottfn(1, 0, 4) + bottfn(1, 0, 5)
        # bottom += bottfn(1, 0, 6) + bottfn(1, 0, 7) + bottfn(1, 0, 8)
        # bottom += bottfn(1, 0, 9)
        # bottom *= 0.1
        bottom = 1
        print "P(",digit,"| E ) = ", top/bottom


    # args: current prob, current index, current digit
    # returns top of bayes equation (without prior of digit)
    def topfn(curr, index):
        if index is (rows) - 1:
            return curr * (feature_prob_table[(rows)-1, digit*2] / eval(which(digit)))
        return topfn(curr*(feature_prob_table[index, digit*2] / eval(which(digit))), index+1, digit);

    # see above
    # bottom of bayes, normalizer?
    # probably wont call this since it takes forever
    def bottfn(curr, index):
        if index is (rows*cols*2) - 1:
            return curr * (feature_prob_table[(rows*cols*2)-1, (digit*2)+1] / eval(which(digit)))
        return bottfn((curr*feature_prob_table[index, (digit*2)+1] / eval(which(digit))), index+1, digit)

    # input: array = [0, 0, 0, ..., 1, 0, 0, 1, 1, ...]
    # output: void, feature_prob_table is loaded with values
    def table_builder_digit(self):
        for i in range(0, len(image)):
            if image[i] == (35 or 43):# THIS NEEDS TO BE CHANGED
                row = 2 * i + 1# 0 comes first in frequency table
            else:
                row = 2 * i
            for col in range(1, (2*numDigs), 2):# assume none is digit
                if int(math.ceil(col / 2)) == digit[0]:
                    feature_prob_table[row, col-1] = feature_prob_table[row, col-1] + 1;
                    self.increment(digit)
                else:
                    feature_prob_table[row, col] = feature_prob_table[row, col] + 1

    # i realize theres better ways of doing this but
    # ill worry about optimizing int he end.
    def increment(self, digitm):
        if digitm is 0:
            global n0
            n0 += 1
        elif digitm is 1:
            global n1
            n1 += 1
        elif digitm is 2:
            global n2
            n2 += 1
        elif digitm is 3:
            global n3
            n3 += 1
        elif digitm is 4:
            global n4
            n4 += 1
        elif digitm is 5:
            global n5
            n5 += 1
        elif digitm is 6:
            global n6
            n6 += 1
        elif digitm is 7:
            global n7
            n7 += 1
        elif digitm is 8:
            global n8
            n8 += 1
        elif digitm is 9:
            global n9
            n9 += 1

    # i realize theres better ways of doing this but
    # ill worry about optimizing int he end.
    def which(digitm):
        if digitm is 0:
            return "n0"
        elif digitm is 1:
            return "n1"
        elif digitm is 2:
            return "n2"
        elif digitm is 3:
            return "n3"
        elif digitm is 4:
            return "n4"
        elif digitm is 5:
            return "n5"
        elif digitm is 6:
            return "n6"
        elif digitm is 7:
            return "n7"
        elif digitm is 8:
            return "n8"
        elif digitm is 9:
            return "n9"

    def __init__(self, array_of_imgarrays, array_of_corrdigs):
        rows = 20;#more for facial
        cols = 30;
        global images, digits, feature_prob_table
        images = array_of_imgarrays
        digits = array_of_corrdigs
        feature_prob_table = numpy.zeros((2000, 20))
        for i in range(0, 5):
            # set values that get used by above functions
            global image, digit
            image = images[i]
            digit = digits[i]
            self.table_builder_digit()
        print feature_prob_table

    def main():
        y = ImageClassifier()
        #print classifier.data['digit']['test']['features'][0:2]
        #print [len(f) for f in classifier.data['digit']['test']['features'][0:1000:10]]


classifier = ImageClassifier()
bayes_digit_learn = LearnBayes(classifier.data['digit']['train']['features'], classifier.data['digit']['train']['classification'])
