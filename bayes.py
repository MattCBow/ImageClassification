# bayes classifier
# input: array of pixels, 0's and 1's, representing a digit
import numpy

numDigs = 10
rows = 20;#more for facial
cols = 30;
array = []
digit = 0
feature_prob_table = numpy.zeros((2*rows*cols, 20));
global n0
n0 = 0
n1, n2, n3, n4, n5, n6, n7, n8, n9 = 0,0,0,0,0,0,0,0,0 # times digit appears

#------------------COMPUTE P(D = d | E)-----------------------
def compP(digit):
    top = 0.1 * topfn(1, 0, digit);
    # 0.1 bc all have same probability. bottfn returns one product
    # representing P(e|D=d)
    # bottom = bottfn(1, 0, 0) + bottfn(1, 0, 1) + bottfn(1, 0, 2)
    # bottom += bottfn(1, 0, 3) + bottfn(1, 0, 4) + bottfn(1, 0, 5)
    # bottom += bottfn(1, 0, 6) + bottfn(1, 0, 7) + bottfn(1, 0, 8)
    # bottom += bottfn(1, 0, 9)
    # bottom *= 0.1
    bottom = 1

    print "P(",digit,"| E ) = ", top/bottom

# args: current prob, current index, current digit
# returns top of bayes equation (without prior of digit)
def topfn(curr, index, digit):
    if index is (rows) - 1:
        return curr * (feature_prob_table[(rows)-1, digit*2] / eval(which(digit)))
    return topfn(curr*(feature_prob_table[index, digit*2] / eval(which(digit))), index+1, digit)

# see above
# bottom of bayes, normalizer?
# probably wont call this since it takes forever
def bottfn(curr, index, digit):
    if index is (rows*cols*2) - 1:
        return curr * (feature_prob_table[(rows*cols*2)-1, (digit*2)+1] / eval(which(digit)))
    return bottfn((curr*feature_prob_table[index, (digit*2)+1] / eval(which(digit))), index+1, digit)

# input: array = [0, 0, 0, ..., 1, 0, 0, 1, 1, ...]
# output: void, feature_prob_table is loaded with values
def table_builder_digit():
    for i in range(0, len(array)):
        if array[i] == 0:
            row = 2 * i # 0 comes first in frequency table
        else:
            row = 2 * i + 1
        for col in range(1, (2*numDigs), 2):# assume none is digit
            if int(col / 2) is digit:
                feature_prob_table[row, col-1] = feature_prob_table[row, col-1] + 1;
                increment(digit)
            else:
                feature_prob_table[row, col] = feature_prob_table[row, col] + 1;

# i realize theres better ways of doing this but
# ill worry about optimizing int he end.
def increment(digit):
    if digit is 0:
        global n0
        n0 += 1
    elif digit is 1:
        global n1
        n1 += 1
    elif digit is 2:
        global n2
        n2 += 1
    elif digit is 3:
        global n3
        n3 += 1
    elif digit is 4:
        global n4
        n4 += 1
    elif digit is 5:
        global n5
        n5 += 1
    elif digit is 6:
        global n6
        n6 += 1
    elif digit is 7:
        global n7
        n7 += 1
    elif digit is 8:
        global n8
        n8 += 1
    elif digit is 9:
        global n9
        n9 += 1

# i realize theres better ways of doing this but
# ill worry about optimizing int he end.
def which(digit):
    if digit is 0:
        return "n0"
    elif digit is 1:
        return "n1"
    elif digit is 2:
        return "n2"
    elif digit is 3:
        return "n3"
    elif digit is 4:
        return "n4"
    elif digit is 5:
        return "n5"
    elif digit is 6:
        return "n6"
    elif digit is 7:
        return "n7"
    elif digit is 8:
        return "n8"
    elif digit is 9:
        return "n9"

# for array in arrays:
    # self(array) = array
array1 = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1]
array2 = [1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
array3 = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1]
array4 = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]
arrays = [array1, array2, array3, array4]
    # digit = what this array (image) corresponds to (known because this is training)
digit1 = 0; # temporary
digit2 = 1;
digit3 = 2;
digit4 = 1;
digits = [digit1, digit2, digit3, digit4]
    # call table_builder_digit(array) to update frequency table

for i in range(0, len(arrays)):
    # set values that get used by above functions
    array = arrays[i]
    digit = digits[i]
    table_builder_digit()

print feature_prob_table[]
