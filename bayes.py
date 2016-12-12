# bayes classifier
# input: array of pixels, 0's and 1's, representing a digit
import numpy

cols=30
rows=20

# extract the frequency table from file (or other class); this is temporary!
feature_prob_table = numpy.ones((2*rows*cols, 20));


def compP(digit):
    top = 0.1;
    col = 2*digit
    # multiply prior prob of any digit with the product of the pr(pixel_x | Digit = digit)
    for i in range(0, cols*rows):
        top = top * feature_prob_table[i, col]
    # divide by the sum of products of P(pixel | D = d) where we sum over all digits
    # 0.1 * (P(E|d0) + P(E|d1) + P(E|d2) + ... + P(E|d9)
    bottom = 0
    for j in range(1, 20, 2):
        product = 1
        if int(i / 2) is digit:
            continue
        for k in range(0, cols*rows):
            product = product * feature_prob_table[k, j]
        bottom = bottom + product
    return top / bottom


p_zero = compP(0)
print p_zero
