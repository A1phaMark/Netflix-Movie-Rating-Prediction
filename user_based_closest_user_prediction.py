import math
import sys
import argparse
import numpy as np
from operator import add
from time import time
from pyspark import SparkContext
import time


def getUserAverage(data):
    num = data.map(lambda x: (x[0], 1)).reduceByKey(lambda x,y: x+y)
    total = data.map(lambda x: (x[0], x[2])).reduceByKey(lambda x, y: x+y)
    ave = total.join(num).map(lambda x: (x[0], x[1][0]/x[1][1]))
    return ave


#########################################
def getPearson(data, ave):
    print('running pearson...')
    data = data.map(lambda x: (x[0], (x[1], x[2]))).join(ave)
    data = data.map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1])))

    full = data.join(data).filter(lambda x: x[1][0][0] != x[1][1][0])
    # rate = ( (user1, user2), ( (rate1, rate2), (average1, average2) ) )
    rate = full.map(lambda x: ((x[1][0][0], x[1][1][0]), ((x[1][0][1], x[1][1][1]), (x[1][0][2], x[1][1][2]))))

    # rate = ( (user1, user2), ( (rate1-average1), (rate2-average2) ) )
    rate = rate.map(lambda x: ((x[0][0], x[0][1]), (x[1][0][0]-x[1][1][0], x[1][0][1]-x[1][1][1])))

    # den1 = ( userid_pair, denominator1 )            suqare root of the sum of (r1 - ave)^2
    den1 = rate.map(lambda x: ((x[0][0], x[0][1]), x[1][0]**2)).reduceByKey(lambda x, y: x+y)
    den1 = den1.map(lambda x: ((x[0][0], x[0][1]), math.sqrt(x[1])))

    # den2 = ( userid_pair, denominator2 )
    den2 = rate.map(lambda x: ((x[0][0], x[0][1]), x[1][1] ** 2)).reduceByKey(lambda x, y: x + y)
    den2 = den2.map(lambda x: ((x[0][0], x[0][1]), math.sqrt(x[1])))

    num = rate.map(lambda x: ((x[0][0], x[0][1]), x[1][0]*x[1][1])).reduceByKey(lambda x, y: x+y)

    den = den1.join(den2).map(lambda x: ((x[0][0], x[0][1]), x[1][0]*x[1][1]))
    # return a rdd which has item ( (user1, user2), pearson value )
    ret = num.join(den).map(lambda x: ((x[0][0], x[0][1]), x[1][0]/x[1][1]) if x[1][1] != 0 else ((x[0][0], x[0][1]), 0))
    print("Pearson finished!!!")
    return ret


#########################################
def getCos(data):
    print('running Cosine Similarity...')
    data = data.map(lambda x: (x[1], (x[0], x[2])))

    full = data.join(data).filter(lambda x: x[1][0][0] != x[1][1][0])
    # rate = ((user1, user2), (rate1, rate2))
    rate = full.map(lambda x: ((x[1][0][0], x[1][1][0]), (x[1][0][1], x[1][1][1])))

    # den1 = ( userid_pair, denominator1 )            suqare root of the sum of (r1 - ave)^2
    den1 = rate.map(lambda x: ((x[0][0], x[0][1]), x[1][0]**2)).reduceByKey(lambda x, y: x+y)
    den1 = den1.map(lambda x: ((x[0][0], x[0][1]), math.sqrt(x[1])))

    # den2 = ( userid_pair, denominator2 )
    den2 = rate.map(lambda x: ((x[0][0], x[0][1]), x[1][1] ** 2)).reduceByKey(lambda x, y: x + y)
    den2 = den2.map(lambda x: ((x[0][0], x[0][1]), math.sqrt(x[1])))

    num = rate.map(lambda x: ((x[0][0], x[0][1]), x[1][0]*x[1][1])).reduceByKey(lambda x, y: x+y)

    den = den1.join(den2).map(lambda x: ((x[0][0], x[0][1]), x[1][0]*x[1][1]))
    # return a rdd which has item ( (user1, user2), pearson value )
    ret = num.join(den).map(lambda x: ((x[0][0], x[0][1]), x[1][0]/x[1][1]) if x[1][1] != 0 else ((x[0][0], x[0][1]), 0))
    print("Cosine finished!!!")
    return ret


##############################
def getCase(data):
    print('running Case modification...')
    data = getCos(data)
    print("Case modification finished!!!")
    return data.map(lambda x: ((x[0][0], x[0][1]), x[1]**2))


############# return all (user1, (user2, pearson/cosing)) pair
def helper(p):
    return [(p[0][0], (p[0][1], p[1])), (p[0][1], (p[0][0], p[1]))]

##########################
def predict(data, known, p):
    # ((user1), (user2, pearson/cosine))
    pear = p.flatMap(helper)

    # (user, (movie, rating))
    test = data.map(lambda x: (x[0], (x[1], x[2])))

    # ( (user1, movie, rating), (user2, pearson/cosine) )
    test = test.join(pear).map(lambda x: ((x[0], x[1][0][0], x[1][0][1]), (x[1][1][0], x[1][1][1])))

    # ( (user1, movie, rating), (user2, pearson/cosine) ). Now we have only one (user2, pearson/cosine) pair for each
    # (user1, movie, rating) key, the remaining user2 has highest pearson value with user1 among all users.
    test = test.reduceByKey(lambda x, y: y if y[1] > x[1] else x)

    # ( (user2, movie), (user1, rating) )
    test = test.map(lambda x: ((x[1][0], x[0][1]), (x[0][0], x[0][2])))
    # ( (user2, movie), predicted_rating )
    preddata = known.map(lambda x: ((x[0], x[1]), x[2]))

    # ( (user2, movie), ( (user1, rating), predicted_rating) )
    test = test.join(preddata)
    # ret = ( user1, movie, actual_rating, predicted_rating )
    ret = test.map(lambda x: (x[1][0][0], x[0][1], x[1][0][1], x[1][1]))

    return ret


def mse(data):
    print('calculating MSE...')
    n = data.count()
    ret = data.map(lambda x: (x[2] - x[3])**2).reduce(lambda x, y: x+y)
    ret = ret*1.0/n
    return ret




###############################################################
start = time.time()
sc = SparkContext("local[20]")
sc.setLogLevel("WARN")

test = sc.textFile('test_small_1.txt').map(eval)
rdd = sc.textFile('train_small_1.txt').map(eval)
ave = getUserAverage(rdd)
pearson = getPearson(rdd, ave)
cos = getCase(rdd)


# print(pearson.take(10))

print('predicting data...')
pred = predict(test, rdd, cos)

print(pred.take(30))
accuracy = mse(pred)
print('MSE = ' + str(accuracy))
print('Execution : ' + str(time.time()-start)+' secs')




