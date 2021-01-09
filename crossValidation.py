import main
import os
import sys
import re
from multiprocessing import Pool
from time import time, sleep

def runTraining(infile, randomSeed, dropR, dimensions, linearSize, maxTcr, maxPep, folds, testFold, validationFold, blosum, save_model):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if blosum != None:
        blosumPath = 'data/BLOSUM' + str(blosum)
    if validationFold == -1:
        name=f'{infile}_TESTING{testFold}_lenpep{maxPep}_lentcr{maxTcr}_drop{dropR}_hid{dimensions}_lin{linearSize}_blosum{blosum}.ckpt'
    else:
        name=f'{infile}_validationFold{validationFold}_testFold{testFold}_lenpep{maxPep}_lentcr{maxTcr}_drop{dropR}_hid{dimensions}_lin{linearSize}_blosum{blosum}.ckpt'
    main.run(infile,
            model_name=name,
            seed=randomSeed,
            dropRate=dropR,
            hid=dimensions,
            linearSize=linearSize,
            maxTcrLen=maxTcr, 
            maxPepLen=maxPep,
            blosum=blosumPath,
            folds=folds,
            testFold=testFold, 
            validationFold=validationFold,
            save_model=save_model)

# Define basic information
infile = 'data/tcrgp_data.txt'
indexfile = re.sub('.txt', '_shuffleIdx.txt', infile)
randomSeed = 29

# Cross Validation Parameters
folds = 10
testingFolds = range(10)
save_model = False

# Parameters for cross Validation
dropRate = [0.3, 0.5]
hid = [16, 64, 128]
maxTcrLen = [None, 10, 15]
maxPepLen = [None, 8, 12]
linearTransformationSize = [32, 64]
blosumMatrixes = [None, 45, 50]

pool = Pool(processes=5)
jobs = []
start = time()

# Set Validation Fold = -1 when we want to test best set of hyperparams
for testFold in testingFolds:
    for maxTcr in maxTcrLen:
        for maxPep in maxPepLen:
            for linearSize in linearTransformationSize:
                for dropR in dropRate:
                    for dimensions in hid:
                        for blosum in blosumMatrixes:
                            #for validationFold in range(10):
                                #if validationFold != testFold:
                                    #if not os.path.exists(indexfile):
                                        #runTraining(infile, randomSeed, dropR, dimensions, linearSize, maxTcr, maxPep, folds, testFold, validationFold, blosum, save_model)
                                    #else:
                                        #proc = pool.apply_async(runTraining, (infile, randomSeed, dropR, dimensions, linearSize, maxTcr, maxPep, folds, testFold, validationFold, blosum, save_model))
                                        #jobs.append(proc)
                            proc = pool.apply_async(runTraining, (infile, randomSeed, dropR, dimensions, linearSize, maxTcr, maxPep, folds, testFold, -1, blosum, True))
                            jobs.append(proc)

while(not all([p.ready() for p in jobs])):
    sleep(5)

pool.close()
pool.join()

print(f'Time taken total: {time() - start}')
