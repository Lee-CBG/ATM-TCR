import os
infile = 'tcrgp_data.txt'

def list_diff(list1, list2):
    out = [item for item in list1 if not item in list2]
    return out

# Cross Validation Parameters
folds = 10
testingFolds = range(10)

# Parameters for cross Validation
dropRate = [0.3, 0.5]
hid = [16, 64, 128]
maxTcrLen = [None, 10, 15]
maxPepLen = [None, 8, 12]
linearTransformationSize = [32, 64]
blosumMatrixes = [None, '45', '50']

"""
perf_tcrgp_data.txt_validationFold8_testFold0_lenpepNone_lentcr15_drop0.3_hid64_lin64_blosumNone.csv
"""
fileNames = os.listdir(os.getcwd() + '/result')
notCompleted = []

for testFold in testingFolds:
    for maxTcr in maxTcrLen:
        for maxPep in maxPepLen:
            for linearSize in linearTransformationSize:
                for dropR in dropRate:
                    for dimensions in hid:
                        for blosum in blosumMatrixes:
                            for validationFold in range(10):
                                if validationFold != testFold:
                                    name=f'perf_{infile}_validationFold{validationFold}_testFold{testFold}_lenpep{maxPep}_lentcr{maxTcr}_drop{dropR}_hid{dimensions}_lin{linearSize}_blosum{blosum}.csv'
                                    if name not in fileNames:
                                        notCompleted.append(name)
                            name=f'perf_{infile}_validationFold{validationFold}_testFold{testFold}_lenpep{maxPep}_lentcr{maxTcr}_drop{dropR}_hid{dimensions}_lin{linearSize}_blosum{blosum}.csv'
                            if name not in fileNames:
                                notCompleted.append(name)

print('THESE ARE FINISHED')
inResults = list_diff(fileNames, notCompleted)
for i in inResults:
    print(i, end="\n")
print('THESE ARE NOT FINISHED')
for i in notCompleted:
    print(i, end="\n")
