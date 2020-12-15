import main
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Define hyperparameters and basic information
infile = 'data/tcrgp_data.txt'
indepData = 'data/netTCR_training_positive_labeled.txt'
blosumPath = 'data/BLOSUM50'
batchSize = 100
name = sys.argv[1]
epochs = 200
alpha = 0.001
cudaFlag = True
randomSeed = 621
modeType = 'test'
modelType = 'cnn'

# Train a model based on the defined paramaters
main.run(infile, indepfile=indepData, blosum=blosumPath, batch_size=batchSize, model_name=name, epoch=epochs, lr=alpha, cuda=cudaFlag, seed=randomSeed, mode=modeType, model=modelType)
