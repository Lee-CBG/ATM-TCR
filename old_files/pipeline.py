import main
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Define hyperparameters and basic information
infile = 'data/tcrgp_data.txt'
indepData = None
blosumPath = None
batchSize = 100
name = 'original.ckpt'
epochs = 200
alpha = 0.001
cudaFlag = True
randomSeed = 621
modeType = 'train'
modelType = 'cnn'

# Train a model based on the defined paramaters
main.run(infile, indepfile=indepData, validationFold=-1, blosum=blosumPath, batch_size=batchSize, model_name=name, epoch=epochs, lr=alpha, cuda=cudaFlag, seed=randomSeed, mode=modeType, model=modelType, save_model=True)

