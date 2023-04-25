from __future__ import division
from __future__ import print_function


import cv2
import tensorflow as tf
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess


class FilePaths:
    "filenames and paths to data"
    fnCharList = r'C:\Users\Ramis\Downloads\FYP\model\charList.txt'
    fnAccuracy = r'C:\Users\Ramis\Downloads\FYP\model\accuracy.txt'
    fnTrain = '../data/'
    fnInfer = '../data/1.png'
    fnCorpus = '../data/corpus.txt'


def train(model, loader):
    "train NN"
    epoch = 0 # number of training epochs since start
    bestCharErrorRate = float('inf') # best valdiation character error rate
    noImprovementSince = 0 # number of epochs no improvement of character error rate occured
    earlyStopping = 5 # stop training after this number of epochs without improvement
    totalEpoch = 1
    
    while True:
        epoch += 1
        print('Epoch:', epoch, '/', totalEpoch)
        
        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

        # validate
        charErrorRate = validate(model, loader)
        
        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1
            
        if epoch>=totalEpoch:
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
            saver= tf.train.Saver(max_to_keep=1)
            print(saver)
            break
        
        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break
        

def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0],'/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)
        
        print('Ground truth -> Recognized')    
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
    
    # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
    return charErrorRate


def infer(fnImg): #infer(model, fnImg): 
#    "recognize text in image provided by file path"
    tf.reset_default_graph()
    decoderType = DecoderType.BestPath
#    parser = argparse.ArgumentParser()
#    args = parser.parse_args()
#    fnImg=r"C:\Users\Ramis\Downloads\SimpleHTR-master\examsheets\ans1\ans_1_1.png"

    model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)  #, dump=args.dump
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
#    cv2.imshow('test',img)
#    cv2.waitKey(0) 
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0])
    return recognized[0] #added

#def main():
#    "main function"
#  
#    decoderType = DecoderType.BestPath
#    if args.beamsearch:
#        decoderType = DecoderType.BeamSearch
#    elif args.wordbeamsearch:
#        decoderType = DecoderType.WordBeamSearch
#
#    # train or validate on IAM dataset    
#    if args.train or args.validate:
#        # load training data, create TF model
#        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
#
#        # save characters of model for inference mode
#        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
#        
#        # save words contained in dataset into file
#        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))
#
#        # execute training or validation
#        if args.train:
#            model = Model(loader.charList, decoderType)
#            train(model, loader)
#        elif args.validate:
#            model = Model(loader.charList, decoderType, mustRestore=True)
#            validate(model, loader)
#
#    # infer text on test image
#    else:
#        print(open(FilePaths.fnAccuracy).read())
#        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
#        infer(model, FilePaths.fnInfer)
#
#
#if __name__ == '__main__':
#    main()

