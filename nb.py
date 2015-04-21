import sys
import math
import numpy
 
class NaiveBayes: 
    def __init__(self): 
        print '__init__' 

    def readInput(self, file_name): 
        vocabulary = set()
        dataset = {}
        
        self.classes = {}
        prior = {}
        
        docs = 0 
        f = open(file_name, 'r') 
        while True: 
            line = f.readline() 
            if line.strip() == '': 
                break 
            docs += 1 
            term = line.split(' ')
            lclass = term[0] 
            if not self.classes.get(lclass):
                dataset[lclass]={} 
                self.classes[lclass]=1
            else: 
                self.classes[lclass] += 1
            for wordc in term[1:]:
                word, count = wordc.split(':')
                count = int(count)
                if not word in dataset[lclass].keys():
                    dataset[lclass][word] = count
                else:
                    dataset[lclass][word] += count 
                vocabulary.add(word)
                
        for cl,nc in self.classes.iteritems():
            prior[cl] = (nc+0.0)/docs
        
        print 'vocab len: ', len(vocabulary)
        print 'number of documents: ', docs 
        print 'classes: ', self.classes
        print 'prior: ', prior, '\n\n'
        for i,j in dataset.iteritems():
            #print i, j
            pass 
 
        return vocabulary, dataset, prior, docs 
 
    def trainClassifier(self, file_name):
        vocab, dataset, prior, N = self.readInput(file_name) 
        print len(vocab), len(dataset), prior, N
        condProb = {}

        for classes in dataset.keys():
            condProb[classes] = {}
        for cl, words in dataset.iteritems():
            denom = 0.0
            for t in vocab:
                if words.get(t):
                    denom += words[t] + 1
                else:
                    denom += 1
            for t in vocab:
                if words.get(t):
                    condProb[cl][t] = (words[t] + 1.0) / denom 
                else:
                    condProb[cl][t] = 1.0 / denom 
            #print cl 
        #for cl, cprob in condProb.iteritems():
            #print word, freq
        return vocab, prior, condProb 
        

    def runClassifier(self, v,p,cp,file_name):
        with open(file_name,'r') as f:
            testWords = {}
            classified = {}

            for document in f:

                document = document.strip()
                classified[document] = {'actual': 0, 'predicted': 0}

                terms = document.split()



                classified[document]['actual'] = int(terms[0])
                score = {}

                for cl, ignore in self.classes.iteritems():
                    intCl = int(cl)
                    score[intCl] = math.log(p[cl])
                    for term in terms[1:]:
                        word, freq = term.split(':')
                        if word in v:
                            score[intCl] += math.log(cp[cl][word])*int(freq)

                argmax = 0
                temp = -float('inf')
                for key in score:

                    if (score[key] > temp):
                        temp = score[key]
                        argmax = key

                #argmax = score.index(max(score))
                classified[document]['predicted'] = argmax
                #print classified[document]
        correct = 0.0
        false = 0.0
        for document in classified:
            if classified[document]['actual'] == classified[document]['predicted']:
                correct += 1
            else:
                false += 1
        print correct
        print false

        cf = numpy.zeroes(len(self.classes),len(self.classes))

        for doc in classified:


 
if __name__ == '__main__': 
    #if len(sys.argv) != 2: 
    #        print("Format:python NaiveBayes.py") 
    #        sys.exit()
    #name = sys.argv[1]

    training_file= "train_email.txt"
    testing_file = "test_email.txt"
    nb = NaiveBayes() 
    v,p,cp = nb.trainClassifier(training_file)               #Train the classifier 
    nb.runClassifier(v,p,cp,testing_file)                  # Run  the classifier
    nb.findMLW(v,p,cp,training_file)