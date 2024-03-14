'''
This Program makes the BERT embedding matrix and test-/traindata, using the tokenisation for BERT
First 'getBERTusingColab' should be used to compile the subfiles containing embeddings.
The new test-/traindata files contain original data, with every word unique and corresponding to vector in emb_matrix
'''
from config import *
BERT_MODEL = 'Base'
# When BERT instead of BERT_Large, change filenames. Not possible with flags, as "xtilly" differs
# <editor-fold desc="Combining embedding files, retrieved with 'getBERTusingColab">
temp_filenames = ['data/temporaryData/temp_BERT_'+BERT_MODEL+'/part1.txt',
                  'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part2.txt',
                  'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part3.txt',
                  'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part4.txt',
                  'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part5.txt',
                  'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part6.txt',
                  'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part7.txt',
                  'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part8.txt',
                  'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part9.txt',
                  'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part10.txt',
                  'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part11.txt',
                  'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part12.txt',
                  'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part13.txt',
                  'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part14.txt',
                  'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part15.txt']
                #   'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part16.txt',
                #   'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part17.txt',
                #   'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part18.txt',
                #   'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part19.txt',
                #   'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part20.txt',
                #   'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part21.txt',
                #   'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part22.txt',
                #   'data/temporaryData/temp_BERT_'+BERT_MODEL+'/part23.txt']

with open('data/temporaryData/temp_BERT_'+BERT_MODEL+'/BERT_'+BERT_MODEL+'embedding.txt','w') as outf:
    for tfname in temp_filenames:
        with open(tfname) as infile:
            for line in infile:
                if line.startswith("\n") or line.startswith("[CLS]") or line.startswith("[SEP]"):
                    pass
                else:
                    outf.write(line)
with open('data/temporaryData/temp_BERT_'+BERT_MODEL+'/BERT_'+BERT_MODEL+'embedding_withCLS_SEP.txt','w') as outf:
    for tfname in temp_filenames:
        with open(tfname) as infile:
            for line in infile:
                if line.startswith("\n"):
                    pass
                else:
                    outf.write(line)
# </editor-fold>

print("Checkpoint 1 ...")
# <editor-fold desc="make table with unique words">
vocaBERT = []
vocaBERT_SEP = []
unique_words = []
unique_words_index = []
with open('data/temporaryData/temp_BERT_'+BERT_MODEL+'/BERT_'+BERT_MODEL+'embedding_withCLS_SEP.txt') as BERTemb_sep:
    for line in BERTemb_sep:
        word = line.split(" ")[0]
        if word == "[CLS]":
            pass
        else:
            vocaBERT_SEP.append(word)
            if word == "[SEP]":
                pass
            else:
                if word not in unique_words:
                    unique_words.append(word)
                    unique_words_index.append(0)
                vocaBERT.append(word)
# </editor-fold>

# <editor-fold desc="make embedding matrix with unique words, prints counter">
uniqueVocaBERT=[]
with open('data/temporaryData/temp_BERT_'+BERT_MODEL+'/BERT_'+BERT_MODEL+'embedding.txt') as BERTemb:
    with open('data/'+str(FLAGS.year)+str(FLAGS.embedding_type)+'_emb.txt','w') as outfile:
        for line in BERTemb:
            word =  line.split(" ")[0]
            weights = line.split(" ")[1:]
            index = unique_words.index(word)  # get index in unique words table
            word_count = unique_words_index[index]
            unique_words_index[index] += 1
            item = str(word) + '_' + str(word_count)
            outfile.write("%s " % item)
            uniqueVocaBERT.append(item)
            first = True
            for weight in weights[:-1]:
                outfile.write("%s " % weight)
            outfile.write("%s" % weights[-1])
# </editor-fold>,
#BERT_YEAR.DIM.txt is now the embedding matrix with all the words. Shape = (44638,dim)

print("Checkpoint 2 ...")
# <editor-fold desc="make uniqueBERT_SEP variable">
uniqueVocaBERT_SEP =[]
counti =0
for i in range(0,len(vocaBERT_SEP)):
    if vocaBERT_SEP[i] == '[SEP]':
        uniqueVocaBERT_SEP.append('[SEP]')
    else:
        uniqueVocaBERT_SEP.append(uniqueVocaBERT[counti])
        counti +=1
# </editor-fold

# <editor-fold desc="make a matrix (three vectors) containing for each word in bert-tokeniser style:
#   word_id (x_word), sentence_id (x_sent), target boolean (x_targ)">
wot = []
with open('data/temporaryData/temp_BERT_Tiny/wot.txt') as input:
    for line in input:
        if line.startswith("\n") or line.startswith("[CLS]"):
            pass
        else:
            wot.append(line.strip())
lines = open('data/temporaryData/temp_BERT_Tiny/raw.txt').readlines()
index = 0
x_word = []
x_sent = []
x_targ = []
sentenceCount = 0
counti = 0
sentiment=[]
for i in range(0, len(lines), 3):
    sentiment.append(lines[i + 2])
for i in range(0, len(vocaBERT_SEP)):
    x_word.append(i)
    word = vocaBERT_SEP[i]
    x_sent.append(sentenceCount)
    if word == "[SEP]":
        sentenceCount += 1
    if word == wot[counti]:
        counti +=1
        x_targ.append(0)
    else:
        x_targ.append(1)
# </editor-fold>

# <editor-fold desc="print to BERT data to text file">
sentence_senten_unique = ""
sentence_target_unique = ""
sentCount = 0
dollarcount = 0
with open('data/temporaryData/temp_BERT_'+BERT_MODEL+'/unique_BERT_Data_All.txt','w') as outFile:
    for u in range(0,len(uniqueVocaBERT_SEP)):
        if uniqueVocaBERT_SEP[u] == "[SEP]":
            outFile.write(sentence_senten_unique + '\n')
            outFile.write(sentence_target_unique + '\n')
            outFile.write(''.join(sentiment[sentCount]))
            sentence_senten_unique = ""
            sentence_target_unique = ""
            sentCount +=1
        else:
            if x_targ[u] == 1:
                dollarcount += 1
                if dollarcount == 1:
                    sentence_senten_unique += "$T$ "
                sentence_target_unique += uniqueVocaBERT_SEP[u] + ' '
            else:
                dollarcount=0
                sentence_senten_unique += uniqueVocaBERT_SEP[u] + ' '
# </editor-fold>

# <editor-fold desc="Split in train and test file">
linesAllData = open('data/temporaryData/temp_BERT_'+BERT_MODEL+'/unique_BERT_Data_All.txt').readlines()
with open('data/'+str(FLAGS.year)+'train'+str(FLAGS.embedding_type)+'.txt','w') as outTrain, \
        open('data/'+str(FLAGS.year)+'test'+str(FLAGS.embedding_type)+'.txt','w') as outTest:
    for j in range(0, 3938*3):
        outTrain.write(linesAllData[j])
    for k in range(3938*3, len(linesAllData)):
        outTest.write(linesAllData[k])
# </editor-fold>