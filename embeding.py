from gensim.models.word2vec import Word2Vec
import gzip

f = gzip.open('D:\Wikipedia\OpenSubtitles2018.en.gz')
data = f.readlines()
for i in range(len(data)):
    data[i] = data[i][:-1].decode("utf-8") 
    
model = Word2Vec(data, size=300, min_count=5,window=10, workers=7)
model.save("model-subtitles")
print('done')
print('done')