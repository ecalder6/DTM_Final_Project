import pickle
meta = pickle.load( open( "metadata", "rb" ) )
vocab = meta["vocab"]
for i in range(len(vocab)):
    if not vocab[i].isalpha():
        print(i)
        print(vocab[i])
        print(len(vocab[i]))