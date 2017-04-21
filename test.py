import pickle
meta = pickle.load( open( "metadata", "rb" ) )
vocab = meta["vocab"]
non_alpha = 0
space = 0
for i in range(len(vocab)):
    if not vocab[i].isalpha():
        # print(i)
        # print(vocab[i].encode("utf-8"))
        # print(len(vocab[i]))
        non_alpha += 1
    if vocab[i].isspace():
        space += 1
print("Total number of unicode chars: " + str(non_alpha))
print("Total number of space chars (should be 0): " + str(space))