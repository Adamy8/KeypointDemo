import pickle

with open('AwA-Pose/Annotations/antelope/antelope_10003.pickle', 'rb') as f:
    x = pickle.load(f)
    print(x)
