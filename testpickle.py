import pickle

with open('AwA-Pose/Annotations/cow/cow_11336.pickle', 'rb') as f:
    x = pickle.load(f)
    print(x)
