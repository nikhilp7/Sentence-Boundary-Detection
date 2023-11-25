import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def split(record: str):
    record = record.rstrip("\n")
    return record.split(" ")[1:]


def read(filename: str):
    records = None
    with open(filename, "r") as data_file:
        all_lines = data_file.readlines()
        records = [split(record) for record in all_lines]
    return np.array(records)


def filter_data(data: np.ndarray):
    indices = []
    for i in range(data.shape[0]):
        record_class = data[i][-1]
        if record_class != "TOK":
            indices.append(i)
            
    return data[indices]


def is_capital(word: str):
    iscapital = 0
    if word is None:
        return iscapital
    
    for c in word:
        ascii_value = ord(c)
        if 90 >= ascii_value >= 65:
            iscapital = 1
            break
    return iscapital


def extract_features(data: np.ndarray):
    feature_list_combined = []
    feature_list_core = []
    feature_list_special = []
    
    word_attributes = get_rank(data[:, 0])
    
    for i in range(data.shape[0]):
        word = data[i][0]
        
        L = None
        R = None
        
        if "." in word:
            L, R = word.split(".")[:2]
        else:
            L = word
            R = ""
    
        if R == "":
            R = None

        lenL = 1 if len(L) < 3 else 0
        capitalL = is_capital(L)
        capitalR = is_capital(R)
        word_class = data[i][-1]
        word_frequency = word_attributes[word][0]
        word_rank = word_attributes[word][1]
        is_numeric = isnumeric(word)
        
        features_combined = np.array([L, R, lenL, capitalL, capitalR, word_frequency, word_rank, is_numeric, word_class])
        features_core = np.array([L, R, lenL, capitalL, capitalR, word_class])
        features_special = np.array([L, word_frequency, word_rank, is_numeric, word_class])
       
        feature_list_combined.append(features_combined)
        feature_list_core.append(features_core)
        feature_list_special.append(features_special)
        
    return feature_list_combined, feature_list_core, feature_list_special


def isnumeric(word: str):
    try:
        float(word)
        return 1
    except:
        return 0

def get_rank(words: np.ndarray):
    unique_words, counts = np.unique(words, return_counts=True)

    current_rank = 1
    scounts = np.argsort(counts)[::-1]
    words = unique_words[scounts]
    counts = counts[scounts]
    
    frequency_table = {}
    prev_freq = 0
    current_rank = 0
    
    for word, frequency in zip(words, counts):
        attr = [frequency, current_rank]
        if frequency != prev_freq:
            prev_freq = frequency
            current_rank += 1
            attr[1] = current_rank
        frequency_table[word] = attr
    return frequency_table  



def norm(word: str):
    chars = list(word)
    vec = list(map(lambda x: ord(x), chars))
    return int(np.linalg.norm(vec))

def label_convert(label: str):
    return 0 if label == 'NEOS' else 1


def accuracy(predictions, original):
    if len(predictions) == len(original):
        labels = np.asarray(predictions == original)
        ntrue = np.count_nonzero(labels)
        return round(ntrue / len(predictions) * 100, 2)
    else:
        raise Exception("Number of records in prediction list and original list must match")
        

def preprocess(data: np.ndarray):
    words_L = data[:, 0]
    
    mask_L = words_L != ""
    data = data[mask_L]
    words_L = data[:, 0]
    words_R = data[:, 1]
    
    words_L = np.asarray(list(map(lambda x: x.lower(), words_L)))
    words_R = np.asarray(list(map(lambda x: x.lower() if type(x) == str else "", words_R)))
    
    word_represent_L = np.asarray(list(map(lambda x: norm(x), words_L)))
    word_represent_R = np.asarray(list(map(lambda x: norm(x), words_R)))
    data[:, 0] = word_represent_L
    data[:, 1] = word_represent_R
  
    data[:, -1] = np.asarray(list(map(lambda x: label_convert(x), data[:, -1])))
    data = np.array([sample.astype(int) for sample in data])
    
    return data
    

def modelling(train_data: np.ndarray, 
              test_data: np.ndarray, 
              result_title: str = "Results", nfeatures: int = 9):

    model = DecisionTreeClassifier()
    
    X_train = train_data[:, 0:nfeatures - 1]
    y_train = train_data[:, -1]
    X_test = test_data[:, 0:nfeatures - 1]
    y_test = test_data[:, -1]
    
    model.fit(X_train, y_train)
    
    train_accuracy = round(model.score(X_train, y_train) * 100, 2)
    predictions = model.predict(X_test)
    test_accuracy = accuracy(predictions, y_test)
    
    print(result_title)
    print("\t - Training accuracy:", train_accuracy)
    print("\t - Testing accuracy:", test_accuracy)


def main():
    
    train_data = preprocess(
        np.asarray(np.load("train_combined.npy", allow_pickle = True)))
    
    test_data = preprocess(
        np.asarray(np.load("test_combined.npy", allow_pickle = True)))
    
    modelling(train_data, test_data, result_title = "Combined dataset results")
    
    train_data = preprocess(
        np.asarray(np.load("train_core.npy", allow_pickle = True)))
    
    test_data = preprocess(
        np.asarray(np.load("test_core.npy", allow_pickle = True)))
    
    modelling(train_data, test_data, result_title = "Core dataset results", nfeatures = 6)
    
    
    train_data = preprocess(
        np.asarray(np.load("train_special.npy", allow_pickle = True)))
    
    test_data = preprocess(
        np.asarray(np.load("test_special.npy", allow_pickle = True)))
    
    modelling(train_data, test_data, result_title = "Special dataset results", nfeatures = 4)


if __name__ == "__main__":
    nargv = len(sys.argv)
    if nargv < 3:
        raise Exception("Invalid number of parameters. Try again")
    
    argv = sys.argv
    train = argv[1]
    test = argv[2]
        
    if train != "SBD.train" or test != "SBD.test":
        raise Exception("Dataset names do not match")
        
    try:
        train_data = read(train)
        test_data = read(test)
                
        train_data = filter_data(train_data)
        test_data = filter_data(test_data)
                
        trfeature_combined, trfeature_core, trfeature_special = extract_features(train_data)
        tefeature_combined, tefeature_core, tefeature_special = extract_features(test_data)
        
        np.save("train_combined", trfeature_combined)
        np.save("test_combined", tefeature_combined)
        np.save("train_core", trfeature_core)
        np.save("test_core", tefeature_core)
        np.save("train_special", trfeature_special)
        np.save("test_special", tefeature_special)
        
        main()
        
        
    except:
        raise Exception("Exception occured while reading")