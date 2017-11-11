import collections

'''
Read file character by character and return dictionaries 
'''
def read_file(path):
    file = open(path, encoding='utf8')
    data = file.read()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)

    print ('Dataset has', data_size, 'characters', vocab_size, 'unique.')

    dictionary, reverse_dictionary = build_dict(chars)

    return data, dictionary, reverse_dictionary

'''
Create dictionary and reverse dictionary
'''
def build_dict(chars):
    dictionary = {
        ch: i for i, ch in enumerate(chars)
    }
    reverse_dictionary = {
        i: ch for i, ch in enumerate(chars)
    }

    return dictionary, reverse_dictionary