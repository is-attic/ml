import collections


def build_dataset(words):
  count = collections.Counter(words).most_common()
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return dictionary, reverse_dictionary


v, h = build_dataset('this is a sentence'.split())
print(v, h)