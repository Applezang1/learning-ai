import re, collections

# Define input text
text = "a sailor went to sea sea sea "+\
                  "to see what he could see see see "+\
                  "but all that he could see see see "+\
                  "was the bottom of the deep blue sea sea sea"

# Define a function that separates the input text into individual words
def initialize_vocabulary(text):
  vocab = collections.defaultdict(int)
  words = text.strip().split()
  for word in words:
      vocab[' '.join(list(word)) + ' '] += 1
  return vocab

# Execute initialize_vocabulary to compute a list of vocabulary words
vocab = initialize_vocabulary(text)

# Print all the words in the vocabulary and the size of the vocabulary
print('Vocabulary: {}'.format(vocab))
print('Size of vocabulary: {}'.format(len(vocab)))

# Define a function that computes tokens from the individual words and computes the frequency of each token
def get_tokens_and_frequencies(vocab):
  tokens = collections.defaultdict(int)
  for word, freq in vocab.items():
      word_tokens = word.split()
      for token in word_tokens:
          tokens[token] += freq
  return tokens

# Execute get_tokens_and_frequencies to compute a list of tokens and their frequencies
tokens = get_tokens_and_frequencies(vocab)

# Print all the tokens and each of their frequencies
print('Tokens: {}'.format(tokens))
print('Number of tokens: {}'.format(len(tokens)))

# Define a function that computes letter pairs and their frequencies
def get_pairs_and_counts(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

# Execute get_pairs_and_counts to compute all the letter pairs and each of their frequences
pairs = get_pairs_and_counts(vocab)

# Print all the letter pairs and their frequencies
print('Pairs: {}'.format(pairs))
print('Number of distinct pairs: {}'.format(len(pairs)))

# Compute and print the most frequent letter pair
most_frequent_pair = max(pairs, key=pairs.get)
print('Most frequent pair: {}'.format(most_frequent_pair))

# Define a function that merges letter pairs in the vocabulary
def merge_pair_in_vocabulary(pair, vocab_in):
    vocab_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab_in:
        word_out = p.sub(''.join(pair), word)
        vocab_out[word_out] = vocab_in[word]
    return vocab_out

# Execute merge_pair_in_vocabulary and print the new vocabulary
vocab = merge_pair_in_vocabulary(most_frequent_pair, vocab)

# Print all the words in the vocabulary and the vocabulary size
print('Vocabulary: {}'.format(vocab))
print('Size of vocabulary: {}'.format(len(vocab)))

# Execute get_tokens_and_frequencies to get the new tokens and its frequencies from the new vocabulary
tokens = get_tokens_and_frequencies(vocab)

# Print all the tokens and each of the token's frequencies
print('Tokens: {}'.format(tokens))
print('Number of tokens: {}'.format(len(tokens)))

# Define a tokenization function, which converts input texts into tokens
def tokenize(text, num_merges):
  # Initialize the vocabulary from the input text
  vocab = initialize_vocabulary(text)

  # For each pair merge in a defined amount of pair merges
  for i in range(num_merges):
    # Compute the tokens and their frequency in the vocabulary
    tokens = get_tokens_and_frequencies(vocab)

    # Compute the pairs of adjacent tokens and their frequencies
    pairs = get_pairs_and_counts(vocab)

    # Find the most frequent pair
    most_frequent_pair = max(pairs, key=pairs.get)
    print('Most frequent pair: {}'.format(most_frequent_pair))

    # Merge the most frequent pair in the vocabulary
    vocab = merge_pair_in_vocabulary(most_frequent_pair, vocab)

  # Compute the tokens and their frequency in the vocabulary
  tokens = get_tokens_and_frequencies(vocab)

  return tokens, vocab

# Tokenize the input text
tokens, vocab = tokenize(text, num_merges=22)

# Print the tokens, frequency of tokens, vocabulary, and size of vocabulary
print('Tokens: {}'.format(tokens))
print('Number of tokens: {}'.format(len(tokens)))
print('Vocabulary: {}'.format(vocab))
print('Size of vocabulary: {}'.format(len(vocab)))