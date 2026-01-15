from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
import torch
import torch.nn.functional as F
import numpy as np

# Import transformer model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Print 20 different tokens defined in the tokenizer
np.random.seed(1)
print("Number of tokens in dictionary = %d"%(tokenizer.vocab_size))
for i in range(20):
  index = np.random.randint(tokenizer.vocab_size)
  print("Token: %d "%(index)+tokenizer.decode(torch.tensor(index), skip_special_tokens=True))

# Define a function that randomly chooses a token from the model's probability distribution
def sample_next_token(input_tokens, model, tokenizer):
  # Run the transformer model to get the prediction over the next output
  outputs = model(input_ids = input_tokens['input_ids'], attention_mask = input_tokens['attention_mask'])
  # Compute the probabilities of the prediction
  prob_over_tokens = F.softmax(outputs.logits, dim=-1).detach().numpy()[0,-1]
  # Choose random token according to the probabilities
  next_token = [np.random.choice(tokenizer.vocab_size, p = prob_over_tokens)]

  # Append chosen token to sentence
  output_tokens = input_tokens
  output_tokens["input_ids"] = torch.cat((output_tokens['input_ids'],torch.tensor([next_token])),dim=1)
  output_tokens['attention_mask'] = torch.cat((output_tokens['attention_mask'],torch.tensor([[1]])),dim=1)
  output_tokens['last_token_prob'] = prob_over_tokens[next_token]

  return output_tokens

# Expected output: "The best thing about Bath is that they don't even change or shrink anymore."
# Define input text for the tokenizer
set_seed(0)
input_txt = "The best thing about Bath is"

# Convert input text to tokens using tokenizer
input_tokens = tokenizer(input_txt, return_tensors='pt')

# Run the model using sample_next_token to observe how the sentence is completed
for i in range(10):
    input_tokens = sample_next_token(input_tokens, model, tokenizer)
    print(tokenizer.decode(input_tokens["input_ids"][0], skip_special_tokens=True))

# Define a function that chooses the token with the highest probability from the model's probability distribution
def get_best_next_token(input_tokens, model, tokenizer):
  # Run the transformer model to get the prediction over the next output
  outputs = model(input_ids = input_tokens['input_ids'], attention_mask = input_tokens['attention_mask'])
  # Compute the probabilities of the prediction
  prob_over_tokens = F.softmax(outputs.logits, dim=-1).detach().numpy()[0,-1]
  # Compute the token index with the maximum probability
  next_token = [np.argmax(prob_over_tokens)]

  # Append chosen token to sentence
  output_tokens = input_tokens
  output_tokens["input_ids"] = torch.cat((output_tokens['input_ids'],torch.tensor([next_token])),dim=1)
  output_tokens['attention_mask'] = torch.cat((output_tokens['attention_mask'],torch.tensor([[1]])),dim=1)
  output_tokens['last_token_prob'] = prob_over_tokens[next_token]
  return output_tokens

# Expected output: The best thing about Bath is that it's a place where you can go to
# Define input text for the tokenizer
set_seed(0)
input_txt = "The best thing about Bath is"

# Convert input text to tokens using tokenizer
input_tokens = tokenizer(input_txt, return_tensors='pt')

# Run the model using get_best_next_token to observe how the sentence is completed
for i in range(10):
    input_tokens = get_best_next_token(input_tokens, model, tokenizer)
    print(tokenizer.decode(input_tokens["input_ids"][0], skip_special_tokens=True))

# Define a function that randomly chooses the top K most probable tokens from the model's probability distribution
def get_top_k_token(input_tokens, model, tokenizer, k=20):
  # Run the transformer model to get the prediction over the next output
  outputs = model(input_ids = input_tokens['input_ids'], attention_mask = input_tokens['attention_mask'])
  # Compute the probabilities of the prediction
  prob_over_tokens = F.softmax(outputs.logits, dim=-1).detach().numpy()[0,-1]

  # Sort the probabilities from largest to smallest
  sorted_prob_over_tokens =  np.sort(prob_over_tokens)[::-1]

  # Find the probability at the k'th position
  kth_prob_value = sorted_prob_over_tokens[k]

  # Set all probabilities below the k'th value to zero
  prob_over_tokens[prob_over_tokens<kth_prob_value] = 0

  # Renormalize the non-zero probabilities so that they sum to one
  prob_over_tokens = prob_over_tokens/np.sum(prob_over_tokens)

  # Draw random token
  next_token = np.random.choice(len(prob_over_tokens), 1, replace=False, p=prob_over_tokens)

  # Append token to sentence
  output_tokens = input_tokens
  output_tokens["input_ids"] = torch.cat((output_tokens['input_ids'],torch.tensor([next_token])),dim=1)
  output_tokens['attention_mask'] = torch.cat((output_tokens['attention_mask'],torch.tensor([[1]])),dim=1)
  output_tokens['last_token_prob'] = prob_over_tokens[next_token]
  return output_tokens

# Expected output: The best thing about Bath is that you get to see all the beautiful faces of
# Define input text for the tokenizer
set_seed(0)
input_txt = "The best thing about Bath is"

# Convert input text to tokens using tokenizer
input_tokens = tokenizer(input_txt, return_tensors='pt')

# Run the model using get_top_k_token to observe how the sentence is completed
for i in range(10):
    input_tokens = get_top_k_token(input_tokens, model, tokenizer, k=10)
    print(tokenizer.decode(input_tokens["input_ids"][0], skip_special_tokens=True))

# Define a function that randomly chooses a token from a list of sorted tokens whose cumulative sum doesn't exceed a threshold
def get_nucleus_sampling_token(input_tokens, model, tokenizer, thresh=0.25):
  # Run the transformer model to get the prediction over the next output
  outputs = model(input_ids = input_tokens['input_ids'], attention_mask = input_tokens['attention_mask'])
  # Compute the probabilities of the prediction
  prob_over_tokens = F.softmax(outputs.logits, dim=-1).detach().numpy()[0,-1]

  # Sort the probabilities in decreasing order
  sorted_probs_decreasing = np.sort(prob_over_tokens)[::-1]

  # Compute the cumulative sum of these probabilities
  cum_sum_probs = np.cumsum(sorted_probs_decreasing)

  # Find index where that the cumulative sum is greater than the threshold
  thresh_index = np.argmax(cum_sum_probs>thresh)
  print("Choosing from %d tokens"%(thresh_index))

  # Compute the probability value at the tresh_index
  thresh_prob = sorted_probs_decreasing[thresh_index]

  # Set any probabilities below the tresh_prob to zero
  prob_over_tokens[prob_over_tokens<thresh_prob] = 0

  # Renormalize the probabilities to sum to 1
  prob_over_tokens = prob_over_tokens / np.sum(prob_over_tokens)

  # Draw a random token
  next_token = np.random.choice(len(prob_over_tokens), 1, replace=False, p=prob_over_tokens)

  # Append token to sentence
  output_tokens = input_tokens
  output_tokens["input_ids"] = torch.cat((output_tokens['input_ids'],torch.tensor([next_token])),dim=1)
  output_tokens['attention_mask'] = torch.cat((output_tokens['attention_mask'],torch.tensor([[1]])),dim=1)
  output_tokens['last_token_prob'] = prob_over_tokens[next_token]
  return output_tokens

# Expected output: The best thing about Bath is that it's not a city that has been around
set_seed(0)
# Define input text for the tokenizer
input_txt = "The best thing about Bath is"

# Convert input text to tokens using tokenizer
input_tokens = tokenizer(input_txt, return_tensors='pt')

# Run the model using get_nucleus_sampling_token to observe how the sentence is completed
for i in range(10):
    input_tokens = get_nucleus_sampling_token(input_tokens, model, tokenizer, thresh = 0.2)
    print(tokenizer.decode(input_tokens["input_ids"][0], skip_special_tokens=True))

# Define a function that returns the k'th most likely next token from the model's probability distribution
def get_kth_most_likely_token(input_tokens, model, tokenizer, k):
  # Run the transformer model to get the prediction over the next output
  outputs = model(input_ids = input_tokens['input_ids'], attention_mask = input_tokens['attention_mask'])
  # Compute the probabilities of the prediction
  prob_over_tokens = F.softmax(outputs.logits, dim=-1).detach().numpy()[0,-1]

  # Sort the probabilities from largest to smallest
  sorted_prob_over_tokens = np.sort(prob_over_tokens)[::-1]

  # Find the k'th sorted probability
  kth_prob_value = sorted_prob_over_tokens[k]

  # Locate the position of the token with the k'th probability
  next_token = np.where(prob_over_tokens == kth_prob_value)[0]

  # Append token to sentence
  output_tokens = input_tokens
  output_tokens["input_ids"] = torch.cat((output_tokens['input_ids'],torch.tensor([next_token])),dim=1)
  output_tokens['attention_mask'] = torch.cat((output_tokens['attention_mask'],torch.tensor([[1]])),dim=1)
  output_tokens['last_token_prob'] = prob_over_tokens[next_token]
  output_tokens['log_prob'] = output_tokens['log_prob'] + np.log(prob_over_tokens[next_token])
  return output_tokens

# Expected output: The best thing about Bath is the way you get the most bang outta the
set_seed(0)
# Define input text for the tokenizer
input_txt = "The best thing about Bath is"

# Convert input text to tokens using tokenizer
input_tokens = tokenizer(input_txt, return_tensors='pt')
input_tokens['log_prob'] = 0.0

# Run the model using get_kth_most_likely_token, where k = 1, to observe how the sentence is completed
for i in range(10):
    input_tokens = get_kth_most_likely_token(input_tokens, model, tokenizer, k=1)
    print(tokenizer.decode(input_tokens["input_ids"][0], skip_special_tokens=True))

# Expected output: The best thing about Bath is mixed profits partnerships» buy generic+ Honda throttlecont
# Define input text for the tokenizer
input_txt = "The best thing about Bath is"

# Convert input text to tokens using tokenizer
input_tokens = tokenizer(input_txt, return_tensors='pt')
input_tokens['log_prob'] = 0.0

# Run the model using get_kth_most_likely_token, where k = 2000, to observe how the sentence is completed
for i in range(10):
    input_tokens = get_kth_most_likely_token(input_tokens, model, tokenizer, k=2000)
    print(tokenizer.decode(input_tokens["input_ids"][0], skip_special_tokens=True))

# Define a function that print each beam and its log probability
def print_beams(beams):
  for index,beam in enumerate(beams):
    print("Beam %d, Prob %3.3f: "%(index,beam['log_prob'])+tokenizer.decode(beam["input_ids"][0], skip_special_tokens=True))
  print('---')


# Define beam search, which computes n_beams most likely tokens as its initial beams, makes possible continuation of these beams, and keeps only the top n_beams beams
def do_beam_search(input_tokens_in, model, tokenizer, n_beam=5, beam_length=10):
  # Store beams in a list
  input_tokens['log_prob'] = 0.0

  # Initialize the n_beams most likely tokens as the initial beam
  beams = [None] * n_beam
  for c_k in range(n_beam):
    beams[c_k] = dict(input_tokens_in)
    beams[c_k] = get_kth_most_likely_token(beams[c_k], model, tokenizer, c_k)

  print_beams(beams)

  # For each token in the sequence we will add
  for c_pos in range(beam_length-1):
    # For each computed beam, initialize the n_beams most likely tokens as possible continuations of the beam
    beams_all = [None] * (n_beam*n_beam)
    log_probs_all = np.zeros(n_beam*n_beam)
    # For each current hypothesis
    for c_beam in range(n_beam):
      # For each continuation
      for c_k in range(n_beam):
        # Store the continuation and the probability
        beams_all[c_beam * n_beam + c_k] = dict(get_kth_most_likely_token(beams[c_beam], model, tokenizer, c_k))
        log_probs_all[c_beam * n_beam + c_k] = beams_all[c_beam * n_beam + c_k]['log_prob']

    # Keep the best n_beams sequences with the highest probabilities
    sorted_index = np.argsort(np.array(log_probs_all)*-1)
    for c_k in range(n_beam):
      beams[c_k] = dict(beams_all[sorted_index[c_k]])

    # Print the beams
    print_beams(beams)

  return beams[0]

# Expected output: The best thing about Bath is that it's a place where you don't have to
set_seed(0)
# Define input text for the tokenizer
input_txt = "The best thing about Bath is"

# Convert input text to tokens using tokenizer
input_tokens = tokenizer(input_txt, return_tensors='pt')

# Run the model using do_beam_search to observe how the sentence is completed
n_beams = 5
best_beam = do_beam_search(input_tokens,model,tokenizer)
print("Beam search result:")
print(tokenizer.decode(best_beam["input_ids"][0], skip_special_tokens=True))

