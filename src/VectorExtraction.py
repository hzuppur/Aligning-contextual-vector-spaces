import torch


def get_token_ids(tokenizer, ids, tokens):
    # tokenize a empty string
    empty_str_tokenized = tokenizer.encode("", return_tensors="np")[0]
    # if there is only one token, do nothing
    if len(empty_str_tokenized) == 1:
        # return the ids
        return ids
    # If there is two tokens, increment to accout for baginning token
    elif len(empty_str_tokenized) == 2:
        # add offset to id-s to account for beginning token
        ids = [1 + i for i in ids]
        return ids
    else:
        raise NotImplementedError("Tokenizer returned 3 tokens on empty string, this should not happen")


def find_common_tokens(tokenizer_1, tokenizer_2, sent):
    tokens_1 = tokenizer_1.tokenize(sent)
    tokens_2 = tokenizer_2.tokenize(sent)
    # punct_and_space is used to find the next word
    punct_and_space = ["▁", ".", ",", ";", "!", "?"]
    # iterators for model 1 and model 2
    i_1, i_2 = 0, 0
    # id-s for tokens that are same for both model tokenizers
    ids_1, ids_2 = [], []
    # iterate over tokens
    while i_1 < len(tokens_1) and i_2 < len(tokens_2):
        # if tokens match, add id-s to list
        if tokens_1[i_1] == tokens_2[i_2]:
            ids_1.append(i_1)
            ids_2.append(i_2)
            i_1 += 1
            i_2 += 1
        # if tokens do not match, find the index of the next word first token
        else:
            i_1 += 1
            i_2 += 1
            # find the index of first token next word for first model
            while i_1 < len(tokens_1) - 1 and tokens_1[i_1][0] not in punct_and_space:
                i_1 += 1
            # find the index of first token next word for second model
            while i_2 < len(tokens_2) - 1 and tokens_2[i_2][0] not in punct_and_space:
                i_2 += 1

    return get_token_ids(tokenizer_1, ids_1, tokens_1), get_token_ids(tokenizer_2, ids_2, tokens_2)


def get_hidden_states(encoded, token_ids_word, model, layers):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
      Select only those subword token outputs that belong to our word of interest
      and average them."""
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    word_tokens_output = output[token_ids_word]
    return word_tokens_output


def get_token_embedings_for_sent(sent, encoder_1, encoder_2, tokenizer_1, tokenizer_2):
    ids_1, ids_2 = find_common_tokens(tokenizer_1, tokenizer_2, sent)
    # Encode the sentence
    encoded_1 = tokenizer_1.encode_plus(sent, return_tensors="pt")
    encoded_2 = tokenizer_2.encode_plus(sent, return_tensors="pt")

    hidden_states_1 = get_hidden_states(encoded_1, ids_1, encoder_1, [-1])
    hidden_states_2 = get_hidden_states(encoded_2, ids_2, encoder_2, [-1])
    return hidden_states_1, hidden_states_2
