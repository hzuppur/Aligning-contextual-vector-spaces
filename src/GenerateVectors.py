import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from . import VectorExtraction
from . import DataLoader


def get_output_file_name(prefix, main_model, secondary_model):
    main_model_name = main_model.split("/")[-1]
    secondary_model_name = secondary_model.split("/")[-1]

    if prefix != None:
        return f"{prefix}_{main_model_name}_to_{secondary_model_name}"
    else:
        return f"{main_model_name}_to_{secondary_model_name}"


def generate_vectors_from_data(
        test_dataset,
        n_sents,
        model_1,
        model_2,
        outdir=None,
        prefix=None
):
    tokenizer_1 = AutoTokenizer.from_pretrained(model_1)
    tokenizer_2 = AutoTokenizer.from_pretrained(model_2)

    encoder_1 = AutoModelForSeq2SeqLM.from_pretrained(model_1, output_hidden_states=True).model.encoder
    encoder_2 = AutoModelForSeq2SeqLM.from_pretrained(model_2, output_hidden_states=True).model.encoder

    # Read in the test data
    sents = DataLoader.read_sents(test_dataset, n_sents)
    print(f"Number of sentences: {len(sents)}")
    # Get the embeddings for model 1 and 2
    token_embedings_1 = []
    token_embedings_2 = []

    print("Starting to generate vectors")
    for sent in tqdm(sents, desc="Generating vectors"):
        hs_1, hs_2 = VectorExtraction.get_token_embedings_for_sent(sent, encoder_1, encoder_2, tokenizer_1, tokenizer_2)

        token_embedings_1.extend(hs_1.numpy())
        token_embedings_2.extend(hs_2.numpy())

    # Convert list to numpy array
    token_embedings_1 = np.array(token_embedings_1)
    token_embedings_2 = np.array(token_embedings_2)

    if token_embedings_1.shape[0] != token_embedings_2.shape[0]:
        raise RuntimeError("There is a mismatch of the embeddings count!")

    if outdir is not None:
        np.save(outdir + get_output_file_name(prefix, model_1, model_2), token_embedings_1)
        np.save(outdir + get_output_file_name(prefix, model_2, model_1), token_embedings_2)

    return token_embedings_1, token_embedings_2


def generate_or_load(
        test_dataset,
        n_sents,
        model_1,
        model_2,
        outdir,
        prefix
):
    vec_file_1 = outdir + get_output_file_name(prefix, model_1, model_2) + ".npy"
    vec_file_2 = outdir + get_output_file_name(prefix, model_2, model_1) + ".npy"

    if not os.path.exists(vec_file_1) and not os.path.exists(vec_file_2):
        generate_vectors_from_data(
            test_dataset=test_dataset,
            n_sents=n_sents,
            model_1=model_1,
            model_2=model_2,
            outdir=outdir,
            prefix=prefix
        )
        print(f"Test data generated for {prefix}")

    token_embedings_1 = np.load(vec_file_1)
    token_embedings_2 = np.load(vec_file_2)

    return token_embedings_1, token_embedings_2
