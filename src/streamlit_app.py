from datetime import datetime
import os

os.environ["IS_STREAMLIT"] = "Y"

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import streamlit as st
from get_lyrics.get_lyrics import get_lyrics
from train import TrainModel, ModelConfig
from model_utils import check_model_cache, load_model, predict
from models.lstm import LSTMModel
from prep_data import CleanTextData


wd = os.getcwd()
# Header image
st.image(f"./src/img/logo.png")

# Some default values
MODEL_CACHE = "./src/model_files"
FREETEXT_DEFAULT = "Enter Artist Name..."
DROPDOWN_DEFAULT = "Select Artist..."
SELECT_MODEL_TYPE_DEFAULT = "Select Model Type..."
SELECT_MODEL_DEFAULT = "Select Model..."

# Initial values for some booleans
select_own_artist = False
do_predictions = False

# Torch device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device {DEVICE}")

# If training from scratch, the possible models to use
model_map = {
    "lstm": LSTMModel,
}


def train(text: str, artist: str, model_to_train: nn.Module, model_config: ModelConfig):
    """
    Train a model and show a loss graph
    """
    tnp = TrainModel(
        text=text,
        model_name=artist,
        model=model_to_train,
        model_config=model_config,
        model_save_folder=MODEL_CACHE,
        run_in_streamlit=True,
    )

    losses, model, vocab_to_int, int_to_vocab = tnp.run_training()

    fig, ax = plt.subplots(1, 1, figsize=[16, 8])
    ax.plot(losses)
    ax.set_title("Loss Vs epoch")
    st.write(fig)

    return model, vocab_to_int, int_to_vocab


def prediction(model, vocab_to_int, int_to_vocab):
    """
    Run prediction using user presets
    """
    st.sidebar.header("Prediction")
    top_k = st.sidebar.number_input("Top K words", value=3)

    DEFAULT_STARTING_LYRICS = "Enter Lyrics..."
    starting_lyrics = st.sidebar.text_input(
        "Starting Lyrics", value=DEFAULT_STARTING_LYRICS
    )

    whole_song = False
    if starting_lyrics == DEFAULT_STARTING_LYRICS:
        st.sidebar.write("or...")
        whole_song = st.sidebar.checkbox(label="Generate a whole song")
    else:
        ending_lyrics = None
        length_of_output = st.sidebar.number_input("# Of lyrics to generate", 100)

    if whole_song:
        starting_lyrics = "<songstart>"
        ending_lyrics = "<songend>"
        length_of_output = 1e5
    else:
        starting_lyrics = starting_lyrics
        ending_lyrics = None

    if starting_lyrics != DEFAULT_STARTING_LYRICS:
        prediction = predict(
            model=model,
            device=DEVICE,
            starting_string=starting_lyrics,
            vocab_to_int=vocab_to_int,
            int_to_vocab=int_to_vocab,
            output_len=length_of_output,
            stopping_word=ending_lyrics,
            top_k=top_k,
        )

        st.header("And here is your new masterpiece!")
        st.text(prediction)


"""
## 0. Choose from a preset model...
"""

model_cache = check_model_cache(MODEL_CACHE)

if model_cache == {}:
    st.write("No models found, so you will have to create your own")
    select_own_artist = True
else:
    model_type = st.selectbox(
        "Select Model Type",
        options=[SELECT_MODEL_TYPE_DEFAULT] + list(model_cache.keys()),
    )
    if model_type != SELECT_MODEL_TYPE_DEFAULT:
        model_name = st.selectbox(
            "Select Model", options=[SELECT_MODEL_DEFAULT] + model_cache[model_type]
        )
        if model_name != SELECT_MODEL_DEFAULT:
            model, int_to_vocab, vocab_to_int = load_model(
                f"{MODEL_CACHE}/{model_type}/{model_name}"
            )
            do_predictions = True

run_own_model = st.checkbox("I want to choose my own artist!", value=select_own_artist)

if run_own_model:
    do_predictions = False
    """
    ## 1. Select an artist

    You can either enter an artist you like, or, use one of the pre
    artists that already have models built
    """

    artist = st.text_input(
        "Name of the artist you want to generate", value=FREETEXT_DEFAULT
    )

    artist_chosen = False
    if artist != FREETEXT_DEFAULT:
        artist = artist.replace(" ", "-")
        lyrics = get_lyrics(artist)
        artist_chosen = True

    if artist_chosen:

        lower_text = st.sidebar.checkbox("Lower text?", value=True)
        lemma = st.sidebar.checkbox("Apply lemmatisation?", value=True)
        strip_punctuation = st.sidebar.checkbox("Strip Punctuation?", value=True)
        keep_numbers = strip_punctuation = (
            st.sidebar.checkbox(
                "Keep Numbers? (Alternatively, replace them with <NUMBER> special token)",
                value=True,
            ),
        )

        # TODO: Expose these via the interface
        special_tokens = ["<SONGSTART>", "<SONGEND>", "\n"]
        if not keep_numbers:
            special_tokens.append("<NUMBER>")

        clean_text_args = {
            "lower_text": lower_text,
            "lemma": lemma,
            "strip_punctuation": strip_punctuation,
            "keep_numbers": keep_numbers,
            "special_tokens_to_keep": special_tokens,
        }

        st.header("2. Select the clean text options in the sidebar")
        do_text_cleaning = st.checkbox("Once happy, click this button", value=False)

        if do_text_cleaning:
            # Clean the text data
            ctd = CleanTextData(**clean_text_args)
            clean_text = ctd.clean_data(lyrics)

            # Select the hyperparameters
            st.sidebar.header("Model Hyperparameters")
            seq_size = st.sidebar.number_input(label="Sequence Length", value=32)
            batch_size = st.sidebar.number_input(label="Batch Size", value=16)
            embedding_size = st.sidebar.number_input(label="Embedding Size", value=64)
            hidden_layer_size = st.sidebar.number_input(
                label="Hidden Layer Size", value=64
            )
            n_layers = st.sidebar.number_input(label="Number of layers", value=1)
            dropout = st.sidebar.number_input(
                "Droput", min_value=0.0, max_value=1.0, value=0.4
            )
            gradients_clipping = st.sidebar.number_input(
                label="Gradient Clipping Max Value", value=5
            )
            epochs = st.sidebar.number_input(label="Number of Epochs", value=50)
            model_selection = st.sidebar.selectbox(
                label="Model Type", options=list(model_map.keys())
            )

            model_config = ModelConfig(
                seq_size=seq_size,
                batch_size=batch_size,
                embedding_size=embedding_size,
                hidden_layer_size=hidden_layer_size,
                n_layers=n_layers,
                dropout=dropout,
                gradients_clipping=gradients_clipping,
                epochs=epochs,
            )

            st.header(
                "3. Select the hyperparameters in the sidebar, and the model type you want to train"
            )
            do_train = st.checkbox("Once happy, click here")

            if do_train:
                start_train = datetime.now()
                model, vocab_to_int, int_to_vocab = train(
                    text=clean_text,
                    artist=artist,
                    model_to_train=model_map[model_selection],
                    model_config=model_config,
                )
                end_train = datetime.now()
                train_time = (end_train - start_train).seconds / 60
                st.text(
                    f"""
                You've just trained your great {artist} model
                for {epochs} epochs.
                It took {train_time:,.1f} minutes
                To make some tunes:
                1 - Go back to step 0
                2 - Untick "I Want to choose my own artist!"
                3 - Choose the following:
                 - Model Type: {model_selection}
                 - Model Name: model-{artist}-{epochs}
                and rerun.
                Now you can generate full songs or song extracts using this model.
                Enjoy!
                """
                )


if do_predictions:
    prediction(model=model, vocab_to_int=vocab_to_int, int_to_vocab=int_to_vocab)