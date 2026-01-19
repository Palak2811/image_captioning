import streamlit as st
import torch
from PIL import Image
import pickle
from torchvision import transforms

from models import EncoderCNN, DecoderRNN
from beam_search import beam_search

st.set_page_config(page_title="Image Caption Generator")
st.title(" Image Caption Generator")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

@st.cache_resource
def load_models():
    with open("word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open("idx2word.pkl", "rb") as f:
        idx2word = pickle.load(f)

    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN(512, 512, len(word2idx)).to(device)

    encoder.load_state_dict(torch.load("encoder_attention.pth", map_location=device))
    decoder.load_state_dict(torch.load("decoder_attention.pth", map_location=device))

    encoder.eval()
    decoder.eval()

    return encoder, decoder, word2idx, idx2word

encoder, decoder, word2idx, idx2word = load_models()

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("Generate Caption"):
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            encoder_out = encoder(img_tensor)
            caption = beam_search(encoder_out, decoder, word2idx, idx2word)

        st.success(caption)
