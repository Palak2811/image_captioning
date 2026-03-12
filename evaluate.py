"""
BLEU Evaluation Script for Image Captioning
Run: python evaluate.py
Results saved to: bleu_results.txt
"""

import torch
import pickle
import os
from PIL import Image
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from models import EncoderCNN, DecoderRNN
from beam_search import beam_search

# ── Config ────────────────────────────────────────────────────────────────────
CAPTIONS_FILE  = "Flickr8k.token.txt"
IMAGE_DIR      = "Flicker8k_Dataset"
ENCODER_WEIGHTS = "encoder_attention.pth"
DECODER_WEIGHTS = "decoder_attention.pth"
WORD2IDX_FILE  = "word2idx.pkl"
IDX2WORD_FILE  = "idx2word.pkl"
RESULTS_FILE   = "bleu_results.txt"

MAX_EVAL_IMAGES = 500   # reduce to 200 if it runs too slow
MAX_CAPTION_LEN = 20
# ──────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_models():
    with open(WORD2IDX_FILE, "rb") as f:
        word2idx = pickle.load(f)
    with open(IDX2WORD_FILE, "rb") as f:
        idx2word = pickle.load(f)

    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN(512, 512, len(word2idx)).to(device)

    encoder.load_state_dict(torch.load(ENCODER_WEIGHTS, map_location=device))
    decoder.load_state_dict(torch.load(DECODER_WEIGHTS, map_location=device))

    encoder.eval()
    decoder.eval()
    return encoder, decoder, word2idx, idx2word


def load_captions(captions_file):
    """Returns dict: {image_name: [caption1, caption2, ...]}"""
    captions = {}
    with open(captions_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            img_id, caption = parts
            img_name = img_id.split("#")[0]
            captions.setdefault(img_name, []).append(caption.lower())
    return captions


def greedy_generate(encoder_out, decoder, word2idx, idx2word):
    """Simple greedy decoding — no shared-state bug."""
    hidden = torch.zeros(1, decoder.hidden_size).to(device)
    cell   = torch.zeros(1, decoder.hidden_size).to(device)

    word = torch.tensor([word2idx["<start>"]]).to(device)
    result = []

    for _ in range(MAX_CAPTION_LEN):
        embed   = decoder.embedding(word)
        context = decoder.attention(encoder_out, hidden)
        lstm_input = torch.cat([embed, context], dim=1)
        hidden, cell = decoder.lstm(lstm_input, (hidden, cell))
        logits  = decoder.fc(hidden)
        word    = logits.argmax(dim=1)

        token = idx2word[word.item()]
        if token == "<end>":
            break
        if token not in ["<start>", "<pad>", "<unk>"]:
            result.append(token)

    return result


def evaluate():
    print("Loading models...")
    encoder, decoder, word2idx, idx2word = load_models()

    print("Loading captions...")
    all_captions = load_captions(CAPTIONS_FILE)

    image_names = [
        img for img in list(all_captions.keys())
        if os.path.exists(os.path.join(IMAGE_DIR, img))
    ][:MAX_EVAL_IMAGES]

    print(f"Evaluating on {len(image_names)} images...")

    references   = []
    hyps_greedy  = []
    hyps_beam    = []
    failed       = 0

    for i, img_name in enumerate(image_names):
        img_path = os.path.join(IMAGE_DIR, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            failed += 1
            continue

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            encoder_out = encoder(img_tensor)
            greedy_hyp  = greedy_generate(encoder_out, decoder, word2idx, idx2word)
            beam_str    = beam_search(encoder_out, decoder, word2idx, idx2word)
            beam_hyp    = beam_str.split()

        refs = [cap.split() for cap in all_captions[img_name]]
        references.append(refs)
        hyps_greedy.append(greedy_hyp)
        hyps_beam.append(beam_hyp)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(image_names)} images...")

    smoother = SmoothingFunction().method1

    def bleu_scores(refs, hyps):
        b1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0))
        b2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0))
        b3 = corpus_bleu(refs, hyps, weights=(0.33, 0.33, 0.33, 0))
        b4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=smoother)
        return b1, b2, b3, b4

    g1, g2, g3, g4 = bleu_scores(references, hyps_greedy)
    b1, b2, b3, b4 = bleu_scores(references, hyps_beam)

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("          BLEU EVALUATION RESULTS")
    print("="*50)
    print(f"  Images evaluated : {len(references)}   Failed: {failed}")
    print(f"  {'Metric':<10} {'Greedy':>10} {'Beam (k=3)':>12}")
    print(f"  {'-'*35}")
    print(f"  {'BLEU-1':<10} {g1:>10.4f} {b1:>12.4f}")
    print(f"  {'BLEU-2':<10} {g2:>10.4f} {b2:>12.4f}")
    print(f"  {'BLEU-3':<10} {g3:>10.4f} {b3:>12.4f}")
    print(f"  {'BLEU-4':<10} {g4:>10.4f} {b4:>12.4f}")
    print("="*50)

    # ── Save results ───────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        f.write("BLEU EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"Model       : ResNet-50 + Attention LSTM\n")
        f.write(f"Dataset     : Flickr8k\n")
        f.write(f"Images      : {len(references)}\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Metric':<10} {'Greedy':>10} {'Beam (k=3)':>12}\n")
        f.write(f"{'-'*35}\n")
        f.write(f"{'BLEU-1':<10} {g1:>10.4f} {b1:>12.4f}\n")
        f.write(f"{'BLEU-2':<10} {g2:>10.4f} {b2:>12.4f}\n")
        f.write(f"{'BLEU-3':<10} {g3:>10.4f} {b3:>12.4f}\n")
        f.write(f"{'BLEU-4':<10} {g4:>10.4f} {b4:>12.4f}\n")
        f.write("="*50 + "\n")
        f.write("\nSAMPLE CAPTIONS (first 10 images)\n")
        f.write("-"*50 + "\n")
        for j in range(min(10, len(references))):
            f.write(f"\nImage     : {image_names[j]}\n")
            f.write(f"Greedy    : {' '.join(hyps_greedy[j])}\n")
            f.write(f"Beam      : {' '.join(hyps_beam[j])}\n")
            f.write(f"Reference : {' '.join(references[j][0])}\n")

    print(f"\nResults saved to '{RESULTS_FILE}'")


if __name__ == "__main__":
    evaluate()
