 Attention-based Image Captioning with CNN–LSTM and Beam Search

A **production-quality Image Captioning system** that generates natural language descriptions for images using a **pretrained CNN encoder**, an **attention-based LSTM decoder**, and **beam search decoding**.  
The project demonstrates strong fundamentals in **deep learning, sequence modeling, and multimodal AI**.


##  Highlights 

- End-to-end **Vision + NLP** pipeline
- **Attention mechanism** for region-level image understanding
- **Beam Search decoding** for improved sequence generation
- Clean separation of **training vs inference**
- Trained on real-world dataset (**Flickr8k**)
- Fully reproducible and modular codebase



##  Problem Statement

Given an image, automatically generate a **semantically accurate and grammatically fluent caption**, e.g.:

> *"A dog is running through the grass."*

Unlike classification, multiple correct captions can exist for the same image, requiring **sequence-level modeling** rather than pointwise prediction.

---

## Live Demo
You can try out the live application here: https://image--captioning.streamlit.app/

Application Screenshot:-<img width="575" height="759" alt="image" src="https://github.com/user-attachments/assets/45f7a4aa-986f-45e9-8b2d-214f052868e5" />


##  Architecture Overview

Input Image
↓
ResNet-50 CNN (Pretrained, Frozen)
↓
Spatial Feature Maps (49 × 2048)
↓
Attention Mechanism
↓
LSTM Decoder
↓
Beam Search Decoder
↓
Generated Caption



##  Model Components

### Encoder (CNN)
- Backbone: **ResNet-50 (ImageNet pretrained)**
- Classification layers removed
- Outputs spatial features `(49 × 2048)`
- Frozen during training to prevent overfitting

### Decoder (Attention-based LSTM)
- Word Embedding Size: 512
- Hidden Size: 512
- Uses **soft visual attention** at every decoding step
- Learns alignment between image regions and words

### Decoding Strategy
- **Greedy decoding** (baseline)
- **Beam Search (k=3)** for higher-quality captions



##  Dataset

### Flickr8k
- 8,000 images
- 5 captions per image
- Standard benchmark for image captioning

### Preprocessing
- Lowercasing
- Punctuation removal
- Special tokens: `<start>`, `<end>`, `<pad>`
- Vocabulary pruning for efficiency

#Tech Stack
-Deep Learning: PyTorch, Torchvision
-Web Framework: Streamlit
-Data Science & Machine Learning: NumPy, NLTK
-Image Processing: Pillow

##  Training Details

- Loss: Cross Entropy Loss (padding ignored)
- Optimizer: Adam
- Encoder frozen, decoder trained
- Epochs: 3–5 (sufficient for Flickr8k)
- Hardware: Kaggle (GPU)


##  Evaluation

Image captioning is evaluated using **sequence-level metrics**:-

- **BLEU-4 score** used for quantitative evaluation
- Qualitative inspection of generated captions

 Results on Flickr8k:
- BLEU-4 ≈ **0.30 – 0.33** (Attention + Beam Search)


##  Inference

Inference is decoupled from training and performed using:
- Saved model checkpoints
- Beam Search decoding

Example:
Input Image: dog.jpg
Generated Caption: a dog is running in the grass


##  How to Run Locally
 Clone the repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install the dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

## Key Technical Takeaways
Implemented encoder–decoder architectures from scratch

Applied attention mechanisms for visual grounding

Understood sequence generation vs classification

Designed modular ML pipelines with reproducibility in mind

Worked with real-world datasets and evaluation metrics

 Future Work
Scale to MS COCO dataset

Add attention visualization

Experiment with Transformer-based decoders

Compare CNN backbones (ResNet-101, EfficientNet)
