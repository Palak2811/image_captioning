import torch
from PIL import Image

#keeps the best sequences 
def beam_search(encoder_out, decoder, word2idx, idx2word, beam_size=3, max_len=20):
    device = encoder_out.device
    sequences = [([word2idx["<start>"]], 0.0)] #initlise the caption

    hidden = torch.zeros(1, decoder.hidden_size).to(device)
    cell = torch.zeros(1, decoder.hidden_size).to(device)

    for _ in range(max_len): 
        all_candidates = []
        for seq, score in sequences: #We try extending each existing candidate sentence.
            word = torch.tensor([seq[-1]]).to(device)#Get the last word
            embed = decoder.embedding(word)#Turn last word into embedding.

            context = decoder.attention(encoder_out, hidden)#Looks at the image based on the current sentence memory.
            lstm_input = torch.cat([embed, context], dim=1)#Combines current word + current visual focus→ updates sentence understanding.
            hidden, cell = decoder.lstm(lstm_input, (hidden, cell))
            logits = decoder.fc(hidden)

            log_probs = torch.log_softmax(logits, dim=1)#Gets probability of every possible next word.
            topk = log_probs.topk(beam_size) # Choose top beam_size next words.

            for i in range(beam_size): #Store new sequences
                candidate = (
                    seq + [topk.indices[0][i].item()],
                    score + topk.values[0][i].item()
                )
                all_candidates.append(candidate)

        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    best = sequences[0][0] #Keep only the best beam_size sequences
    return " ".join([idx2word[idx] for idx in best if idx2word[idx] not in ["<start>", "<end>"]])
