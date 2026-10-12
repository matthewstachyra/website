# Can you align with one added embedding (e.g., "safety" token) vector?

Every token exists in some space for the model. Before language models were large, words had static meanings or embeddings. In models like word2vec, each word got one fixed vector no matter its context. There was no attention to route around information and update the embedding of some token with relevant meaning. Today there exist interventional techniques that operate on weights and activations of a model, but: can you align in the raw embedding space?

Consider the word "safety". It has an initial meaning. (Note with superscript: By the way, this could be a challenge with this approach where words have double or triple duty depending on context, like the word "safety" could be a position in American football, or the catch on a gun). This meaning starts from its form as an embedding vector. This vector shares the same dimensions as every other token modeled by some tokenizer. The scalars per dimension encode some of this initial meaning of "safety", before the model's layers build on it with context. What if a small bit of each of these dimensions, in this particular "safety" ratio, were nudges added to every token in an input? Would the model proceed to be safer? The hypothesis is that the content going in will feel safe to the model. The model will have safe-like concepts activated by the tokens going in. This safety will make it less often misaligned, without making its answers incoherent.

This is a set of experiments investigating a kind of "steering" with raw embeddings. 

Let's follow the rabbit and understand why it doesn't work. Let's normalize reporting negative results. They are still results. The meaning of that word doesn't change.

## Design
The basic idea is to try to undo the misalignment introduced by a LoRA fine-tune of a 14B Qwen model (Qwen2.5-14B-Instruct) from the paper *Model Organisms for Emergent Misalignment* (Turner, Soligo et al., 2025), which builds on *Emergent Misalignment* (Betley et al., 2025). The LoRA was trained on bad medical advice, and the model became misaligned more broadly.

The misaligned model answers 4 questions from the *Emergent Misalignment* paper, 100 times each per condition. The word "rabbit" is the control: it tells us what adding any word does, compared with adding "safety" or no word at all. The added vector is scaled to a fraction of each token's own size; that fraction is the strength.

There is a second model (Qwen2.5-32B-Instruct) that acts as a judge on the misaligned model's outputs. It tells us whether the outputs are aligned, misaligned, or incoherent. Each strength is picked on half of the answers (the one with the most aligned answers) and reported on the other half, so we don't fool ourselves.

Code was hand-designed in Cambridge, and coded by Claude Opus 5.5 in the sky.


## What happens when you add the embedding of "safety"? 

## Inside the model? 

## Other words? "alignment", "virtuous", "healthy".

## Do we beat learned directions? 

## More, better? 

