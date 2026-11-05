---
title: From Recurrence to Attention
topic: Transformers vs RNNs/LSTMs — the architecture, what it changed about training, and what it costs
compiled_at: 2026-10-22
videos: 13
chunks: 756
sources:
  - video_id: Ub3GoFaUcds
    title: "Stanford CME295 Transformers & LLMs | Autumn 2025 | Lecture 1 - Transformer"
  - video_id: VlA_jt_3Qc4
    title: "Stanford CME295 Transformers & LLMs | Autumn 2025 | Lecture 4 - LLM Training"
  - video_id: PmW_TMQ3l0I
    title: "Stanford CME295 Transformers & LLMs | Autumn 2025 | Lecture 5 - LLM tuning"
  - video_id: k5Fh-UgTuCo
    title: "Stanford CME295 Transformers & LLMs | Autumn 2025 | Lecture 6 - LLM Reasoning"
  - video_id: 8fNP4N46RRo
    title: "Stanford CME295 Transformers & LLMs | Autumn 2025 | Lecture 8 - LLM Evaluation"
  - video_id: idIck_TmPTE
    title: "LLM Fine-Tuning 24: Embedding & Embedding Fine-Tuning Full Guide | Train Your Own Embedding Model"
  - video_id: hBzUokVYQkI
    title: "AI Infrastructure Explained (GPUs, vLLM, and LLM-D)"
  - video_id: 4yNswvhPWCQ
    title: "Build an LLM from Scratch 7: Instruction Finetuning"
  - video_id: a3SMraZWNNs
    title: "How to Systematically Setup LLM Evals (Metrics, Unit Tests, LLM-as-a-Judge)"
  - video_id: cO6Ly7mIziQ
    title: "LoRA & QLoRA Explained Simply | Full Fine-Tuning vs PEFT + Intuition + Practical (Complete Guide)"
  - video_id: uLrOI65XbDw
    title: "Everything you need to know about Fine-tuning and Merging LLMs: Maxime Labonne"
  - video_id: -TiET_K-E_g
    title: "From 46% to 90%: Fine-Tuning Tiny LLMs for On-Device Agents — Cormac Brick, Google"
  - video_id: zjM2-_cFupo
    title: "Data + AI Summit Keynote, Wednesday Part 4 - Democratizing LLMs with MosaicML and Databricks"
---

# The sequence bottleneck, and the architecture that broke it

A field guide from the transcript corpus. What the transformer actually changed about
neural network training, how it differs from RNNs and LSTMs mechanism by mechanism, and
what that change costs you downstream — distilled from Stanford's CME295 lectures
(Afshine and Shervine Amidi), Maxime Labonne, Sebastian Raschka, Cormac Brick, Sunny
Savita, Dave Ebbelaar, KodeKloud and the Databricks/MosaicML keynote.

13 source videos · 756 transcript chunks read in full · 5 parallel extraction passes ·
compiled 2026-10-22.

## Thesis: attention traded memory for wiring

The short answer to the question. An RNN carries the meaning of a sentence forward in a
single hidden vector, one token at a time; the transformer deletes that vector and gives
every token a direct link to every other token at once. The 2017 paper's whole bet was
to move away from sequential processing [Ub3GoFaUcds@43] and instead let the model have
direct connections with all parts of the text at once [Ub3GoFaUcds@43]. Three
consequences fall out of that one change, and they are why the architecture took over:
the gradient no longer has to survive a long multiplicative chain through time, the
representation of a word becomes context-specific, and the computation becomes a pile of
matrix multiplications instead of a loop — GPUs love matrices [Ub3GoFaUcds@45].

That is the mechanism. The revolution in *training* is a second-order effect: once the
forward pass is parallel matrix math, you can spend compute instead of wall-clock. An RNN
had to compute all the hidden states before predicting a word [Ub3GoFaUcds@41], so long
sequences meant long training; the transformer runs the whole sequence in one pass and
runs its attention heads with all of that done in parallel [Ub3GoFaUcds@61]. Scale became
a budget line rather than a schedule: a modern pre-training run lands around 1e25 FLOPs
[VlA_jt_3Qc4@9] and tens of millions of dollars [VlA_jt_3Qc4@14]. The same bet that
bought parallelism is what bills you: self-attention is n² in sequence length
[VlA_jt_3Qc4@19], and most of the serving stack you will ever touch exists to pay that
bill.

The corpus is blunt about the one thing beginners over-read: the architecture is not
where the intelligence lives. The transformer is only a page of code [hBzUokVYQkI@5], and
GPT, Claude and Llama all run essentially that same formula — what differs is only the
learned weight values, which are numbers learned from examples [hBzUokVYQkI@4]. Read this
guide as architecture first, then as the training and serving consequences that
architecture forces on you.

> "Even from the title, you can see that the authors wanted to just rely on that part."
> — on the paper *Attention is All You Need* [Ub3GoFaUcds@43], 2017.

**Delete the hidden state.** In an RNN the sentence's meaning is solely encapsulated into
this hidden state [Ub3GoFaUcds@38]. Attention replaces that single channel with a direct
link per token pair, so nothing has to be remembered through a bottleneck.

**Make it a matrix.** Expressing the computation as query/key/value matrices lets the
whole sequence be attended in one matrix multiply — made for the hardware we have
[Ub3GoFaUcds@45], which does about 100× more math per second [hBzUokVYQkI@9] than a CPU.

**Pay the n² tax on purpose.** Activation memory now scales with how large your context
length is [VlA_jt_3Qc4@19]. Flash attention, KV caching and prefill/decode splitting are
all downstream of that single trade.

## Lineage: forty years to get to one page of code

The corpus dates the field back well before ChatGPT and is precise about why the early
ideas stalled: the models existed, the inputs did not. In the 1980s and 1990s there was
not a lot of compute [Ub3GoFaUcds@14] and no internet-scale corpus. Read the timeline as
a sequence of patches, each fixing the previous one's failure, until 2017 removed the
thing being patched.

### 1. 1980s — recurrence introduces word order

The first real answer to "a bag of embeddings loses the order of the words."

- An RNN processes tokens one at a time and keeps a hidden representation of the sentence
  so far [Ub3GoFaUcds@35], variously called the hidden state, activation or context
  vector.
- The win was real: with RNNs the word order matters [Ub3GoFaUcds@36], and the same
  machinery served classification (project the last hidden state), tagging, and
  generation.
- Blocker of the era: no internet and no compute, explicitly called out as one of the
  limiting factors [Ub3GoFaUcds@14] that kept today's models untrainable then.

### 2. 1990s — the LSTM bolts on a second memory

Patch one: if one hidden state forgets, add a channel whose job is not forgetting.

- Long Short-Term Memory [Ub3GoFaUcds@38] adds another tracked quantity called the cell
  state [Ub3GoFaUcds@38] on top of the hidden state.
- The lecture's verdict on the patch is flat: it aims at improving that piece, but it was
  not perfect either [Ub3GoFaUcds@39] — the forgetting problem stayed the main issue of
  RNN-based methods.

### 3. 2013 — Word2vec proves representations can be learned

Before you can attend over tokens you need vectors worth attending over.

- Word2vec learned embeddings from a proxy task and produced interpretable analogies —
  king is to queen what this is to that [Ub3GoFaUcds@25].
- Dimensions sit in the hundreds, far below vocabulary size: 768 is one commonly cited
  example [Ub3GoFaUcds@27], against a vocabulary of tens of thousands of tokens for one
  language [Ub3GoFaUcds@24].
- Its fatal limit, and the exact gap attention closes: Word2vec and FastText produce
  static embeddings [idIck_TmPTE@28] with no awareness of context.

### 4. 2014 — attention arrives as an RNN accessory

Patch two: stop relying on memory, add a shortcut to the past.

- Attention gives a direct link between what you are trying to predict
  [Ub3GoFaUcds@42] and something earlier in the text; it was introduced in 2014
  [Ub3GoFaUcds@42], three years before the transformer.
- It was invented specifically to solve for these long-range dependency issues
  [Ub3GoFaUcds@42] — the RNN's defect, not a new capability.

### 5. 2017 — the paper removes the thing being patched

If the shortcut is what works, keep the shortcut and throw away the loop.

- The transformer, introduced in a paper named Attention is All You Need
  [Ub3GoFaUcds@43], is at the foundation of all of the models you see today
  [Ub3GoFaUcds@15].
- To build the representation of a token the model will look at all the other tokens in
  the sequence at once [Ub3GoFaUcds@44] — this is self-attention.
- The attention mechanism itself is not exotic: it is a weight-oriented mechanism
  [cO6Ly7mIziQ@26], deciding which word matters to which other word.

### 6. 2020s — the same block, scaled until it became an LLM

No architecture change was needed to get from translation to ChatGPT; only data and
compute.

- The models were scaled up both by compute and in terms of the data [Ub3GoFaUcds@15] —
  and that is how LLMs were dubbed.
- The scale delta is visible in two numbers: GPT-3 trained on 300 billion tokens
  [VlA_jt_3Qc4@9], Llama 3 on 15 trillion.
- Over 90% of modern LLMs built this way are decoder-only models [VlA_jt_3Qc4@8] — the
  2017 encoder-decoder got cut in half along the way.
- Adoption numbers from the same era: in one 30-day window, more than 1,500 customers
  used a Transformer model on Databricks [zjM2-_cFupo@13], with GPU usage growing 25%
  month over month [zjM2-_cFupo@13].

## The recurrence tax: four bills an RNN hands you

If you only remember one framing for the RNN-versus-transformer question, make it this:
recurrence is not wrong, it is *expensive in four specific ways*, and attention pays off
all four at once.

**1. The gradient dies on the way back.** Predicting a late word depends on every hidden
state before it, so updating weights means back-propagating through time. Written out,
that is a product of many quantities: if it's less than 1, it's vanishing
[Ub3GoFaUcds@40] — multiply enough of those and the update goes to zero. The literature
name for it is the vanishing gradient [Ub3GoFaUcds@39].

**2. One vector has to hold the whole sentence.** Because the meaning is solely
encapsulated into this hidden state [Ub3GoFaUcds@38], long inputs degrade the model's
ability to "remember" what it saw. This is what people mean by long-range dependencies;
the LSTM's cell state was an attempt, not a fix.

**3. Training is serial, so GPUs idle.** Stated plainly in the lecture: the computations
are very slow [Ub3GoFaUcds@41], because at training time you need to compute all these
hidden states before [Ub3GoFaUcds@41] the token you want. Longer sequence, linearly
longer wall-clock, with no way to spend parallel hardware on it.

**4. Word representations are position-blind.** The pre-attention era's representations
were token-specific regardless of where they're at [Ub3GoFaUcds@35]. Self-attention fixes
exactly this: in "river bank" versus "robbing a bank", the bank would have different
representations [Ub3GoFaUcds@44] — and the same contrast is why sentence-transformer
embeddings beat Word2vec, since bank will be represented according to the context
[idIck_TmPTE@32].

## The swap, mechanism by mechanism

| Recurrent mechanism | What the transformer put there instead |
| --- | --- |
| Hidden state passed forward token by token | All-to-all self-attention: look at all the other tokens at once [Ub3GoFaUcds@44], with no memory bottleneck to survive. |
| Order comes free from the loop | Order must be injected. Because the links are direct, unlike the RNN you lose the sense of order [Ub3GoFaUcds@51], so you add position encodings to inform on the position [Ub3GoFaUcds@51]. |
| Sequential forward and backward passes | One matrix-multiply pass over the sequence; heads run with all of that done in parallel [Ub3GoFaUcds@61], then concatenated and projected back. |
| Memory cost roughly linear in sequence length | Memory cost quadratic: n² complexity because of this self attention operation [VlA_jt_3Qc4@19]. You traded time for space, deliberately. |
| Static, context-free token embeddings feeding the loop | Context-specific representations computed from the whole sequence, so each token's representation is unique to its context [Ub3GoFaUcds@44]. |
| Hand-built task architectures per NLP task | One stack, pre-trained once then adapted: a pre-training stage [VlA_jt_3Qc4@6] followed by cheap tuning, instead of training from scratch per task. |

## Read the block in one pass

Six steps, in the order the data moves. This is the original encoder-decoder shape — an
encoder on the left side and a decoder [Ub3GoFaUcds@47] — built around translation.
Decoder-only LLMs keep steps 1–3, 5 and 6, and keep only the masked flavour of step 4.

### 1. Tokenize

- Tokenization is cutting the text with respect to some arbitrary unit of text
  [Ub3GoFaUcds@16]; that unit is the token.
- Subword is the default middle ground because it works by leveraging roots of words
  [Ub3GoFaUcds@17], so related words share structure — at the cost of longer sequences.
- Word-level carries the worst out-of-vocabulary risk — this thing is called OOV
  [Ub3GoFaUcds@19] — while character-level eliminates it and inflates sequence length.
- Rough serving conversion: a token is roughly 3/4 of a word [hBzUokVYQkI@16].

### 2. Embed

- An embedding is a dense high-dimensional numerical vector representing semantic meaning
  [idIck_TmPTE@9]; common sizes run 384, 768, 1024, 1536, 2048.
- Target vocabulary size is tens of thousands for one language, hundreds of thousands
  [Ub3GoFaUcds@24] for multilingual or code-covering models.

### 3. Add position encodings

- Direct token-to-token links destroy the implicit ordering a loop gave you, so you add
  position encodings to inform on the position of the word in the sequence
  [Ub3GoFaUcds@51].
- The lecture flags it and skips the formula — we're not going to dig into that today
  [Ub3GoFaUcds@52] — so treat the mechanism as understood and the math as a gap.

### 4. Multi-head attention (three flavours)

- Encoder self-attention computes embeddings from the input sentence as a function of
  themselves [Ub3GoFaUcds@51].
- Decoder self-attention is masked — it only looks at the tokens [Ub3GoFaUcds@50] already
  decoded, never to the right of the one being predicted.
- Cross-attention expresses the output as a function of what has been seen in the input
  [Ub3GoFaUcds@51].

### 5. Feed-forward expansion

- Unlike Word2vec's bottleneck, here the hidden layer is of a bigger dimension than input
  and output [Ub3GoFaUcds@63], to buy degrees of freedom for richer features.
- This block matters more than its reputation suggests for adaptation: analysis after the
  LoRA paper found that the bulk of the performance improvements is actually contained in
  the feed-forward [VlA_jt_3Qc4@70] block, not the attention matrices the paper
  originally targeted.

### 6. Stack N, then softmax over the vocabulary

- You get a stacked succession of N encoders [Ub3GoFaUcds@63] and N decoders; the final
  encoder representation is what is fed to the N decoders' [Ub3GoFaUcds@63]
  cross-attention.
- At the end you project to a vector of vocabulary size and softmax it to guess what the
  next word is [Ub3GoFaUcds@53].
- Stacking is why weight counts balloon: with blocks repeated, attention plus
  feed-forward weights land in the billions — it's going to be in billions
  [cO6Ly7mIziQ@14] — with individual matrices at 512×512 and bigger [cO6Ly7mIziQ@34].
- One training trick from the original paper is worth copying: instead of a hard one-hot
  target, label smoothing says predict this word, but there's a chance it's not this word
  [Ub3GoFaUcds@55], because there is rarely only one valid continuation.

## The attention computation, line by line

Four operations. If you can write these from memory you can read any transformer diagram.
Nothing here is more than a linear projection, a dot product, a softmax and a weighted sum
— it is a matrix game [cO6Ly7mIziQ@30] end to end.

### 1. Project each token three ways

- Inside multi-head attention the input is multiplied with the three main matrices
  [cO6Ly7mIziQ@28], Q, K and V, as a linear transformation.
- The projection matrices are what the word "head" names: the word head is given to the
  projection matrices [Ub3GoFaUcds@54] used to obtain query, key and value.

### 2. Score query against every key, then scale

- You compare the query to the key to quantify how similar [Ub3GoFaUcds@45] they are, for
  every pair in the sequence at once.
- Dot products grow with dimension, so you normalize them [Ub3GoFaUcds@60] — which is why
  you divide by the square root of the dimension of keys [Ub3GoFaUcds@60].
- A worked intuition: in "I love machine learning" the attention weight ties *learning*
  to *machine* — it belongs to the machine only, not love [cO6Ly7mIziQ@29].

### 3. Softmax, then take a weighted sum of values

- Softmax over the scaled scores gives a probability distribution of the projection of
  the query [Ub3GoFaUcds@60] over keys, one row per query.
- Multiply by V and you end up with a weighted sum of values for each query
  [Ub3GoFaUcds@61]: weight the most similar keys and take the corresponding value
  [Ub3GoFaUcds@45].
- Self-attention's job, from the practitioner side: it is trying to check the relation
  [cO6Ly7mIziQ@27] between every pair of words in the same sentence.

### 4. Run h heads in parallel, concatenate, project back

- Multiple heads mean multiple learned projections, allowing your model to learn different
  projections [Ub3GoFaUcds@54] and therefore different token-to-token associations. If
  you have a vision background, it is like having multiple filters in a convolution.
- The h head outputs are computed with all of that done in parallel [Ub3GoFaUcds@61],
  concatenated columnwise, then pushed through an output projection Wo back to the
  embedding dimension.
- Nothing forces the heads to differ; they differ because of the magic of gradient descent
  [Ub3GoFaUcds@62] — the network has no incentive to copy itself.
- Why this shape and not another: matrix form means the whole-sequence computation is
  matrix multiplication, and GPUs love matrices [Ub3GoFaUcds@45] — a GPU's thousands of
  small cores do about 100 times more math every second [hBzUokVYQkI@9] than a CPU's few
  big ones.

## The n² bill, and the six things built to pay it

Everything in this section is a consequence of deleting the hidden state. An RNN's memory
cost grew with depth in time; a transformer's grows with the *square* of context.
Training-side and serving-side mitigations are different engineering problems with the
same root cause, so learn both.

**Where training memory actually goes.** A training step must hold activations, gradients
and optimizer moments at once, and activation memory depends on model size, batch size and
context length — because it's dependent on how large your context length is
[VlA_jt_3Qc4@19], at n² complexity for the self attention operation [VlA_jt_3Qc4@19]. The
transformer-era default optimizer is AdamW, which combines the weight decay concept
[cO6Ly7mIziQ@57] with a warm-up schedule.

**Flash attention: tile it into fast memory.** Flash attention, developed at Stanford
[VlA_jt_3Qc4@26] in 2022, exploits the gap between large-slow HBM and small-fast SRAM. The
method is called tiling [VlA_jt_3Qc4@29]: compute softmax exactly on small blocks end to
end in SRAM instead of writing giant intermediates back to HBM. Reported effect is almost
a 10× reduction [VlA_jt_3Qc4@34] in HBM reads and writes, and it saves memory and runtime
at the same time.

**Spread it: data, model, ZeRO.** Data parallelism distributes data across GPUs
[VlA_jt_3Qc4@21] with a full model copy each, paying gradient-averaging traffic. Zero
redundancy optimization [VlA_jt_3Qc4@23] shards optimizer states, gradients and parameters
instead of duplicating them. Model parallelism adds expert parallelism [VlA_jt_3Qc4@24],
tensor parallelism and pipeline parallelism.

**Lower the precision.** Quantization halves footprint and roughly doubles throughput:
dropping FP32 to FP16 means it takes basically half the memory [VlA_jt_3Qc4@36]. The byte
arithmetic is worth memorising — FP32 is around 4 bytes [cO6Ly7mIziQ@64] per weight, FP16
about 2, and 4-bit around 0.5 bytes [cO6Ly7mIziQ@64]. Mixed-precision training keeps
master weights in FP32 while running the passes in FP16.

**At inference: cache the prefill.** Generation is one trick repeated — it predicts the
next token [hBzUokVYQkI@16], appends it and goes again — so a 100-token answer is 100
passes. The prompt is read in one compute-heavy burst whose latency is TTFT, time to first
token [hBzUokVYQkI@19]; that work is saved so each later token reaches into the cache and
reuses everything it already worked out [hBzUokVYQkI@23]. Keep those blocks past the end of
a request and you get prefix caching [hBzUokVYQkI@24], where a cached input token costs
about a tenth of a fresh one [hBzUokVYQkI@25].

**Decode is bandwidth-bound, so batch and shard.** Each decode step rereads the whole model
from VRAM, so a 3 TB/s GPU on a 16 GB model tops out: 200 tokens a second is the maximum
limit [hBzUokVYQkI@21]. The fix is batching, since one read gives one token for one user or
50 tokens for 50 users [hBzUokVYQkI@26]. Concurrency then caps on memory, not math — it's
the memory that decides how many people one GPU [hBzUokVYQkI@28] can serve. When weights
exceed one card (a 70B model is 140 GB [hBzUokVYQkI@5] against an 80 GB H100
[hBzUokVYQkI@13]), you resort to splitting a model across GPUs [hBzUokVYQkI@31] and keep
the chatty talk inside a box [hBzUokVYQkI@31].

Two routing pitfalls follow directly from the KV cache being physical state on a specific
machine. A plain round-robin balancer throws away saved work that was perfectly good
[hBzUokVYQkI@33]; cache-aware routing instead delivers around three times the throughput
[hBzUokVYQkI@35] on identical hardware, and splitting compute-heavy prefill from
bandwidth-heavy decode pools yields up to 70% more tokens per second on the same hardware
[hBzUokVYQkI@36]. If you are self-hosting, note that vLLM speaks the exact same format the
OpenAI API uses [hBzUokVYQkI@15], so the swap is a base-URL change.

## How the architecture rewrote the training pipeline

This is the part of "revolutionised neural network training" that is not about attention
math. Because one expensive pre-training run produces a general model, training became a
staged transfer-learning pipeline rather than a per-task project — a pre-training stage
[VlA_jt_3Qc4@6] followed by cheaper tuning. In the corpus's terminology, fine-tuning plus
preference tuning is what we call alignment [VlA_jt_3Qc4@66], with a newer mid-training
stage emerging between them.

### 1. Pre-train: next-token prediction at internet scale

- The first stage is pre-training [cO6Ly7mIziQ@7], where the model learns to do next-token
  prediction [uLrOI65XbDw@0].
- Chinchilla's compute-optimal rule: use a training set about 20 times the model size
  [VlA_jt_3Qc4@13] in tokens — by which GPT-3 (175B parameters, 300B tokens) was
  undertrained.
- Cost floor is millions and it can even cost tens of millions of dollars
  [VlA_jt_3Qc4@14], plus a tracked ecological cost.
- The output is bounded in time: a base model's knowledge stops at what is called the
  knowledge cutoff date [VlA_jt_3Qc4@15], and injecting new facts afterward without
  damaging other capabilities is unsolved.
- The resulting model is not an assistant: it has been trained on next token prediction,
  not being an assistant [VlA_jt_3Qc4@43], so it will autocomplete your question instead
  of answering it [uLrOI65XbDw@0].
- A counterweight on cost from the enterprise side: MosaicML's MPT-7B was trained from
  scratch where it only costs 250,000 to build it [zjM2-_cFupo@9], versus the 10 to 15
  million dollars [zjM2-_cFupo@8] cited for GPT-3-scale, and MPT-30B at seven or eight
  hundred thousand dollars [zjM2-_cFupo@10].

### 2. Supervised fine-tune: same loss, different text

- SFT is short for supervised fine-tuning [VlA_jt_3Qc4@44]: keep training the pre-trained
  model on input/output pairs so it predicts the right tokens for a given input.
- It reuses the pre-training objective exactly — instruction fine-tuning uses exactly the
  same loss that we used during pre-training [4yNswvhPWCQ@44].
- The loss is masked to the response: we don't put a loss over the input [VlA_jt_3Qc4@45],
  and in practice you train on the outputs only [uLrOI65XbDw@3].
- Mechanically, replace target padding with -100 — replacing all the target padding tokens
  by the minus 100 [4yNswvhPWCQ@27], PyTorch's ignore index — and pad only to the longest
  element in each batch [4yNswvhPWCQ@14], not the dataset.
- Dataset scale is orders of magnitude below pre-training: about 13k for GPT-3 and 10
  million for Llama 3 [VlA_jt_3Qc4@52]; a practical floor for real instruction tuning is
  at least 100,000 training examples [4yNswvhPWCQ@5].
- Template length is a real cost line, not a style choice: shorter prompt templates mean
  fewer tokens per example, and shorter is cheaper [4yNswvhPWCQ@9].
- Hardware sets the iteration loop: the same 355M-parameter 2-epoch run that takes 15–29
  minutes on a laptop finishes in about a minute on an A100 GPU [4yNswvhPWCQ@47].

### 3. Preference-tune: teach tone, not facts

- The unit of data is a rejected output paired with a preferred rewrite for the same
  prompt — a preference pair [PmW_TMQ3l0I@3] — collected pairwise [PmW_TMQ3l0I@8] because
  pointwise scoring is hard to calibrate.
- RLHF [PmW_TMQ3l0I@16] is two stages: fit a reward model using the Bradley-Terry
  formulation [PmW_TMQ3l0I@19], then optimise the policy with RL.
- The RL objective must optimise for higher rewards without going too far
  [PmW_TMQ3l0I@30] from the SFT reference, via a KL penalty — otherwise you get reward
  hacking against an imperfect proxy.
- PPO [PmW_TMQ3l0I@33] keeps updates close to the previous policy with a clipped
  objective, but you need a lot of models [PmW_TMQ3l0I@45] resident: policy, reference,
  reward model, value function.
- DPO [PmW_TMQ3l0I@57] collapses that into one loss on preference pairs, so instead of
  four models you just need two [PmW_TMQ3l0I@62] — it is simpler than RLHF because it does
  not use a separate reward model [4yNswvhPWCQ@53].
- The honest trade: overall PPO performs better than DPO [PmW_TMQ3l0I@63], but the RL bar
  is very hard to tune right [PmW_TMQ3l0I@67]. Pick by whether you have RL expertise to
  spend.
- If you want the reward signal without the training loop, best-of-N [PmW_TMQ3l0I@51]
  scores N samples from a frozen model — but you still wait more than a single pass
  [PmW_TMQ3l0I@57], so it trades cost for latency rather than removing either.

### 4. Post-2024: RL on verifiable rewards, bolted onto the same block

- Vanilla next-token models have quote-unquote limited reasoning [k5Fh-UgTuCo@6] and are
  all talk, no action [k5Fh-UgTuCo@7] — they can only emit text.
- Chain-of-thought works because a hard problem is unlikely to appear verbatim in training
  data, so you make the model decompose the problem into tractable ones [k5Fh-UgTuCo@12];
  letting it emit more tokens first means you just give it more compute [k5Fh-UgTuCo@13].
- The reward needs no reward model at all for math and code: combine a format check —
  checking whether the think tokens are here [k5Fh-UgTuCo@33] — with a correctness check
  against tests or ground truth.
- GRPO [k5Fh-UgTuCo@37] drops PPO's value function entirely, computing advantage as reward
  minus the average over the standard deviation [k5Fh-UgTuCo@42] across a sampled group of
  completions.
- Known bug worth knowing: GRPO's per-token length normalisation pushes outputs longer and
  longer; Dr. GRPO removes the factor altogether [k5Fh-UgTuCo@54].
- Pure RL on a base model works but degrades the chain: R1-Zero would mix languages and
  had syntax issues [k5Fh-UgTuCo@59], which is why R1 added cold-start SFT data and a
  language-consistency reward.
- For small models, distilling the knowledge is more efficient [k5Fh-UgTuCo@68] than
  running RL on them from scratch.

One structural warning that applies to every number in this section. Benchmark comparisons
across models are only meaningful if both models had comparable exposure to the
benchmark's domain — the documented phenomenon is training on the test task, not the test
set, the test task [VlA_jt_3Qc4@59]. Crowd leaderboards are not a refuge either: it's
easily possible to rig such a leaderboard [VlA_jt_3Qc4@62], and when a measure becomes a
target it ceases to be a good measure [8fNP4N46RRo@71]. MMLU [VlA_jt_3Qc4@58] in particular
mostly measures how well pre-training was done [8fNP4N46RRo@54], not reasoning.

## Adapt, do not retrain: the architecture's other gift

LoRA is usually taught as a memory trick. It is better understood as a direct exploitation
of the transformer's shape: the model is a stack of large weight matrices, and the *update*
to one of those matrices turns out to be low-rank. Full fine-tuning updates everything — a
13 billion model means 13 billion weights [cO6Ly7mIziQ@13] — so we cannot perform
full-parameter fine-tuning [cO6Ly7mIziQ@15] on a single GPU.

**The low-rank bet.** LoRA's assumption, stated plainly: it is possible that the weight
update matrix is not full rank [cO6Ly7mIziQ@52]. A rank-2 4×5 matrix (20 numbers) factors
into 4×2 and 2×5 (18 numbers); at scale, billions of parameters become millions
[cO6Ly7mIziQ@49]. The rank r is typically taken very small [VlA_jt_3Qc4@67] against W0's
hundreds or thousands of dimensions.

**Where the adapters attach.** You freeze all the pre-trained weights and add adapters
[uLrOI65XbDw@7], then append the adapter [cO6Ly7mIziQ@60] onto the multi-head attention
matrices and the feed-forward layer of each block. Target choice is itself a
hyperparameter: Q alone is average, Q plus V is better, and the feed-forward block is
where the bulk of the performance improvements is actually contained [VlA_jt_3Qc4@70].

**The numbers that justify it.** On GPT-3, LoRA trains only 4.7 million parameters
[cO6Ly7mIziQ@62] against 175 billion, and the cited comparison has full fine-tuning at
89.5% accuracy [cO6Ly7mIziQ@63] with LoRA matching it. In the worked walkthrough, rank 16
and alpha 32 [cO6Ly7mIziQ@75] leave 0.277% of parameters trainable [cO6Ly7mIziQ@76].

**QLoRA: quantize the frozen half.** QLoRA quantizes the pre-trained model in 4-bit
precision [uLrOI65XbDw@8] — NF4 means Normal Float 4 [cO6Ly7mIziQ@74] — while the adapters
train at higher precision, bringing a task down to 8 to 16 GB of VRAM [cO6Ly7mIziQ@64]. It
is a real trade, not a free lunch: lower VRAM, degraded performance. Stack-wise you add
bitsandbytes for the quantization [cO6Ly7mIziQ@64] on top of peft, which implements the
LoRA [cO6Ly7mIziQ@67].

**Hyperparameters that behave differently.** LoRA is not full fine-tuning with fewer
weights; its dynamics differ. Empirically you want higher learning rates
[VlA_jt_3Qc4@70] — 10 times more is the guidance [VlA_jt_3Qc4@70] — and it does not
perform as well with larger batch size [VlA_jt_3Qc4@70]. For full fine-tuning the search
procedure is blunt: go as high as you can until your loss explodes [uLrOI65XbDw@8], then
back off.

**Merging: capability without gradients.** Because every fine-tune of a given base shares a
weight geometry, you can average them. Merging does not require any GPU [uLrOI65XbDw@9],
and on the Open LLM Leaderboard's 7B tier the entire top eight or ten was merge models
[uLrOI65XbDw@9]. SLERP — spherical linear interpolation [uLrOI65XbDw@10] — merges two; DARE
prunes and rescales and works really well with seven or eight models [uLrOI65XbDw@11]; or
extract feed-forward layers from several fine-tunes and add a router [uLrOI65XbDw@13] for a
FrankenMoE.

At the small end, the same architecture runs entirely on a phone. Models smaller than a
billion parameters [-TiET_K-E_g@0] hit almost 2,000 tokens per second prefill and 140
decode [-TiET_K-E_g@12] on an old Pixel 7, and fine-tuning moved one 270M function-calling
model from 46% to over 90% [-TiET_K-E_g@14] app-intent accuracy. The constraint is
narrowness: at that size it needs to have a very narrow and focused task [-TiET_K-E_g@11].

## Three things to copy

### The whiteboard answer: RNN/LSTM vs transformer

```
RNN / LSTM
- Carries meaning forward in ONE hidden state, token by token.
- Order is implicit in the loop. No position encoding needed.
- Failure 1: vanishing gradient. Backprop through time multiplies many
  terms; terms < 1 drive the update to zero. Long-range memory dies.
- Failure 2: the whole sentence lives in one vector (LSTM's cell state
  is a patch, not a fix).
- Failure 3: training is serial. To predict token t you must compute
  every hidden state before t. Long sequence = long wall clock.
- Failure 4: representations are token-specific regardless of position,
  so "bank" has one vector in every context.
- Cost shape: roughly linear in sequence length.

TRANSFORMER (2017, "Attention is All You Need")
- Deletes the hidden state. Every token gets a DIRECT link to every
  other token, computed at once (self-attention).
- Fixes 1 and 2: no long multiplicative chain through time to survive.
- Fixes 3: the sequence runs in one pass, and the h attention heads run
  in parallel. Matrix multiplies, which is what GPUs are built for
  (~100x more math/sec than a CPU).
- Fixes 4: a token's representation is computed from the whole
  sequence, so it is context-specific.
- New obligations: add position encodings (order is no longer implicit);
  pay O(n^2) in sequence length for attention.
- Depth is repetition: N identical blocks. Attention + FFN weights land
  in the billions, which is why PEFT exists.

WHAT IT CHANGED ABOUT TRAINING
- Parallel forward pass => scale becomes a budget, not a schedule.
  ~1e25 FLOPs, tens of millions of dollars for a frontier run.
- One pre-train, then cheap staged tuning (SFT -> preference tuning)
  replaced per-task architectures.
- >90% of modern LLMs are decoder-only: the 2017 encoder got dropped.
- Chinchilla: ~20 tokens per parameter is compute-optimal.

WHAT IT COST
- n^2 activation memory => flash attention (tiling, ~10x fewer HBM
  reads/writes), ZeRO, mixed precision, sharding.
- Autoregressive decode => KV cache, prefix caching, batching, and
  VRAM (not compute) capping concurrent users.
```

Every line traces to a cite in the Lineage, Recurrence tax, Swap, Attention math, n² bill
and Training sections above.

### Triage: which transformer cost is actually hurting you

```
TRAINING SIDE
[ ] OOM at long context, not large batch?
    -> It is the n^2 activation term. Enable flash attention (tiling)
       before buying GPUs. Reported ~10x fewer HBM reads/writes.
[ ] OOM with every GPU holding identical optimizer state?
    -> ZeRO stages 1-3. Shard optimizer state, gradients, parameters.
[ ] Memory fine, throughput poor?
    -> Mixed precision (FP32 master weights, FP16 passes). FP32 ~4 B/
       weight, FP16 ~2 B, 4-bit ~0.5 B.
[ ] Cannot fit full fine-tuning on one GPU at all?
    -> LoRA first (10x higher LR, smaller batches), QLoRA if still
       short (4-bit NF4 base, 8-16 GB VRAM, accepts a quality hit).
[ ] Loss curve flat and you have not searched LR?
    -> Push LR up until the loss explodes, then back off.

SERVING SIDE
[ ] Slow first token on long prompts?
    -> That is prefill/TTFT, compute-bound. Prefix-cache shared system
       prompts: a cached input token costs ~1/10 of a fresh one.
[ ] Slow per-token streaming on short prompts?
    -> That is decode/TPOT, memory-bandwidth-bound. One full weight
       read per token caps you (e.g. ~200 tok/s for 16 GB at 3 TB/s).
[ ] Low concurrency but idle GPU cores?
    -> It is VRAM for per-user KV cache, not compute. Batch harder
       before scaling out; one read serves 50 users as cheaply as 1.
[ ] Latency spikes on multi-turn chat behind a load balancer?
    -> Round-robin throws away valid cached work. Cache-aware routing
       reported ~3x throughput on the same hardware.
[ ] Mixed workload (huge summaries next to short chats)?
    -> Disaggregate prefill and decode pools: up to 70% more tok/s.
[ ] Weights exceed one card (70B ~= 140 GB vs 80 GB H100)?
    -> Shard. Tensor-parallel inside a box, pipeline-parallel across.
```

Order matters: every item above is cheaper than adding hardware, and the corpus notes
teams overbuy GPUs that then sit half-idle because memory, not math, was the limit.

### Starting LoRA / QLoRA config

```python
# Stack: transformers + peft + bitsandbytes + accelerate + trl + datasets
#   peft        -> implements LoRA
#   bitsandbytes-> 4-bit quantization (QLoRA)
#   trl         -> SFT and preference tuning (DPO / ORPO / GRPO)

# --- QLoRA base load (skip quant_config for plain LoRA) ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # Normal Float 4
    bnb_4bit_compute_dtype="float16", # compute stays higher precision
)

# --- Adapter ---
lora_config = LoraConfig(
    r=16,                 # walkthrough default; rank is "typically very small"
    lora_alpha=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[      # attention Q/V *and* the FFN blocks
        "q_proj", "v_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
# Expect ~0.3% of parameters trainable. Print it and check.

# --- Training notes, not defaults to copy blindly ---
# learning_rate: start ~10x your full-fine-tune LR, then search upward
#                until the loss explodes and back off.
# batch size   : LoRA performs WORSE with large batches. Do not max it
#                out the way you would for full fine-tuning.
# loss masking : train on the response only; set prompt/padding target
#                ids to -100 so they are ignored.
# padding      : pad to the longest example in each batch, not the
#                longest in the dataset.
# after training: merge the adapter into the base model and re-save as
#                a standalone checkpoint, or keep one adapter per task
#                and swap adapters instead of whole models.
```

Rank 16 / alpha 32 and the Q+V target choice are the corpus's worked example, not a tuned
recommendation — the corpus gives no procedure for choosing rank or alpha on a new task.

## Sources

Thirteen talks, read in full across five extraction passes. The architecture spine comes
from the Stanford lectures; everything about cost, serving and adaptation comes from
practitioners who had to pay for it.

### Architecture and the training pipeline

1. [Stanford CME295 Transformers & LLMs | Autumn 2025 | Lecture 1 - Transformer](https://www.youtube.com/watch?v=Ub3GoFaUcds) `Ub3GoFaUcds` — the spine of this guide: tokenization tradeoffs, Word2vec, why RNNs and LSTMs fail, and the full Q/K/V mechanics with a worked translation example.
2. [Stanford CME295 Transformers & LLMs | Autumn 2025 | Lecture 4 - LLM Training](https://www.youtube.com/watch?v=VlA_jt_3Qc4) `VlA_jt_3Qc4` — pre-training scale and cost, Chinchilla, the distributed-training toolkit (ZeRO, flash attention, mixed precision), SFT and LoRA/QLoRA.
3. [Stanford CME295 Transformers & LLMs | Autumn 2025 | Lecture 5 - LLM tuning](https://www.youtube.com/watch?v=PmW_TMQ3l0I) `PmW_TMQ3l0I` — preference tuning end to end: Bradley-Terry reward models, PPO mechanics and footprint, best-of-N, DPO and the PPO-vs-DPO trade.

### Reasoning, evaluation, embeddings

4. [Stanford CME295 Transformers & LLMs | Autumn 2025 | Lecture 6 - LLM Reasoning](https://www.youtube.com/watch?v=k5Fh-UgTuCo) `k5Fh-UgTuCo` — how reasoning gets trained onto a vanilla transformer: chain-of-thought, pass@k, GRPO and its length-bias bug, the DeepSeek R1 recipe.
5. [Stanford CME295 Transformers & LLMs | Autumn 2025 | Lecture 8 - LLM Evaluation](https://www.youtube.com/watch?v=8fNP4N46RRo) `8fNP4N46RRo` — the evaluation taxonomy that proves whether architectural gains are real: rater agreement, rule-based metric limits, judge biases, the benchmark landscape.
6. [LLM Fine-Tuning 24: Embedding & Embedding Fine-Tuning Full Guide | Train Your Own Embedding Model](https://www.youtube.com/watch?v=idIck_TmPTE) `idIck_TmPTE` — the cleanest static-vs-contextual contrast in the corpus: why Word2vec gives one vector per word and a sentence transformer does not.

### Serving the formula, and judging what it emits

7. [AI Infrastructure Explained (GPUs, vLLM, and LLM-D)](https://www.youtube.com/watch?v=hBzUokVYQkI) `hBzUokVYQkI` — frames the transformer as a fixed formula plus learned weights, then walks the serving machinery: VRAM, KV cache, prefill/decode, batching, sharding, cache-aware routing.
8. [Build an LLM from Scratch 7: Instruction Finetuning](https://www.youtube.com/watch?v=4yNswvhPWCQ) `4yNswvhPWCQ` — the hands-on proof that SFT reuses the pre-training loss: prompt templating, per-batch padding, -100 loss masking, and the mess of judging free-form output.
9. [How to Systematically Setup LLM Evals (Metrics, Unit Tests, LLM-as-a-Judge)](https://www.youtube.com/watch?v=a3SMraZWNNs) `a3SMraZWNNs` — the three-level eval ladder (unit tests, human/model judge, A/B) and the human-alignment process a judge needs before you can trust it.

### Adapting, merging, and owning a model

10. [LoRA & QLoRA Explained Simply | Full Fine-Tuning vs PEFT + Intuition + Practical (Complete Guide)](https://www.youtube.com/watch?v=cO6Ly7mIziQ) `cO6Ly7mIziQ` — grounds attention and the FFN in concrete matrix-rank math, then shows exactly how LoRA exploits that structure, with the worked rank-16/alpha-32 config.
11. [Everything you need to know about Fine-tuning and Merging LLMs: Maxime Labonne](https://www.youtube.com/watch?v=uLrOI65XbDw) `uLrOI65XbDw` — the SFT taxonomy plus the merging deep dive (SLERP, DARE, passthrough self-merge, FrankenMoE) and the leaderboard evidence that merges beat trained models.
12. [From 46% to 90%: Fine-Tuning Tiny LLMs for On-Device Agents — Cormac Brick, Google](https://www.youtube.com/watch?v=-TiET_K-E_g) `-TiET_K-E_g` — the sub-1B end of the architecture: real on-device throughput numbers, the 46%-to-90% fine-tuning delta, and why tiny models need one narrow task.
13. [Data + AI Summit Keynote, Wednesday Part 4 - Democratizing LLMs with MosaicML and Databricks](https://www.youtube.com/watch?v=zjM2-_cFupo) `zjM2-_cFupo` — the enterprise price list for training transformers from scratch (MPT-7B at $250K, MPT-30B at ~$750K against GPT-3's $10-15M) and the adoption curve behind it.

## Gaps: what these thirteen talks do not answer

Read this list before you treat the guide as complete. Each item is something a reader of
this topic would reasonably ask and the corpus does not support — stated here rather than
filled in from memory.

- **No position-encoding math.** Lecture 1 names position encodings and explicitly
  declines the formula — so you get why they exist, not how sinusoidal (or any other)
  encoding is computed.
- **No layer norm, no residual connections.** Two load-bearing components of every real
  transformer block are absent from the whole corpus. The LoRA walkthrough stays at the
  Q/K/V and feed-forward level by its own admission.
- **No head-splitting math.** "h heads with different projection matrices, concatenated
  and projected back" is as deep as it goes; how d_model divides across heads is never
  shown.
- **No measured transformer-vs-LSTM head-to-head.** The displacement is argued from
  mechanism — vanishing gradients, slow sequential compute, context-blind representations
  — and from the 2017 paper's translation results. No corpus talk reports a score delta
  on the same task.
- **No modern architectural variants.** RoPE, grouped-query attention, sliding-window
  attention and MoE routing internals are never covered. Multi-head latent attention and
  MoE are named once in passing in the DeepSeek-V3 context and not explained.
- **No counter-argument to attention.** Nothing in the corpus asks whether recurrence is
  returning — linear attention and state-space models (Mamba and relatives) do not appear
  at all.
- **No link from architecture choices to KV-cache sizing.** Context length, head count and
  attention variant drive KV-cache memory directly, but the serving talk treats the cache
  purely as an ops concern and never gives a sizing formula.
- **No cost figures for the alignment stages.** Pre-training is quantified in dollars and
  FLOPs; RLHF/PPO and GRPO runs are never given wall-clock or dollar numbers, and the
  DeepSeek R1 cold-start dataset size is explicitly unknown.
- **No procedure for choosing LoRA rank or alpha.** Rank 16 and alpha 32 appear as one
  demo's values. Nothing in the corpus tells you how to pick them for a new task, and
  QLoRA gets no perplexity or accuracy numbers of its own.
- **Quantization internals are named, not taught.** NF4, double quantization, zero-point
  and absmax are all mentioned as pointers; none is derived.
- **No scaling-law interrogation.** Chinchilla's 20-tokens-per-parameter result and its
  claim that architecture choice barely matters are reported from the paper, never
  independently examined or challenged.
- **No safety or interpretability depth.** Red-teaming, guardrails and attention-level
  interpretability are absent; the corpus treats harmlessness as one reward term and
  safety benchmarks as policy-relative.

---

From Recurrence to Attention — a field guide compiled from a 13-video transcript corpus:
756 chunks read in full, 5 parallel extraction passes, compiled 2026-10-22. Every claim
above carries a cite resolving to a video id and chunk index you can open and check. Where
the corpus is silent, see Gaps.
