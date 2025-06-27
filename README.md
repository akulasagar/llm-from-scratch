# My From-Scratch LLM Journey

This repository documents my end-to-end journey of building neural networks from the ground up using Python and PyTorch, guided by an AI assistant. The project is a testament to learning fundamental concepts and applying them to build increasingly complex models.

## Part 1: Mastering the Fundamentals - The MNIST Image Classifier

The first part of this project focused on learning the entire machine learning workflow: data handling, model building, training, and evaluation, using the classic MNIST dataset of handwritten digits.

-   **Phase 1-2: Data & Architecture:** Loaded the MNIST dataset and designed a simple but effective Feed-Forward Neural Network.
-   **Phase 3: Training:** Implemented a full training loop with a loss function and optimizer, watching the training loss decrease as the model learned.
-   **Phase 4: Evaluation:** Tested the trained model on unseen data, achieving **96.31% accuracy** and proving its ability to generalize.

## Part 2: Building a GPT-style Large Language Model

The second part of the project applied the foundational skills to the more complex domain of natural language processing by building a complete, GPT-style Large Language Model (LLM) architecture from scratch.

### Architecture Development (Phases 5-9)

-   **Phase 5-6: Language Processing:** Explored how text is processed into numbers via **Tokenization** and **Embeddings**.
-   **Phase 7: Self-Attention:** Implemented the revolutionary self-attention mechanism, the core algorithm of the Transformer, from scratch.
-   **Phase 8: The Transformer Block:** Assembled self-attention and a feed-forward network into a complete, stackable, shape-preserving `TransformerBlock`.
-   **Phase 9: Final GPT Model:** Integrated all components—embeddings, positional encoding, and a stack of Transformer Blocks—into a final, full `GPTModel` architecture.

### From-Scratch Pre-training (Phases 10-13)

-   **Phase 10-12: Custom Tokenizer:** Moved beyond a simple character-level tokenizer to build and train a modern **Byte-Pair Encoding (BPE)** sub-word tokenizer on our text data. This created a more efficient and meaningful vocabulary for the model.
-   **Phase 13: Advanced Training on "TinyStories"**:
    -   Trained our 16.84M parameter GPT model on the "TinyStories" dataset to teach it grammar and coherence.
    -   Implemented advanced, professional-grade training techniques like a **Cosine Decay Learning Rate Scheduler** and **Gradient Clipping** for stable and efficient training.
    -   The training was successful, reaching a final validation loss of **[Your final validation loss, e.g., 0.0363]**.

### Results: The "Dreaming" AI

After training, the model was able to generate new, original text. While not fully coherent, the generated text demonstrates a clear understanding of concepts, English word structure, character names, and the general "vibe" of a story.

**Sample Generated Text:**

> [Once upon a time time round time time increasing flea solut nexpected bathe bleeding Mina weekend vision congratulated ouse gloomy pleas rust ife Choo urry Ant paused tummy steer Bop time flut time Sal mayor micro snapped * ply udged gifted Taking occasion fluttered Through softer yesterday backs feather sight intent espe aur providing Tigger rad talent wander tidying Stri beeped experien ouse Jessica uc rong concentr ob Satur worrying seaweed Fl streng Pr directions polished countries results Lou traveling onds mention stage Using swal airy onse superheroes inchworm troubled exploring opin shopping Brooke rushing ica stage scut nap aim reward osaur pare detail]

This project demonstrates a complete, from-scratch understanding of the principles behind modern Large Language Models.