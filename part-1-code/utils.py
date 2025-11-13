import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import string   
import random
import argparse
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

## Before transform accuracy:0.93092
## After transform accuracy:0.81956
random.seed(0)

def custom_transform(example):
    """
    Polarity-Flipping Transformation:
    Focus on confusing the sentiment signal by:
    1. Strategic negation insertion (flips meaning)
    2. Replacing positive words with negative synonyms and vice versa
    3. Adding contradictory statements
    4. Sarcasm markers
    5. Context removal (keeping only confusing parts)
    
    This simulates: sarcastic reviews, contradictory opinions, mistaken sentiment
    """
    
    # Positive sentiment words
    positive_words = {
        "amazing", "awesome", "excellent", "fantastic", "great", "perfect", "wonderful",
        "outstanding", "incredible", "superb", "brilliant", "beautiful", "touching",
        "enjoyable", "funny", "hilarious", "smart", "clever", "moving", "emotional",
        "heartwarming", "charming", "delightful", "spectacular", "masterpiece",
        "good", "nice", "cool", "love", "loved", "like", "liked", "favorite", "best",
        "engaging", "thrilling", "powerful", "entertaining", "impressive",
        "wonderfully", "greatest", "inspiring", "fascinating"
    }
    
    # Negative sentiment words
    negative_words = {
        "bad", "boring", "awful", "terrible", "horrible", "disappointing", "poor",
        "worse", "worst", "weak", "predictable", "flat", "stupid", "dumb", "ugly",
        "painful", "mediocre", "lame", "messy", "slow", "nonsense", "pointless",
        "forgettable", "uninteresting", "unfunny", "annoying", "garbage", "trash",
        "pathetic", "ridiculous", "flawed", "waste", "uninspired", "hate", "hated"
    }
    
    # Antonym pairs (positive -> negative mapping)
    antonyms = {
        "good": "bad", "great": "terrible", "love": "hate", "loved": "hated",
        "like": "dislike", "best": "worst", "amazing": "awful", "perfect": "flawed",
        "wonderful": "horrible", "excellent": "poor", "beautiful": "ugly",
        "smart": "stupid", "funny": "unfunny", "entertaining": "boring",
        "enjoyable": "painful", "fantastic": "disappointing", "brilliant": "dumb",
        "incredible": "mediocre", "outstanding": "forgettable"
    }
    
    # Reverse mapping
    antonyms_reverse = {v: k for k, v in antonyms.items()}
    
    # Sarcasm/contradiction markers
    sarcasm_markers = [
        "yeah right", "sure", "obviously", "totally", "absolutely",
        "of course", "naturally", "clearly"
    ]
    
    # Contradictory connectors
    contradictions = [
        "but actually", "however", "although", "despite that", "on the other hand",
        "but then again", "even though", "yet"
    ]
    
    def get_antonym(word, is_positive):
        """Get opposite sentiment word"""
        lw = word.lower()
        if is_positive and lw in antonyms:
            return antonyms[lw]
        elif not is_positive and lw in antonyms_reverse:
            return antonyms_reverse[lw]
        return None
    
    text = example["text"]
    sentences = sent_tokenize(text)
    new_sentences = []
    
    for sent_idx, sent in enumerate(sentences):
        words = word_tokenize(sent)
        new_words = []
        
        # Strategy 1: Add sarcasm marker at sentence start (30%)
        if random.random() < 0.3 and len(words) > 5:
            new_words.append(random.choice(sarcasm_markers))
            new_words.append(",")
        
        for i, w in enumerate(words):
            lw = w.lower()
            
            # Strategy 2: Replace with ANTONYM (40% for positive, 40% for negative)
            if lw in positive_words and random.random() < 0.4:
                antonym = get_antonym(lw, True)
                if antonym:
                    # Keep capitalization
                    if w[0].isupper():
                        antonym = antonym.capitalize()
                    new_words.append(antonym)
                    continue
            
            if lw in negative_words and random.random() < 0.4:
                antonym = get_antonym(lw, False)
                if antonym:
                    if w[0].isupper():
                        antonym = antonym.capitalize()
                    new_words.append(antonym)
                    continue
            
            # Strategy 3: Flip with negation (50% for sentiment words)
            if (lw in positive_words or lw in negative_words) and random.random() < 0.5:
                # Insert "not" or "never" before the word
                if random.random() < 0.5:
                    new_words.append("not")
                else:
                    new_words.append("never")
            
            new_words.append(w)
        
        # Strategy 4: Add contradictory phrase in middle of sentence (20%)
        if random.random() < 0.2 and len(new_words) > 8:
            mid = len(new_words) // 2
            new_words.insert(mid, ",")
            new_words.insert(mid + 1, random.choice(contradictions))
            new_words.insert(mid + 2, ",")
        
        new_sentence = TreebankWordDetokenizer().detokenize(new_words)
        
        # Strategy 5: Add contradictory follow-up sentence (15%)
        if random.random() < 0.15 and len(new_sentence.split()) > 5:
            contradiction = random.choice([
                "Not really though.",
                "Just kidding.",
                "I'm being sarcastic.",
                "But the opposite is true.",
                "That's a lie."
            ])
            new_sentence = new_sentence + " " + contradiction
        
        new_sentences.append(new_sentence)
    
    # Strategy 6: Keep only 60% of sentences (aggressive context removal)
    if len(new_sentences) > 5 and random.random() < 0.4:
        keep_count = max(3, int(len(new_sentences) * 0.6))
        # Randomly select which sentences to keep
        indices = random.sample(range(len(new_sentences)), keep_count)
        indices.sort()
        new_sentences = [new_sentences[i] for i in indices]
    
    # Strategy 7: Reverse sentiment-heavy sentences (30%)
    if random.random() < 0.3 and len(new_sentences) > 2:
        # Find and reverse a sentence
        idx = random.randint(0, len(new_sentences) - 1)
        sent_words = new_sentences[idx].split()
        if len(sent_words) > 5:
            # Add explicit contradiction
            new_sentences[idx] = "Actually, the opposite: " + new_sentences[idx]
    
    # Strategy 8: Shuffle sentences (50%)
    if random.random() < 0.5 and len(new_sentences) > 2:
        random.shuffle(new_sentences)
    
    # Strategy 9: Insert random neutral sentence (20%)
    if random.random() < 0.2:
        neutral_fillers = [
            "The movie exists.",
            "It's a film.",
            "There are actors in it.",
            "It has a runtime.",
            "Things happen in the plot.",
            "The movie is on screen."
        ]
        insert_pos = random.randint(0, len(new_sentences))
        new_sentences.insert(insert_pos, random.choice(neutral_fillers))
    
    example["text"] = " ".join(new_sentences)
    return example
