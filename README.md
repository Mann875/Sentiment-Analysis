# Sentiment-Analysis
Sentiment Analysis — VADER vs RoBERTa vs HuggingFace Pipeline
An end-to-end NLP project comparing three sentiment analysis approaches on Amazon product reviews. The project progresses from a classical bag-of-words lexicon method (VADER) through to a state-of-the-art transformer-based model (RoBERTa), with a final look at how easy HuggingFace's high-level pipeline makes the whole process.


## Dataset Overview
The dataset consists of Amazon product reviews with a 1–5 star Score and a free-text Text field. For computational efficiency, the notebook works on the first 500 reviews sampled from the full dataset.
ColumnDescriptionIdUnique review identifierTextFull review textScoreStar rating (1 = worst, 5 = best)
A bar chart of review counts by star rating is generated during EDA to understand the class distribution.

## Approaches
1. NLTK Basics (Tokenisation & NER)
Before scoring begins, the notebook demonstrates foundational NLTK operations on a sample review:

Tokenisation — splitting text into individual word tokens with nltk.word_tokenize
POS Tagging — labelling each token with its part-of-speech tag using nltk.pos_tag
Named Entity Recognition (NER) — chunking tagged tokens into named entity groups with nltk.chunk.ne_chunk


2. VADER Sentiment Scoring
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based tool tuned specifically for social media text.
How it works:

Stop words are removed
Each remaining word is looked up in a sentiment dictionary and scored
Scores are aggregated into a final compound value

Output scores per review:
ScoreDescriptionposProportion of positive sentimentneuProportion of neutral sentimentnegProportion of negative sentimentcompoundNormalised aggregate score (−1 to +1)
Scores are computed for all 500 reviews using SentimentIntensityAnalyzer, merged back onto the original DataFrame, and visualised with Seaborn bar plots — one for the compound score and a side-by-side grid of pos/neu/neg across all star ratings.
Key finding: Compound scores increase monotonically with star rating, confirming VADER broadly aligns with human-assigned scores.

3. RoBERTa Pretrained Transformer Model
Model: cardiffnlp/twitter-roberta-base-sentiment-latest from HuggingFace.
Unlike VADER, RoBERTa is a transformer that accounts for full sentence context — word meanings shift based on surrounding words.
Pipeline:

Tokenise input with AutoTokenizer
Pass tokens through AutoModelForSequenceClassification
Apply softmax to raw logits to get probabilities
Extract roberta_neg, roberta_neu, roberta_pos

Error handling: A try/except block catches RuntimeError for unusually long or malformed inputs, printing the offending Id and continuing.
Both VADER and RoBERTa scores are combined into a single results DataFrame for direct comparison.

4. Model Comparison & Error Analysis
Pairplot comparison (sns.pairplot) visualises all six sentiment scores — vader_neg, vader_neu, vader_pos, roberta_neg, roberta_neu, roberta_pos — coloured by star rating to reveal where the two models agree and diverge.
Interesting edge cases surface:
CaseDescriptionPositive 1-star reviewA review rated 1 star but scored as highly positive by RoBERTa/VADER — sarcasm or nuanced textNegative 5-star reviewA review rated 5 stars but flagged as negative sentiment — possibly ironic praise
These edge cases highlight the fundamental limitation of both methods: star rating ≠ text sentiment.

5. HuggingFace Transformers Pipeline (Bonus)
The notebook closes by showcasing HuggingFace's one-liner pipeline("sentiment-analysis"), which automatically downloads a default model and runs inference with minimal code — ideal for rapid prototyping.
pythonfrom transformers import pipeline
sent_pipeline = pipeline("sentiment-analysis")
sent_pipeline("I like Hot Chocolate")   # → POSITIVE
sent_pipeline("No")                     # → NEGATIVE

## Getting Started
Prerequisites
bashpip install pandas numpy matplotlib seaborn nltk tqdm transformers scipy torch
Download required NLTK data:
pythonimport nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
Run the Notebook
bashjupyter notebook Untitled.ipynb
Ensure reviews.csv is in the same directory. On first run, HuggingFace will download the RoBERTa model weights (~500MB) automatically.

## Tech Stack
LibraryPurposepandas / numpyData manipulationmatplotlib / seabornVisualisationnltkTokenisation, POS tagging, NER, VADERtransformersRoBERTa tokeniser, model & pipelinescipySoftmax for RoBERTa output normalisationtqdmProgress tracking for row-wise inferencetorchBackend for transformer model inference

## Key Takeaways

VADER is fast, interpretable, and requires no GPU — good for quick prototyping on short social-media-style text.
RoBERTa captures context and handles nuanced language far better, at the cost of higher compute.
HuggingFace Pipeline abstracts away all complexity for a quick sentiment check with a single line of code.
Neither model perfectly mirrors star ratings — human reviewers are complex, and sarcasm/irony remain hard NLP problems.
