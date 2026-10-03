# Smart MCQ Solver

A complete machine learning and deep learning pipeline for solving
multiple-choice questions by ranking the most probable answer options.
The project was developed for the **Smart MCQ Solver Challenge** and
evaluates three different approaches: a traditional TF-IDF retrieval
baseline, a custom BiLSTM with Attention model, and a pretrained RoBERTa
multiple-choice model.

## Project Objective

The objective is to predict and rank the **top three most probable
answers** for each multiple-choice question. The project explores
progressively more sophisticated approaches, starting from lexical
similarity and moving to contextual deep learning and pretrained
transformer models.

The official competition metric is **MAP@3 (Mean Average Precision at
3)**.

## Dataset

The dataset contains:

-   `train.csv` --- 2,000 questions with five answer choices (A--E) and
    the correct answer.
-   `test.csv` --- 500 questions with five answer choices but without
    the correct answer.
-   `sample_submission.csv` --- required submission format.

The training data contains 8 columns:

`id`, `prompt`, `A`, `B`, `C`, `D`, `E`, `answer`

The test data contains the same fields except for `answer`.

## Project Workflow

``` text
Raw Dataset
    |
    v
Exploratory Data Analysis
    |
    v
Data Preprocessing
    |-- Label Encoding
    |-- Text Validation
    |-- Text Cleaning
    |-- Question–Option Pair Formatting
    |
    v
80:20 Stratified Train/Validation Split
    |
    +-------------------+--------------------+
    |                   |                    |
    v                   v                    v
TF-IDF +             BiLSTM +            RoBERTa-base
Cosine Similarity     Attention           Multiple Choice
    |                   |                    |
    +-------------------+--------------------+
                        |
                        v
              Accuracy / Macro F1 / MAP@3
                        |
                        v
              Final Model Selection
                        |
                        v
              Test Set Top-3 Ranking
                        |
                        v
                  submission.csv
```

## Exploratory Data Analysis

The project first examines the structure and quality of the dataset.

Key checks include:

-   Dataset dimensions and column types
-   Missing-value analysis
-   Duplicate-row analysis
-   Correct-answer class distribution
-   Question/prompt length distribution
-   Answer-option length distribution
-   Random MCQ examples

The report identifies **183 duplicate rows** after excluding the ID
column. These rows were retained during model development because they
were exact duplicates with the same question, options, and answer label.

## Data Preprocessing

Four main preprocessing steps are used.

### 1. Label Encoding

The categorical answer labels `A–E` are converted to numerical
identifiers:

``` text
A -> 0
B -> 1
C -> 2
D -> 3
E -> 4
```

### 2. Text Data Validation

Question prompts and answer options are explicitly converted to string
format so that all textual inputs can be processed consistently.

### 3. Text Cleaning

The preprocessing intentionally avoids aggressive linguistic
transformations. Instead, it normalizes formatting by:

-   Removing leading/trailing whitespace
-   Replacing multiple spaces
-   Replacing newline characters
-   Replacing tab characters

Punctuation, capitalization, and sentence structure are preserved to
retain contextual information.

### 4. MCQ Formatting

Each question is paired separately with each of its five candidate
answers:

``` text
Question: <question text>
Answer: <candidate option>
```

This produces a consistent **Question--Option pair** representation that
can be used across the different models.

## Train/Validation Split

The training data is divided using an **80:20 stratified split**:

-   Training: 1,600 questions
-   Validation: 400 questions
-   Random seed: 42

Stratification is used to approximately preserve the answer-label
distribution across the two subsets.

## Models

### 1. TF-IDF + Cosine Similarity

The first approach is a traditional information-retrieval baseline.

The pipeline:

1.  Create Question--Option pairs.
2.  Normalize whitespace.
3.  Fit a `TfidfVectorizer` on the training corpus.
4.  Convert text into sparse TF-IDF vectors.
5.  Calculate cosine similarity.
6.  Rank the five candidate answers.
7.  Select the top three options.

This model provides a simple lexical-similarity baseline without
learning contextual representations.

### 2. BiLSTM + Attention

The second approach is a custom deep learning model implemented from
scratch using PyTorch.

The Question--Option pairs are transformed through:

``` text
Tokenization
    ↓
Vocabulary Construction
    ↓
Integer Encoding
    ↓
Padding
    ↓
Embedding
    ↓
Bidirectional LSTM
    ↓
Attention
    ↓
Dropout
    ↓
Linear Layer
    ↓
Prediction Score
```

The original 1,600 training questions are transformed into **8,000
Question--Option pairs** because each question has five candidate
answers.

The model predicts the relevance of each question--option pair
independently. The five resulting scores are then used to rank the
candidate answers.

Key training components documented in the project include:

-   Maximum sequence length: 64 tokens
-   Batch size: 32
-   Dropout: 0.3
-   Loss: `BCEWithLogitsLoss`
-   Optimizer: Adam
-   Learning rate: `1e-3`

### 3. RoBERTa-base

The third approach uses a pretrained RoBERTa transformer model adapted
for multiple-choice question answering.

The pipeline is:

``` text
Question + Options A–E
        ↓
RoBERTa Tokenizer
        ↓
Input IDs + Attention Masks
        ↓
RoBERTa Encoder
        ↓
Multiple-Choice Classification Head
        ↓
Answer Ranking
```

The project uses Hugging Face's `AutoModelForMultipleChoice` and
`Trainer` API.

Documented fine-tuning configuration:

-   Model: `roberta-base`
-   Epochs: 3
-   Batch size: 8
-   Learning rate: `2e-5`
-   Weight decay: 0.01
-   Maximum sequence length: 128
-   Best checkpoint selected using validation Macro F1

Weights & Biases (W&B) is used to track training and validation metrics.

## Evaluation Metrics

The models are evaluated using:

### Accuracy

Measures whether the highest-ranked prediction matches the correct
answer.

### Macro F1

Computes the F1 score independently for each answer class and then
averages the five class scores.

### MAP@3

The official competition metric. It rewards placing the correct answer
within the top three ranked predictions:

``` text
Correct answer ranked 1st -> 1
Correct answer ranked 2nd -> 1/2
Correct answer ranked 3rd -> 1/3
Correct answer outside top 3 -> 0
```

## Validation Results

The report's comparative validation results are:

  ----------------------------------------------------------------------------
  Model          Type                 Accuracy        Macro F1           MAP@3
  -------------- ------------- --------------- --------------- ---------------
  TF-IDF +       Traditional            0.0934          0.0929          0.2518
  Cosine         Baseline                                      
  Similarity                                                   

  BiLSTM +       Scratch Deep           0.9874          0.9800          0.9886
  Attention      Learning                                      

  RoBERTa-base   Pretrained             0.9066          0.9076          0.9382
                 Transformer                                   
  ----------------------------------------------------------------------------

The validation results show that the custom BiLSTM + Attention approach
achieved the highest reported validation scores among the three
approaches in the project.

## Final Competition Submission

After model comparison, the BiLSTM + Attention model was selected for
final training and used to generate predictions for the unseen test
dataset.

For each test question:

1.  Generate a score for every Question--Option pair.
2.  Rank the five candidate options by predicted probability.
3.  Select the top three options.
4.  Format the predictions according to the competition submission
    format.
5.  Save the result as `submission.csv`.

The final Kaggle submission reported in the project achieved:

**MAP@3: 0.74563**

The report also records a competition rank of **958**.

## Key Learnings

-   Exploratory data analysis helps identify dataset quality issues
    before modeling.
-   Consistent Question--Option pair formatting provides a common
    representation across different modeling approaches.
-   Simple lexical retrieval provides a useful baseline for comparison.
-   A custom BiLSTM with Attention can learn contextual relationships
    without relying on a pretrained language model.
-   Pretrained transformers provide contextual representations with
    comparatively less task-specific preprocessing.
-   MAP@3 is useful for ranking-based multiple-choice tasks where the
    top three predictions matter.

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   PyTorch
-   Hugging Face Transformers
-   Weights & Biases
-   tqdm
-   Kaggle

## Repository Contents

``` text
.
├── Smart_MCQ_Solver.ipynb
├── Smart_MCQ_Solve_report.pdf
├── README.md
└── submission.csv
```

`submission.csv` is generated by the notebook after test-set prediction.

## Running the Project

The notebook was developed for a Kaggle environment and expects the
competition dataset to be available through the Kaggle input directory.

Open:

``` text
Smart_MCQ_Solver.ipynb
```

and execute the notebook sections in order:

1.  Import dependencies
2.  Load and inspect data
3.  Perform EDA
4.  Preprocess the dataset
5.  Create the train/validation split
6.  Train and evaluate the TF-IDF baseline
7.  Train and evaluate the BiLSTM + Attention model
8.  Fine-tune and evaluate RoBERTa
9.  Compare model performance
10. Train the selected final model
11. Predict the test set
12. Generate `submission.csv`

## Project Report

For the detailed methodology, diagrams, exploratory analysis, training
results, and comparative evaluation, see:

**`Smart_MCQ_Solve_report.pdf`**
