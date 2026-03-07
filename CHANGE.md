# BERT Pretraining Repository Changes

## Overview
This repository has been enhanced with modern language model pretraining techniques to improve data utilization, model expressiveness, and training efficiency while maintaining compatibility with existing infrastructure.

## Changes Made

### 1. Hugging Face Dataset Support (pt/config.py, pt/create_pretraining_data.py, pt/requirements.txt)
- **Added parameters** in `config.py`:
  - `--use_hf_dataset`: Enable Hugging Face dataset loading
  - `--hf_dataset_name`: Dataset name (e.g., 'wikitext')
  - `--hf_dataset_config`: Dataset configuration
  - `--hf_dataset_split`: Dataset split to use
- **Modified `create_pretraining_data.py`**:
  - Added `datasets` import
  - Implemented dataset loading logic for Hugging Face datasets
  - Maintains backward compatibility with file-based input
- **Updated `requirements.txt`**: Added `datasets` dependency

### 2. Enhanced Pretraining Model (pt/model.py)
- **Dynamic Masking**: Replaced static masking with dynamic masking during training
  - Rebuilds original sequences from pre-masked data
  - Applies random masking on-the-fly for each training batch
  - Improves data utilization by providing diverse masking patterns

- **Token Replacement Detection (Discriminator)**:
  - Added discriminator head for binary classification of token replacements
  - Inspired by ELECTRA architecture
  - Loss = MLM Loss + 0.5 × Discriminator Loss
  - Enhances model's understanding of token-level perturbations

- **Span-level Masking**:
  - Modified masking to target contiguous spans of 1-3 tokens
  - Better captures phrase-level language structure
  - More realistic corruption patterns compared to single-token masking

- **Label Smoothing**:
  - Applied label smoothing (0.1) to MLM loss
  - Reduces overfitting and improves generalization
  - Modern regularization technique for better model robustness

- **Updated Output Structure**:
  - Extended `XBertForPreTrainingOutput` dataclass to include `discriminator_loss`
  - Maintains backward compatibility

## Technical Details

### Dynamic Masking Process
1. Reconstruct original input_ids from masked inputs and labels
2. Randomly select spans for masking
3. Apply corruption (80% [MASK], 10% random token, 10% unchanged)
4. Track replaced positions for discriminator training

### Discriminator Training
- Binary classification on whether each token was artificially replaced
- Trained jointly with MLM objective
- Improves sample efficiency compared to traditional MLM

### Span Masking Distribution
- Span lengths: 1-3 tokens with uniform distribution
- Non-overlapping spans to maximize coverage
- Excludes special tokens ([CLS], [SEP], [PAD], [MASK])

## Benefits
- **Higher Data Utilization**: Dynamic masking provides infinite variations
- **Better Expressiveness**: Span masking and discriminator capture richer linguistic patterns
- **Improved Efficiency**: ELECTRA-style training reduces computational overhead
- **Enhanced Robustness**: Label smoothing prevents overfitting
- **Infrastructure Compatibility**: All changes are internal, no script modifications needed

## Usage
The enhanced model works seamlessly with existing training scripts:

```bash
# Create data with Hugging Face datasets
python create_pretraining_data.py --use_hf_dataset True --hf_dataset_name wikitext --hf_dataset_config wikitext-103-raw-v1

# Train with improved pretraining
python run_pretraining.py --train_tfrecord_dir ../data/pt_tfrecord/
```

All improvements are automatically applied during training without requiring parameter changes.