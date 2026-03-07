import torch
from typing import Optional
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers.models.bert.modeling_bert import BertForPreTraining, BertForPreTrainingOutput
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup


@dataclass
class XBertForPreTrainingOutput(BertForPreTrainingOutput):
    masked_lm_loss: Optional[torch.FloatTensor] = None
    next_sentence_loss: Optional[torch.FloatTensor] = None
    discriminator_loss: Optional[torch.FloatTensor] = None


class XBertForPreTraining(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        # Add discriminator head for token replacement detection
        self.discriminator = torch.nn.Linear(config.hidden_size, 1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            next_sentence_label=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import BertTokenizer, BertForPreTraining
        >>> import torch

        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>> model = BertForPreTraining.from_pretrained('bert-base-uncased')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Dynamic masking for improved data utilization
        replaced_labels = None
        if labels is not None and self.training:
            # Rebuild original input_ids from masked input_ids and labels
            original_ids = input_ids.clone()
            masked_positions = (labels != -100)
            original_ids[masked_positions] = labels[masked_positions]
            
            # Apply dynamic span masking
            masked_lm_prob = 0.15
            replaced_labels = torch.zeros_like(input_ids, dtype=torch.float)
            for b in range(input_ids.shape[0]):
                seq = original_ids[b]
                # Find candidate positions (skip CLS, SEP, PAD, MASK)
                cand = []
                for i in range(1, len(seq) - 1):
                    if seq[i] not in [101, 102, 103, 0]:  # CLS, SEP, MASK, PAD
                        cand.append(i)
                
                num_to_mask = max(1, int(len(cand) * masked_lm_prob))
                masked_count = 0
                while masked_count < num_to_mask and len(cand) > 0:
                    # Choose span length: 1-3 with higher prob for shorter
                    span_len = min(len(cand), torch.randint(1, 4, (1,), device=input_ids.device).item())
                    
                    # Random start position
                    start_idx = torch.randint(0, len(cand) - span_len + 1, (1,), device=input_ids.device).item()
                    span_positions = cand[start_idx:start_idx + span_len]
                    
                    # Mask the span
                    for pos in span_positions:
                        rand = torch.rand(1, device=input_ids.device).item()
                        if rand < 0.8:
                            input_ids[b, pos] = 103  # [MASK]
                        elif rand < 0.9:
                            # Random token
                            random_token = torch.randint(100, self.config.vocab_size, (1,), device=input_ids.device).item()
                            input_ids[b, pos] = random_token
                        replaced_labels[b, pos] = 1.0
                        labels[b, pos] = seq[pos]
                    
                    masked_count += span_len
                    # Remove masked positions from candidates
                    for pos in reversed(span_positions):
                        if pos in cand:
                            cand.remove(pos)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        seq_relationship_score.view(-1, 2)

        # Discriminator for token replacement detection
        discriminator_logits = self.discriminator(sequence_output).squeeze(-1)
        discriminator_loss = None

        total_loss = None
        masked_lm_loss = None
        next_sentence_loss = None

        # masked_lm labels is necessary when training
        if labels is not None:
            loss_fct = CrossEntropyLoss(label_smoothing=0.1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = masked_lm_loss

        if next_sentence_label is not None:
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            if total_loss is not None:
                total_loss += next_sentence_loss
            else:
                total_loss = next_sentence_loss

        # Add discriminator loss
        if replaced_labels is not None:
            disc_loss_fct = BCEWithLogitsLoss()
            # Only compute loss for non-special tokens
            active_positions = (input_ids != 101) & (input_ids != 102) & (input_ids != 103) & (input_ids != 0)
            discriminator_loss = disc_loss_fct(
                discriminator_logits[active_positions].view(-1),
                replaced_labels[active_positions].view(-1)
            )
            if total_loss is not None:
                total_loss += 0.5 * discriminator_loss  # lambda = 0.5
            else:
                total_loss = 0.5 * discriminator_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return XBertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            masked_lm_loss=masked_lm_loss,
            next_sentence_loss=next_sentence_loss,
            discriminator_loss=discriminator_loss
        )


def build_optimizer(model, args):

    no_decay = ["bias", "LayerNorm.weight"]
    model_parameters = model.named_parameters()
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_parameters if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model_parameters if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                          args.num_warmup_steps,
                                                          args.num_train_steps)
    return optimizer, scheduler
