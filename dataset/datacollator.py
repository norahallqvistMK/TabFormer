from typing import Dict, List, Tuple, Union
import torch
from transformers import DataCollatorForLanguageModeling
import torch

class TransDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):

    def _tensorize_batch(
         self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
     ) -> torch.Tensor:
         # In order to accept both lists of lists and lists of Tensors
         if isinstance(examples[0], (list, tuple)):
             examples = [torch.Tensor(e) for e in examples]
         length_of_first = examples[0].size(0)
         are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
         if are_tensors_same_length:
             return torch.stack(examples, dim=0)
         else:
             if self.tokenizer._pad_token is None:
                 raise ValueError(
                     "You are attempting to pad samples but the tokenizer you are using"
                     f" ({self.tokenizer.__class__.__name__}) does not have one."
                 )
             return self.pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)
             
    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(examples)
        sz = batch.shape
        if self.mlm:
            batch = batch.view(sz[0], -1)
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs.view(sz), "labels": labels.view(sz)}
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": labels}

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove "
                "the --mlm flag if you want to use this tokenizer. "
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability
        # defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class TransDataCollatorForFineTuning(DataCollatorForLanguageModeling):
    def __call__(
        self, examples: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Handle tuples of (input_ids, labels) returned from dataset
        
        Args:
            examples: List of tuples where each tuple is (input_ids_tensor, label_tensor)
        
        Returns:
            Dictionary with 'input_ids' and 'labels' tensors
        """
        # Extract input_ids and labels from tuples
        input_ids_list = [item[0] for item in examples]  # First element of each tuple
        labels_list = [item[1] for item in examples]     # Second element of each tuple
        
        # Stack the tensors to create batches
        input_ids = torch.stack(input_ids_list)
        labels = torch.stack(labels_list)
        
        return {"input_ids": input_ids, "labels": labels}
