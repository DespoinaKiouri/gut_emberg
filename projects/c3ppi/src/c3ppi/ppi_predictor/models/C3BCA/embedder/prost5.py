import re
import torch
from transformers import T5Tokenizer, T5EncoderModel

from c3ppi.ppi_predictor.utils.config import MODEL_REGISTRY

class ProteinEmbeddingService:
    """
    A service that generates a mean-pooled embedding for a single protein (amino acid
    or 3Di structure) using the Rostlab/ProstT5 T5EncoderModel.

    Example 1 - Single Sequence Embedding:
        >>> service = SingleProteinEmbeddingService()
        >>> sequence = "PRTEINO"
        >>> mean_embedding = service.embed_single_sequence(sequence)
        >>> # mean_embedding -> 1D torch.Tensor of shape (hidden_size,)

    Example 2 - Batch Sequence Embedding:
        >>> service = SingleProteinEmbeddingService()
        >>> sequences = ["MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIE", "GAVLGLAIVAPYTLVLLTSVIGTILA"]
        >>> embeddings = service.embed_batch_sequences(sequences)
        >>> # embeddings -> List[torch.Tensor]
    """

    def __init__(self, model_name: str = "Rostlab/ProstT5_fp16"):
        """
        Initialize the SingleProteinEmbeddingService.

        This method loads the tokenizer and T5EncoderModel from the specified model
        name on Hugging Face Hub. It also determines whether a GPU is available and
        sets model precision accordingly (half-precision if GPU is available,
        otherwise full-precision).

        Args:
            model_name (str, optional):
                The Hugging Face model name to load. Defaults to "Rostlab/ProstT5".

        Raises:
            ValueError: If the model or tokenizer fail to load.
        """
        self.device = torch.device("cuda:0")

        self.cache_dir = MODEL_REGISTRY + "/ProstT5"

        # Load tokenizer
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=self.cache_dir, do_lower_case=False)
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer: {e}")

        # Load model
        try:
            self.model = T5EncoderModel.from_pretrained(model_name, cache_dir=self.cache_dir).to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to load T5EncoderModel: {e}")

        # Use half-precision on GPU, full-precision on CPU
        if self.device.type == "cuda":
            self.model.half()
        else:
            self.model.float()

        self.model.eval()  # evaluation mode

    def embed_single_sequence(
        self,
        sequence: str,
        return_as_list: bool = False
    ) -> torch.Tensor:
        """
        Generate a mean-pooled embedding for one protein sequence.

        Steps:
          1. Replace rare amino acids [U, Z, O, B] with X.
          2. Prepend <AA2fold> for uppercase sequences or <fold2AA> for lowercase.
          3. Tokenize, pad (single sequence doesn't require extensive padding logic).
          4. Retrieve last hidden state embeddings, skip prefix token.
          5. Compute mean-pooled embedding.

        Args:
            sequence (str):
                A single protein (amino acid) or 3Di structure sequence. Uppercase
                for amino acids, lowercase for 3Di.
            return_as_list (bool, optional):
                If True, returns the final embedding as a Python list instead of
                a torch.Tensor. Defaults to False.

        Returns:
            torch.Tensor:
                If `return_as_list` is False, returns a 1D torch.Tensor of shape
                (hidden_size,). If `return_as_list` is True, returns a Python list
                of floats (same length).

        Raises:
            ValueError: If the sequence is empty or contains invalid characters.
        """
        if not sequence:
            raise ValueError("No input sequence provided.")

        # Clean sequence (replace [U, Z, O, B] with X)
        seq_clean = re.sub(r"[UZOB]", "X", sequence)

        # Determine prefix token based on case (AA vs. 3Di)
        prefix = "<AA2fold>" if seq_clean.isupper() else "<fold2AA>"

        # Insert whitespace between characters
        spaced_seq = " ".join(list(seq_clean))
        final_seq = prefix + " " + spaced_seq

        # Encode
        with torch.no_grad():
            encoded = self.tokenizer.batch_encode_plus(
                [final_seq],  # single-sequence batch
                add_special_tokens=True,
                return_tensors="pt"
            ).to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask
            )

        # Extract embeddings
        last_hidden_state = outputs.last_hidden_state  # shape: (batch=1, seq_len, hidden_size)

        # Determine how many non-padding tokens exist (excluding prefix token)
        non_pad_tokens = (encoded.attention_mask[0] == 1).sum().item()
        # Skip the prefix token at index 0; the actual residue tokens start at index 1
        valid_emb = last_hidden_state[0, 1:non_pad_tokens]  # shape: (seq_len_without_prefix, hidden_size)

        if valid_emb.shape[0] == 0:
            # Edge case: if there's no valid residue token after prefix
            raise ValueError("Sequence too short or invalid after processing.")

        # Mean-pooled embedding
        mean_emb = valid_emb.mean(dim=0)

        if return_as_list:
            return mean_emb.cpu().tolist()

        return mean_emb
    
    def embed_batch_sequences(
        self,
        sequences: list[str],
        return_as_list: bool = False
    ) -> list[torch.Tensor] | list[list[float]]:
        """
        Embed a batch of protein sequences using ProstT5.

        Args:
            sequences (List[str]):
                A list of protein sequences (AA or 3Di).
            return_as_list (bool, optional):
                If True, returns a list of Python lists instead of torch.Tensors.

        Returns:
            List[torch.Tensor] or List[List[float]]:
                A list of mean-pooled embeddings, one per sequence.

        Raises:
            ValueError: If any sequence is empty or results in invalid embedding.
        """
        if not sequences:
            raise ValueError("Empty list of sequences provided.")

        cleaned_inputs = []
        valid_indices = []

        for i, seq in enumerate(sequences):
            if not seq:
                continue

            # Replace rare AAs
            seq_clean = re.sub(r"[UZOB]", "X", seq)
            prefix = "<AA2fold>" if seq_clean.isupper() else "<fold2AA>"
            spaced = " ".join(list(seq_clean))
            final_seq = f"{prefix} {spaced}"
            cleaned_inputs.append(final_seq)
            valid_indices.append(i)

        if not cleaned_inputs:
            raise ValueError("All sequences were invalid or empty.")

        with torch.no_grad():
            encoded = self.tokenizer.batch_encode_plus(
                cleaned_inputs,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask
            )

        batch_hidden = outputs.last_hidden_state  # [B, L, D]
        attention_mask = encoded.attention_mask  # [B, L]
        embeddings = []

        for i in range(batch_hidden.size(0)):
            # Tokens with attention = 1, skip the first prefix token
            non_pad = attention_mask[i].sum().item()
            if non_pad <= 1:
                raise ValueError(f"Sequence at batch index {valid_indices[i]} is too short or invalid.")

            valid_emb = batch_hidden[i, 1:non_pad]  # skip prefix token
            mean_emb = valid_emb.mean(dim=0)

            if return_as_list:
                embeddings.append(mean_emb.cpu().tolist())
            else:
                embeddings.append(mean_emb)

        return embeddings



if __name__ == "__main__":
    service = ProteinEmbeddingService()
    sequence = "PRTEINO"
    mean_embedding = service.embed_single_sequence(sequence, return_as_list=True)
    print(len(mean_embedding))

    sequences = [ 
    "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIE",
    "GAVLGLAIVAPYTLVLLTSVIGTILA"
    ]

    embeddings = service.embed_batch_sequences(sequences)

    print(embeddings[0].shape)  # torch.Size([1024])
    print(embeddings[1].shape)  # torch.Size([1024])