import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_distill.utils.logging import logger


class QwenModel:
    """Unified wrapper for Qwen2.5 providing generation and logit extraction."""

    def __init__(
        self,
        model_name_or_path: str,
        dtype: str = "bfloat16",
        max_seq_length: int = 4096,
        device: Optional[str] = None,
    ):
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.dtype = dtype_map.get(dtype, torch.bfloat16)
        self.max_seq_length = max_seq_length

        logger.info(f"Loading model: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )

        if device is not None:
            self.model = self.model.to(device)
        self.device = next(self.model.parameters()).device
        logger.info(f"Model loaded on {self.device}, dtype={self.dtype}")

    @torch.no_grad()
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 128,
    ) -> str:
        """Generate text from ChatML messages (for trajectory collection)."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_length
        ).to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        # Decode only the newly generated tokens
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def get_action_logprobs(
        self,
        messages: List[Dict[str, str]],
        actions: List[str],
        require_grad: bool = False,
    ) -> torch.Tensor:
        """Compute log P(action | messages) for each candidate action.

        For each action, we tokenize it, append to the prompt, and compute
        the average log-probability of the action tokens.

        Args:
            messages: ChatML conversation history up to the current step.
            actions: List of candidate action strings.
            require_grad: If True, keep computation graph for backprop (student).
                          If False, run under no_grad (teacher).

        Returns:
            log_probs: Tensor of shape [num_actions] with log-probabilities.
        """
        # Build the prompt (everything before the action)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_length
        ).to(self.device)
        prompt_len = prompt_ids["input_ids"].shape[1]

        log_probs = []
        for action_text in actions:
            lp = self._score_action(prompt, prompt_len, action_text, require_grad)
            log_probs.append(lp)

        return torch.stack(log_probs)

    def _score_action(
        self,
        prompt: str,
        prompt_len: int,
        action_text: str,
        require_grad: bool,
    ) -> torch.Tensor:
        """Score a single action: compute average log-prob of action tokens."""
        full_text = prompt + action_text
        inputs = self.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=self.max_seq_length
        ).to(self.device)
        input_ids = inputs["input_ids"]
        action_len = input_ids.shape[1] - prompt_len

        if action_len <= 0:
            return torch.tensor(float("-inf"), device=self.device, dtype=self.dtype)

        context = torch.enable_grad() if require_grad else torch.no_grad()
        with context:
            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, seq_len, vocab_size]

            # Get logits at positions [prompt_len-1, ..., seq_len-2] predicting
            # tokens at positions [prompt_len, ..., seq_len-1] (the action tokens)
            action_logits = logits[0, prompt_len - 1 : prompt_len - 1 + action_len, :]
            action_token_ids = input_ids[0, prompt_len : prompt_len + action_len]

            token_log_probs = F.log_softmax(action_logits, dim=-1)
            # Gather log-prob for each actual action token
            selected = token_log_probs.gather(1, action_token_ids.unsqueeze(1)).squeeze(1)
            avg_log_prob = selected.mean()

        return avg_log_prob

    def get_full_logits(
        self,
        messages: List[Dict[str, str]],
        require_grad: bool = False,
    ) -> torch.Tensor:
        """Get full vocabulary logits at the next-token position.

        Returns tensor of shape [vocab_size].
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_length
        ).to(self.device)

        context = torch.enable_grad() if require_grad else torch.no_grad()
        with context:
            outputs = self.model(**inputs)
            return outputs.logits[0, -1, :]  # [vocab_size]

    @classmethod
    def from_config(cls, config: dict) -> "QwenModel":
        return cls(
            model_name_or_path=config["name_or_path"],
            dtype=config.get("dtype", "bfloat16"),
            max_seq_length=config.get("max_seq_length", 4096),
        )
