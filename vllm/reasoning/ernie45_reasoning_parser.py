# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Optional, Union

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


@ReasoningParserManager.register_module("ernie45")
class Ernie45ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Ernie45 thinking model.
    The Ernie45 thinking model ouput format is
        abc\n</think>\n\n\n<response>\ndef\n</response>\n
    or  abc\n</think>\ndef
    or  abc\n</think>\n\n\n<tool_call>\nxyz\n</tool_call>\n
    """

    think_start_token_id: int
    think_end_token_id: int
    response_start_token_id: int
    response_end_token_id: int
    tool_call_start_token_id: int
    tool_call_end_token_id: int

    think_start_token: str = "<think>"
    think_end_token: str = "</think>"
    response_start_token: str = "<response>"
    response_end_token: str = "</response>"
    tool_call_start_token: str = "<tool_call>"
    tool_call_end_token: str = "</tool_call>"
    newline_token: str = "<0x0A>"

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

        self.think_start_token_id = self.vocab.get(self.think_start_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)
        self.response_start_token_id = self.vocab.get(
            self.response_start_token)
        self.response_end_token_id = self.vocab.get(self.response_end_token)
        self.tool_call_start_token_id = self.vocab.get(
            self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self.newline_token_id = self.vocab.get(self.newline_token)

        self.parser_token_ids = [
            self.think_start_token_id, self.think_end_token_id,
            self.response_start_token_id, self.response_end_token_id,
            self.tool_call_start_token_id, self.tool_call_end_token_id
        ]

        if self.think_start_token_id is None or self.think_end_token_id is None:
            raise RuntimeError(
                "Ernie45 reasoning parser could not locate think start/end "
                "tokens in the tokenizer!")

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.think_end_token_id in input_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content after the end tokens
        """
        if self.think_end_token_id not in input_ids[:-1]:
            return []
        else:
            return input_ids[input_ids.index(self.think_end_token_id) + 1:]

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        Extract reasoning content from a delta message.
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.
        The Ernie45 thinking model ouput format is
            abc\n</think>\n\n\n<response>\ndef\n</response>\n
        or  abc\n</think>\ndef
        or  abc\n</think>\n\n\n<tool_call>\nxyz\n</tool_call>\n
        - 'abc' goes to reasoning_content
        - 'def' goes to content
        """
        # Skip single special tokens
        if len(delta_token_ids) == 1 and (delta_token_ids[0] in [
                self.think_start_token_id, self.think_end_token_id,
                self.response_start_token_id, self.response_end_token_id
        ]):
            return None

        # No <think> in previous or delta, also need to check for </think>.
        # Because the model may have generated </think> without <think>
        if self.think_end_token_id in delta_token_ids:
            # </think> in delta with more tokens,
            # extract reasoning content and content
            think_end_index = delta_text.find(self.think_end_token)
            reasoning_content = delta_text[:think_end_index]
            content = delta_text[think_end_index + len(self.think_end_token):]
            content = content.lstrip("\n")
            response_start_idx = content.find(self.response_start_token)
            response_end_idx = content.rfind(self.response_end_token)
            if response_start_idx != -1:
                content = content[response_start_idx +
                                  len(self.response_start_token):]
            if response_end_idx != -1:
                content = content[:response_end_idx]
            return DeltaMessage(
                reasoning_content=reasoning_content,
                content=content if content else None,
            )
        elif self.think_end_token_id in previous_token_ids:
            # </think> in previous, thinking content ends
            content = delta_text
            if self.response_start_token_id in delta_token_ids:
                content = content.lstrip("\n")
                response_start_idx = content.find(self.response_start_token)
                content = content[response_start_idx +
                                  len(self.response_start_token):]
                # if have </response>, remove it
                response_end_idx = content.rfind(self.response_end_token)
                if response_end_idx != -1:
                    content = content[:response_end_idx]
            elif self.response_end_token_id in delta_token_ids:
                response_end_idx = content.rfind(self.response_end_token)
                content = content[:response_end_idx]
            # remove \n after </think> or <response> or </response>
            if previous_token_ids[-1] in self.parser_token_ids and \
                (len(delta_token_ids) > 0 and \
                    delta_token_ids[0] == self.newline_token_id):
                content = content.lstrip("\n")
            # remove \n after </think>\n
            if (len(previous_token_ids) > 1 and \
                previous_token_ids[-2] == self.think_end_token_id) and \
                (len(delta_token_ids) > 0 and \
                    delta_token_ids[0] == self.newline_token_id):
                content = content.lstrip("\n")

            return DeltaMessage(content=content if content else None)
        else:
            # no </think> in previous or delta, reasoning content continues
            return DeltaMessage(reasoning_content=delta_text)

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from the model output.
        The Ernie45 thinking model ouput format is
            abc\n</think>\n\n\n<response>\ndef\n</response>\n
        or  abc\n</think>\ndef
        or  abc\n</think>\n\n\n<tool_call>\nxyz\n</tool_call>\n

        - 'abc' goes to reasoning_content
        - 'def' goes to content
        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Check if the start token is present in the model output, remove it
        # if it is present.
        model_output_parts = model_output.partition(self.think_start_token)
        model_output = model_output_parts[2] if model_output_parts[
            1] else model_output_parts[0]

        # We assume the reasoning content is always at the start.
        if self.think_end_token not in model_output:
            return model_output, None
        else:
            reasoning_content, _, content = model_output.partition(
                self.think_end_token)

            if content:
                start_idx = content.find(self.response_start_token)
                end_idx = content.rfind(self.response_end_token)
                # Simultaneously existing and in the correct order
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    content = content[start_idx +
                                      len(self.response_start_token):end_idx]
                    if content.startswith("\n"):
                        content = content[1:]

            # If the end token is not found, return the model output as is.
            # It should not happen since we already checked for the presence
            # of the end token.
            # If generation stops right after end-of-think, return null content
            final_content = content or None

            return reasoning_content, final_content
