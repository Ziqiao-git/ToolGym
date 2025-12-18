#!/usr/bin/env python3
"""
Context Manager for ReAct Agent - Two-Layer Compression Strategy

Layer 1: Tool Result Streaming Compression (Single result > 50K chars)
Layer 2: Trajectory Global Compression (Total context > 80% capacity)

Key Design Principles:
- Keep trajectory structure unchanged: List[{"thought": str, "action": dict, "result": str}]
- Only compress the "result" field content
- Never delete any reasoning steps
- Leave "thought" and "action" fields completely unchanged
"""
from __future__ import annotations

import logging
from typing import Dict, List, Any
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.common.logger import get_logger
from mcpuniverse.llm.openrouter import OpenRouterModel, OpenRouterConfig


class ContextManager:
    """
    Manages context window for ReAct agent with two-layer compression.

    Layer 1: Streaming compression for individual tool results
    Layer 2: Global compression when total context exceeds threshold
    """

    def __init__(
        self,
        llm: BaseLLM,
        summarizer_llm: BaseLLM = None,  # Optional separate LLM for summarization
        model_context_limit: int = 200000,  # Claude 3.5 Sonnet context window
        system_prompt_length: int = 5000,   # Estimated system prompt size
        instruction_length: int = 3000,     # Estimated instruction size
    ):
        """
        Initialize the context manager.

        Args:
            llm: Language model for main agent
            summarizer_llm: Optional separate LLM for summarization (if None, uses llm)
            model_context_limit: Total context window (in characters, ~2.5 chars per token)
            system_prompt_length: Estimated system prompt size
            instruction_length: Estimated instruction size
        """
        self.llm = llm
        self._logger = get_logger("ContextManager")

        # Create default fast/cheap summarizer model if not provided
        if summarizer_llm is None:
            summarizer_config = {"model_name": "google/gemini-2.5-flash"}
            self.summarizer_llm = OpenRouterModel(config=summarizer_config)
            self._logger.info("Using default summarizer: google/gemini-2.5-flash")
        else:
            self.summarizer_llm = summarizer_llm

        self.model_context_limit = model_context_limit
        self.system_prompt_length = system_prompt_length
        self.instruction_length = instruction_length

        # Layer 1 parameters
        self.single_result_threshold = 50000  # Trigger streaming if single result > 50K
        self.chunk_size = 50000  # Process 20K chars at a time

        # Layer 2 parameters
        self.global_compression_threshold = 0.8  # Trigger at 80% capacity
        self.result_compression_min_size = 5000  # Only compress results > 5K

        self.trajectory: List[Dict[str, Any]] = []

    # ========================================================================
    # LAYER 1: Tool Result Streaming Compression
    # ========================================================================

    async def process_tool_result(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: str,
    ) -> str:
        """
        Process a single tool result with streaming compression if needed.

        This is Layer 1 compression - handles individual results that are too large.

        Args:
            tool_name: Name of the tool that was called
            tool_args: Arguments passed to the tool
            result: The raw tool result string

        Returns:
            Compressed result string (or original if below threshold)
        """
        result_str = str(result)

        # If result is small enough, return as-is
        if len(result_str) <= self.single_result_threshold:
            return result_str

        self._logger.info(
            f"Tool result too large ({len(result_str)} chars), "
            f"applying streaming compression..."
        )

        # Apply streaming summarization
        compressed = await self._streaming_summarize(result_str, tool_name, tool_args)

        self._logger.info(
            f"Compressed {len(result_str)} → {len(compressed)} chars "
            f"({len(compressed)/len(result_str)*100:.1f}%)"
        )

        return compressed

    async def _streaming_summarize(
        self,
        result: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> str:
        """
        Streaming summarization: break large result into chunks and summarize each.

        Args:
            result: The large result string
            tool_name: Tool name for context
            tool_args: Tool arguments for context

        Returns:
            Summarized result string
        """
        chunks = []
        for i in range(0, len(result), self.chunk_size):
            chunks.append(result[i:i + self.chunk_size])

        self._logger.info(f"Breaking result into {len(chunks)} chunks for summarization")

        # Summarize each chunk independently
        summaries = []
        for idx, chunk in enumerate(chunks, 1):
            self._logger.info(f"📝 Summarizing chunk {idx}/{len(chunks)} ({len(chunk)} chars)...")

            prompt = f"""Aggressively compress this data chunk to ~20% of original size while preserving ALL critical information.

Tool: {tool_name}
Arguments: {tool_args}

Compression rules:
1. Extract ONLY essential data: numbers, key names, critical facts
2. Remove ALL redundant text, formatting, explanations
3. Use ultra-concise notation (abbreviations, symbols OK)
4. Target: ~{len(chunk) // 5} characters (current: {len(chunk)})
5. Prioritize: data points > descriptions

Chunk {idx}/{len(chunks)}:
{chunk}

Ultra-compressed output (aim for {len(chunk) // 5} chars):"""

            try:
                # Use summarizer_llm (fast cheap model) instead of main llm
                summary = await self.summarizer_llm.get_response_async(
                    system_message="You are an expert at aggressive data compression. Extract only the most critical information.",
                    user_message=prompt,
                    timeout=60  # Reduced timeout for fast model
                )
                self._logger.info(f"✓ Chunk {idx}/{len(chunks)} summarized successfully")
                summaries.append(f"[Part {idx}/{len(chunks)}]\n{summary}")
            except Exception as e:
                self._logger.error(f"❌ Failed to summarize chunk {idx}/{len(chunks)}: {type(e).__name__}: {str(e)}")
                # Fallback: truncate chunk
                summaries.append(f"[Part {idx}/{len(chunks)} - Error during summarization]\n{chunk[:1000]}...")

        # Combine all summaries
        combined = "\n\n".join(summaries)

        # If combined summary is still too long, recursively compress
        if len(combined) > self.single_result_threshold:
            self._logger.warning(
                f"Combined summary still too large ({len(combined)} chars), "
                f"applying second-level compression..."
            )
            combined = await self._second_level_compress(combined, tool_name, tool_args)

        return combined

    async def _second_level_compress(
        self,
        combined_summary: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> str:
        """
        Second-level compression if combined summary is still too long.

        Args:
            combined_summary: The combined summary from all chunks
            tool_name: Tool name for context
            tool_args: Tool arguments for context

        Returns:
            Further compressed summary
        """
        target_size = len(combined_summary) // 5  # Aim for 20% of combined size
        prompt = f"""Aggressively compress this pre-summarized data to ~20% of current size.

Tool: {tool_name}
Arguments: {tool_args}

Current size: {len(combined_summary)} chars
Target size: ~{target_size} chars

Rules:
1. Merge redundant information across parts
2. Keep only the most critical data points
3. Ultra-concise notation
4. Remove all fluff

Pre-summarized parts:
{combined_summary}

Ultra-compressed final summary (~{target_size} chars):"""

        try:
            final_summary = await self.summarizer_llm.get_response_async(
                system_message="You are an expert at aggressive data compression. Extract only the most critical information.",
                user_message=prompt,
                timeout=60  # Reduced timeout for fast model
            )
            return final_summary
        except Exception as e:
            self._logger.error(f"Second-level compression failed: {e}")
            # Emergency fallback: truncate
            return combined_summary[:40000] + "\n\n[Truncated due to compression error]"

    # ========================================================================
    # LAYER 2: Trajectory Global Compression
    # ========================================================================

    async def add_turn(
        self,
        thought: str,
        action: Dict[str, Any],
        result: str,
    ):
        """
        Add a reasoning turn to trajectory and trigger global compression if needed.

        Args:
            thought: The agent's thought/reasoning
            action: The action dict (server, tool, arguments)
            result: The observation/result (already processed by Layer 1 if needed)
        """
        # Add turn to trajectory (structure unchanged)
        self.trajectory.append({
            "thought": thought,
            "action": action,
            "result": result,
        })

        # Check if we need global compression
        current_size = self._calculate_current_context_size()
        threshold = self.model_context_limit * self.global_compression_threshold

        if current_size > threshold:
            self._logger.warning(
                f"Context size {current_size} exceeds threshold {threshold}, "
                f"triggering global compression..."
            )
            await self._compress_all_results()

    def _calculate_current_context_size(self) -> int:
        """
        Calculate total context size including all components.

        Returns:
            Estimated total context size in characters
        """
        # Fixed components
        total = self.system_prompt_length + self.instruction_length

        # Trajectory size
        for turn in self.trajectory:
            total += len(str(turn["thought"]))
            total += len(str(turn["action"]))
            total += len(str(turn["result"]))

        return total

    async def _compress_all_results(self):
        """
        Global compression: compress all result fields in trajectory.

        This is Layer 2 compression - triggered when total context is too large.

        CRITICAL: Only modifies the "result" field in each turn.
        The trajectory structure remains: List[{"thought": str, "action": dict, "result": str}]
        """
        self._logger.info(f"Starting global compression on {len(self.trajectory)} turns...")

        compressed_count = 0
        original_size = self._calculate_current_context_size()

        # Compress each turn's result field if it's large enough
        for idx, turn in enumerate(self.trajectory):
            result_str = str(turn["result"])

            # Skip if result is already small
            if len(result_str) <= self.result_compression_min_size:
                continue

            # Compress this result
            try:
                compressed_result = await self._compress_single_result(
                    result=result_str,
                    thought=turn["thought"],  # For context
                    action=turn["action"],    # For context
                    turn_index=idx,
                )

                # ONLY modify the result field - thought and action unchanged
                turn["result"] = compressed_result
                compressed_count += 1

                self._logger.info(
                    f"Turn {idx}: Compressed result "
                    f"{len(result_str)} → {len(compressed_result)} chars"
                )

            except Exception as e:
                self._logger.error(f"Failed to compress turn {idx} result: {e}")
                # Keep original result on error
                continue

        new_size = self._calculate_current_context_size()
        self._logger.info(
            f"Global compression complete: {compressed_count} results compressed, "
            f"total size {original_size} → {new_size} chars "
            f"({new_size/original_size*100:.1f}%)"
        )

    async def _compress_single_result(
        self,
        result: str,
        thought: str,
        action: Dict[str, Any],
        turn_index: int,
    ) -> str:
        """
        Compress a single result field using LLM.

        Args:
            result: The result string to compress
            thought: The thought from this turn (for context)
            action: The action from this turn (for context)
            turn_index: Index of this turn in trajectory

        Returns:
            Compressed result string
        """
        tool_name = action.get("tool", "unknown")
        tool_args = action.get("arguments", {})
        target_size = len(result) // 5  # Aim for 20% compression

        prompt = f"""Aggressively compress this tool result to ~20% of original size.

Turn #{turn_index}
Tool: {tool_name}
Args: {tool_args}

Current: {len(result)} chars → Target: ~{target_size} chars

Extract ONLY critical data. Ultra-concise notation OK.

Original:
{result}

Compressed (~{target_size} chars):"""

        try:
            compressed = await self.summarizer_llm.get_response_async(
                system_message="You are an expert at aggressive data compression. Extract only the most critical information.",
                user_message=prompt,
                timeout=60  # Reduced timeout for fast model
            )
            return compressed
        except Exception as e:
            self._logger.error(f"Compression failed for turn {turn_index}: {e}")
            # Fallback: simple truncation with ellipsis
            if len(result) > 10000:
                return result[:10000] + "\n\n[Truncated due to compression error]"
            return result

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """
        Get the current trajectory.

        Returns:
            List of turns with structure: [{"thought": str, "action": dict, "result": str}]
        """
        return self.trajectory

    def clear_trajectory(self):
        """Clear the trajectory (for new conversation)."""
        self.trajectory = []
        self._logger.info("Trajectory cleared")

    def get_context_usage_stats(self) -> Dict[str, Any]:
        """
        Get detailed context usage statistics.

        Returns:
            Dict with context usage information
        """
        current_size = self._calculate_current_context_size()
        usage_percent = (current_size / self.model_context_limit) * 100

        return {
            "total_size": current_size,
            "model_limit": self.model_context_limit,
            "usage_percent": round(usage_percent, 2),
            "turns_count": len(self.trajectory),
            "compression_threshold": int(self.model_context_limit * self.global_compression_threshold),
            "needs_compression": current_size > (self.model_context_limit * self.global_compression_threshold),
        }
