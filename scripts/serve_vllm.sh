#!/bin/bash
# Start the LLM server (OpenAI-compatible API on :8000). UNTESTED: needs a GPU and the model weights.
# Phi-3.5-mini-instruct (Microsoft, MIT licence, not gated). Swap MODEL for Llama 3.x / Mistral if you have access.
MODEL=${MODEL:-microsoft/Phi-3.5-mini-instruct}
exec vllm serve "$MODEL" --max-model-len "${MAX_LEN:-8192}" --port "${PORT:-8000}" --dtype bfloat16
