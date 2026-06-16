"""Out-of-process, OpenAI-compatible vLLM broker (the "vGate").

A thin reverse proxy in front of 1-2 stock ``vllm serve`` replicas that adds
replica discovery, least-outstanding routing, a global admission budget, and a
3-class weighted-fair queue. Business code (build streams + source_qa) consumes
it unchanged by pointing its ``base_url`` at the broker. See the v2 design doc.
"""
