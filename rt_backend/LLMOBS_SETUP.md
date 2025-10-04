# Datadog LLM Observability Setup

## Overview

LLM Observability automatically tracks all OpenAI API calls, including:
- **Latency** - Response time for each request
- **Token usage** - Input/output tokens and costs
- **Prompts & responses** - Full request/response data
- **Error tracking** - Failed requests and error details
- **Performance metrics** - Throughput, error rates

## Installation

```bash
pip install ddtrace>=2.9.0
```

## Configuration

Add to your `.env` file:

```bash
# Datadog Configuration (US1)
DD_API_KEY=your_datadog_api_key_here
DD_SITE=datadoghq.com  # US1 region (default)
ENV=production
```

**Important for US1:** Use `datadoghq.com` NOT `us1.datadoghq.com`

**Other regions:**
- US3: `us3.datadoghq.com`
- US5: `us5.datadoghq.com`
- EU1: `datadoghq.eu`
- AP1: `ap1.datadoghq.com`

## Features

### 1. Automatic OpenAI Tracing

LLMObs automatically instruments:
- `openai.ChatCompletion.create()`
- `openai.Completion.create()`
- `openai.Embedding.create()`

**No code changes needed!** Just enable LLMObs and all OpenAI calls are traced.

### 2. What's Tracked

For each OpenAI API call:

```json
{
  "span_id": "abc123",
  "trace_id": "xyz789",
  "operation": "openai.chat.completions",
  "model": "gpt-4o",
  "input_tokens": 150,
  "output_tokens": 85,
  "total_tokens": 235,
  "duration_ms": 1234,
  "status": "ok",
  "metadata": {
    "temperature": 0.3,
    "max_tokens": 500
  }
}
```

### 3. Integration with Logs

All logs sent via `logger.py` are automatically correlated with LLM traces using trace IDs.

## Usage

### Start the Backend

```bash
cd rt_backend
python api.py
```

You should see:
```
✓ Datadog logging enabled (US1)
✓ Datadog LLM Observability enabled (agentless)
✓ NBA AI Commentary API started
```

### View in Datadog

1. **LLM Observability Dashboard**
   - Go to: https://app.datadoghq.com/llm/overview
   - Filter: `ml_app:nba-ai-commentary`

2. **Traces**
   - Go to: https://app.datadoghq.com/apm/traces
   - Filter: `service:nba-ai-commentary`

3. **Logs + Traces Correlation**
   - Go to: https://app.datadoghq.com/logs
   - Click any log → See related trace
   - Click any trace → See related logs

## Key Metrics

### Available in Datadog

- **Request Rate**: `openai.chat.completions.request_rate`
- **Latency (p50, p95, p99)**: `openai.chat.completions.duration`
- **Token Usage**: `openai.chat.completions.tokens`
- **Error Rate**: `openai.chat.completions.error_rate`
- **Cost Estimation**: Based on token usage

### Custom Dashboards

Create dashboards to track:
- Token usage over time
- Average response time by model
- Error rate trends
- Cost per request

## Advanced: Manual Tracing

If you want to add custom spans:

```python
from ddtrace.llmobs import LLMObs

# Annotate a function
@LLMObs.workflow(name="video_analysis")
def analyze_video(frames, question):
    # Your code here
    pass

# Or use context manager
with LLMObs.workflow(name="custom_workflow") as span:
    span.annotate(
        input_data={"question": "What's happening?"},
        metadata={"timestamp": 45.0}
    )
    # ... your code ...
    span.annotate(output_data={"answer": "..."})
```

## Troubleshooting

### LLMObs not showing data?

1. **Check API key**:
   ```bash
   echo $DD_API_KEY
   ```

2. **Verify site setting**:
   ```bash
   echo $DD_SITE  # Should be: datadoghq.com for US1
   ```

3. **Check logs for errors**:
   Look for startup messages in console

4. **Wait 1-2 minutes**:
   Data may take time to appear in Datadog

### Common Issues

**"ddtrace not installed"**
```bash
pip install ddtrace>=2.9.0
```

**"Invalid API key"**
- Verify your DD_API_KEY in .env
- Ensure no extra quotes or whitespace

**"Wrong region"**
- Set DD_SITE correctly for your region
- US1: `datadoghq.com` (default)

## Example Queries

### In LLM Observability UI

```
# All video analysis requests
ml_app:nba-ai-commentary AND operation:video_analysis

# Slow requests (>2s)
ml_app:nba-ai-commentary AND duration:>2000

# High token usage (>1000 tokens)
ml_app:nba-ai-commentary AND total_tokens:>1000

# Errors only
ml_app:nba-ai-commentary AND status:error
```

## Best Practices

1. **Monitor token usage** to control costs
2. **Set up alerts** for high latency or error rates
3. **Track prompt performance** to optimize prompts
4. **Correlate with business metrics** (user questions, timestamps)

## Links

- [Datadog LLM Observability Docs](https://docs.datadoghq.com/llm_observability/)
- [ddtrace Python SDK](https://ddtrace.readthedocs.io/)
- [OpenAI Integration](https://docs.datadoghq.com/llm_observability/setup/sdk/python/#openai)
