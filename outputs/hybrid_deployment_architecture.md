# Executive Brain: Hybrid Deployment Architecture
## Kimi K2 (Self-Hosted) + API Fallbacks (Claude, GPT)

## Document Metadata

**Purpose**: Production architecture combining self-hosted Kimi K2 with API model fallbacks
**Version**: 1.0
**Date**: 2025-11-08
**Status**: Production specification
**Evidence Level**: All architectural decisions traced to rationale

---

## 1. Architecture Overview

### 1.1 Design Rationale

**Primary Model**: Kimi K2 Thinking (Self-Hosted)
**Reasoning**:
1. **Cost control**: Self-hosting eliminates per-token API costs for high-volume usage
2. **Data privacy**: Sensitive organizational data never leaves infrastructure
3. **Latency**: Local inference faster than API roundtrips (ASSUMPTION: <100ms network latency)
4. **Availability control**: Not dependent on external service uptime
5. **Customization**: Can fine-tune, adjust parameters without API constraints

**Fallback Models**: Claude Opus (API), GPT-4 (API)
**Reasoning**:
1. **Reliability**: If self-hosted infrastructure fails, fallback to cloud API
2. **Burst capacity**: Handle traffic spikes beyond self-hosted capacity
3. **Specialization**: Certain tasks may perform better on specific models
4. **Benchmarking**: Compare Kimi K2 performance against SOTA proprietary models

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              Effort Regulation Orchestrator                  │
│  (Analyzes task, determines effort, selects model)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Model Router  │
              │  (with health  │
              │   checking)    │
              └────────┬───────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌─────────────┐ ┌─────────────┐
│   Kimi K2    │ │Claude Opus  │ │   GPT-4     │
│(Self-Hosted) │ │    (API)    │ │    (API)    │
│              │ │             │ │             │
│ Priority: 1  │ │ Priority: 2 │ │ Priority: 3 │
│ 95% traffic  │ │ 4% traffic  │ │ 1% traffic  │
└──────────────┘ └─────────────┘ └─────────────┘
     │                 │               │
     └─────────────────┴───────────────┘
                       │
                       ▼
              ┌────────────────┐
              │     Result     │
              └────────────────┘
```

**Traffic Distribution** (ASSUMPTION - based on reliability targets):
- **Kimi K2**: 95% (primary, handles all normal traffic)
- **Claude**: 4% (fallback when Kimi K2 down, A/B testing)
- **GPT-4**: 1% (benchmarking, specific tasks)

**Reasoning**: Self-hosted should handle vast majority. API usage kept minimal to control costs while maintaining reliability.

---

## 2. Model Provider Implementations

### 2.1 Kimi K2 Self-Hosted Provider

**See**: `kimi_k2_self_hosting_guide.md` for complete deployment

```python
# providers/kimi_k2_provider.py

class KimiK2SelfHostedProvider(LLMProvider):
    """
    Provider for self-hosted Kimi K2 Thinking via vLLM.
    """

    def __init__(self, endpoint: str = "http://kimi-k2-service:8000/v1"):
        self.endpoint = endpoint
        self.client = openai.OpenAI(base_url=endpoint, api_key="dummy")
        self._health_check_endpoint = endpoint.replace("/v1", "/health")

    def is_healthy(self) -> bool:
        """Check if self-hosted instance is available."""
        try:
            response = requests.get(self._health_check_endpoint, timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        # Implementation from self-hosting guide
        ...

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supports_thinking": True,
            "supports_tools": True,
            "context_window": 256000,
            "thinking_budget_range": (1000, 256000),
            "strengths": [
                "reasoning",
                "tool_orchestration",
                "agentic_workflows",
                "long_context"
            ],
            "deployment": "self-hosted",
            "cost_per_request": "infrastructure_only",  # No per-token cost
            "latency_estimate_ms": 500  # ASSUMPTION: Local inference latency
        }
```

### 2.2 Claude Opus API Provider

```python
# providers/claude_provider.py

import anthropic
from typing import Dict, Any, Iterator, Tuple
from .base import LLMProvider

class ClaudeOpusProvider(LLMProvider):
    """
    Provider for Claude Opus via Anthropic API (fallback).
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        """Synchronous generation via Claude API."""
        response = self.client.messages.create(
            model="claude-opus-4-20250514",  # Latest Opus
            max_tokens=params.get("max_tokens", 4096),
            temperature=params.get("temperature", 0.7),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def stream(self, prompt: str, params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Streaming generation."""
        with self.client.messages.stream(
            model="claude-opus-4-20250514",
            max_tokens=params.get("max_tokens", 4096),
            temperature=params.get("temperature", 0.7),
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield {"type": "content", "content": text}

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supports_thinking": True,  # Extended thinking mode
            "supports_tools": True,
            "context_window": 200000,
            "thinking_budget_range": (0, 0),  # No explicit budget control
            "strengths": [
                "reasoning",
                "writing",
                "tool_use",
                "following_instructions"
            ],
            "deployment": "api",
            "cost_per_request": "variable_per_token",
            "latency_estimate_ms": 1500  # ASSUMPTION: API roundtrip latency
        }

    def get_thinking_budget(self) -> Tuple[int, int, int]:
        return (0, 0, 0)  # Claude doesn't expose explicit thinking budget

    def estimate_tokens(self, text: str) -> int:
        """Claude tokenization is ~chars/4."""
        return len(text) // 4

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": "Claude Opus 4",
            "version": "20250514",
            "provider": "anthropic",
            "deployment": "api",
            "license": "proprietary"
        }

    def effort_to_params(self, effort_score: float, strategy: str) -> Dict[str, Any]:
        """Map effort to Claude parameters."""
        params = {}

        # Extended thinking for high-effort tasks
        if effort_score > 0.6:
            params["thinking"] = "extended"  # ASSUMPTION: Parameter name
        elif effort_score > 0.4:
            params["thinking"] = "enabled"
        # else: default (no explicit thinking)

        # Temperature
        params["temperature"] = 0.3 + (effort_score * 0.6)

        # Max tokens
        if effort_score < 0.3:
            params["max_tokens"] = 2048
        elif effort_score < 0.6:
            params["max_tokens"] = 4096
        else:
            params["max_tokens"] = 8192

        return params
```

**⚠️ ASSUMPTION**: Claude's "extended thinking" parameter name. Actual API may differ. See Anthropic docs: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking

### 2.3 GPT-4 API Provider

```python
# providers/gpt_provider.py

import openai
from typing import Dict, Any, Iterator, Tuple
from .base import LLMProvider

class GPT4Provider(LLMProvider):
    """
    Provider for GPT-4 via OpenAI API (benchmarking only).
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-2025-04-09",  # Latest GPT-4
            messages=[{"role": "user", "content": prompt}],
            max_tokens=params.get("max_tokens", 4096),
            temperature=params.get("temperature", 0.7)
        )
        return response.choices[0].message.content

    def stream(self, prompt: str, params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-2025-04-09",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=params.get("max_tokens", 4096),
            temperature=params.get("temperature", 0.7),
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield {
                    "type": "content",
                    "content": chunk.choices[0].delta.content
                }

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supports_thinking": False,  # No exposed thinking trace
            "supports_tools": True,
            "context_window": 128000,
            "thinking_budget_range": (0, 0),
            "strengths": [
                "general_purpose",
                "coding",
                "instruction_following"
            ],
            "deployment": "api",
            "cost_per_request": "variable_per_token",
            "latency_estimate_ms": 1200  # ASSUMPTION
        }

    def get_thinking_budget(self) -> Tuple[int, int, int]:
        return (0, 0, 0)

    def estimate_tokens(self, text: str) -> int:
        """GPT tokenization is ~chars/4 (similar to Claude)."""
        return len(text) // 4

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": "GPT-4 Turbo",
            "version": "2025-04-09",
            "provider": "openai",
            "deployment": "api",
            "license": "proprietary"
        }

    def effort_to_params(self, effort_score: float, strategy: str) -> Dict[str, Any]:
        """Map effort to GPT parameters."""
        params = {}

        # GPT doesn't have thinking budget, use temperature and max_tokens
        params["temperature"] = 0.3 + (effort_score * 0.6)

        if effort_score < 0.3:
            params["max_tokens"] = 2048
        elif effort_score < 0.6:
            params["max_tokens"] = 4096
        else:
            params["max_tokens"] = 8192

        return params
```

---

## 3. Model Router with Health Checking

### 3.1 Routing Logic

```python
# core/model_router.py

from typing import Dict, Any, Optional
from providers.base import LLMProvider
import logging

logger = logging.getLogger(__name__)

class HybridModelRouter:
    """
    Routes requests to optimal model with health-based fallback.
    """

    def __init__(self, providers: Dict[str, LLMProvider]):
        self.providers = providers  # {name: provider instance}
        self.primary = "kimi_k2_self_hosted"
        self.fallback_chain = [
            "kimi_k2_self_hosted",
            "claude_opus_api",
            "gpt4_api"
        ]
        self.health_cache = {}  # {provider_name: (is_healthy, last_check_time)}
        self.health_check_interval = 30  # seconds

    def route(self, task_analysis: Dict[str, Any]) -> str:
        """
        Select optimal provider based on:
        1. Health status (primary check)
        2. Task requirements
        3. Fallback chain
        """

        # User override (for testing, debugging)
        if task_analysis.get("force_provider"):
            return task_analysis["force_provider"]

        # Try primary model if healthy
        if self._is_healthy(self.primary):
            logger.debug(f"Routing to primary: {self.primary}")
            return self.primary

        # Primary unhealthy, try fallback chain
        logger.warning(f"Primary model {self.primary} unhealthy, trying fallbacks")

        for provider_name in self.fallback_chain[1:]:  # Skip primary (already checked)
            if provider_name in self.providers and self._is_healthy(provider_name):
                logger.info(f"Falling back to {provider_name}")
                return provider_name

        # All models unhealthy - critical failure
        logger.error("All models unhealthy!")
        raise AllModelsUnavailableError("No healthy LLM providers available")

    def _is_healthy(self, provider_name: str) -> bool:
        """
        Check if provider is healthy (with caching).
        """
        now = time.time()

        # Check cache
        if provider_name in self.health_cache:
            is_healthy, last_check = self.health_cache[provider_name]
            if now - last_check < self.health_check_interval:
                return is_healthy

        # Perform health check
        provider = self.providers.get(provider_name)
        if not provider:
            return False

        try:
            # For self-hosted: HTTP health check
            if hasattr(provider, 'is_healthy'):
                is_healthy = provider.is_healthy()
            else:
                # For API providers: test with minimal request
                is_healthy = self._test_provider(provider)

            # Update cache
            self.health_cache[provider_name] = (is_healthy, now)
            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed for {provider_name}: {e}")
            self.health_cache[provider_name] = (False, now)
            return False

    def _test_provider(self, provider: LLMProvider) -> bool:
        """Test API provider with minimal request."""
        try:
            # Simple test request (should be fast and cheap)
            result = provider.generate("test", {"max_tokens": 1})
            return len(result) > 0
        except Exception:
            return False


class AllModelsUnavailableError(Exception):
    """Raised when no healthy models available."""
    pass
```

**Reasoning for Health Checking**:
1. **Self-hosted can fail**: Hardware failures, OOM, process crashes
2. **API can fail**: Rate limits, service outages, network issues
3. **Fast recovery**: Cached health checks avoid latency overhead
4. **Automatic fallback**: No manual intervention needed

### 3.2 Fallback Scenarios

| Scenario | Primary (Kimi K2) | Fallback | Reasoning |
|----------|-------------------|----------|-----------|
| **Normal operation** | ✅ Healthy | Not used | Expected 95% of time |
| **Self-hosted OOM** | ❌ Crashed | Claude API | vLLM process crashed, restart takes 5-10min |
| **GPU failure** | ❌ Hardware fail | Claude API | Hardware issue, may take hours/days to fix |
| **Network partition** | ❌ Unreachable | Claude API | K8s node isolated, networking issue |
| **High load** | ⚠️ Slow (>10s latency) | Optionally use Claude for burst | Trade cost for latency |
| **API rate limit** | ✅ Healthy | Use Kimi K2 only | Primary advantage of self-hosting |

**Decision Point**: When to proactively route to API for performance?

**Option A**: Never (cost-optimized)
- Only use API when self-hosted completely fails
- Accept higher latency during peak load
- Reasoning: Self-hosting cost already sunk, API adds marginal cost

**Option B**: Latency-triggered (performance-optimized)
- If Kimi K2 latency >10s, route new requests to Claude
- Reasoning: User experience > cost
- **Trade-off**: Higher API costs during peak times

**Recommendation**: Start with Option A, add Option B if latency becomes issue.

---

## 4. Configuration

### 4.1 Model Registry

```yaml
# config/model_registry.yaml

models:
  # PRIMARY: Self-hosted Kimi K2
  kimi_k2_self_hosted:
    provider_class: KimiK2SelfHostedProvider
    provider_kwargs:
      endpoint: http://kimi-k2-service:8000/v1
    health_check:
      enabled: true
      endpoint: http://kimi-k2-service:8000/health
      interval_seconds: 30
      timeout_seconds: 2
    capabilities:
      supports_thinking: true
      supports_tools: true
      context_window: 256000
      thinking_budget_range: [1000, 256000]
      strengths: [reasoning, tool_orchestration, agentic_workflows, long_context]
    enabled: true
    priority: 1
    usage_policy:
      default_for_tasks: all
      max_concurrent_requests: 20  # vLLM batch size limit

  # FALLBACK 1: Claude Opus API
  claude_opus_api:
    provider_class: ClaudeOpusProvider
    provider_kwargs:
      api_key_env: ANTHROPIC_API_KEY
    health_check:
      enabled: true
      method: test_request  # Send minimal request to test
      interval_seconds: 60
    capabilities:
      supports_thinking: true
      supports_tools: true
      context_window: 200000
      strengths: [reasoning, writing, tool_use, instruction_following]
    enabled: true  # Enable for fallback
    priority: 2
    usage_policy:
      use_when:
        - kimi_k2_self_hosted_unhealthy
        - explicit_user_request
      max_requests_per_day: 1000  # Cost control

  # FALLBACK 2 / BENCHMARKING: GPT-4 API
  gpt4_api:
    provider_class: GPT4Provider
    provider_kwargs:
      api_key_env: OPENAI_API_KEY
    health_check:
      enabled: true
      method: test_request
      interval_seconds: 60
    capabilities:
      supports_thinking: false
      supports_tools: true
      context_window: 128000
      strengths: [general_purpose, coding, instruction_following]
    enabled: false  # Disable by default, enable for benchmarking
    priority: 3
    usage_policy:
      use_when:
        - benchmarking_mode
        - explicit_user_request
      max_requests_per_day: 100  # Very limited

routing:
  strategy: primary_with_fallback
  primary: kimi_k2_self_hosted
  fallback_chain:
    - kimi_k2_self_hosted
    - claude_opus_api
    - gpt4_api
  fallback_triggers:
    - primary_unhealthy
    - primary_timeout  # If response time > 30s
  health_check_interval_seconds: 30
  retry_on_failure: true
  max_retries: 2

monitoring:
  log_model_selection: true
  track_fallback_usage: true
  alert_on_fallback: true  # Send alert if using Claude/GPT (indicates primary issue)
```

### 4.2 Environment Variables

```bash
# .env

# Kimi K2 Self-Hosted
KIMI_K2_ENDPOINT=http://kimi-k2-service:8000/v1
# No API key needed (self-hosted)

# Claude API (Fallback)
ANTHROPIC_API_KEY=sk-ant-xxxxx  # From https://console.anthropic.com/

# GPT-4 API (Benchmarking only)
OPENAI_API_KEY=sk-xxxxx  # From https://platform.openai.com/

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Alerting
ALERT_ON_FALLBACK=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxxxx
```

---

## 5. Monitoring & Alerting

### 5.1 Key Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Model selection tracking
model_requests_total = Counter(
    'llm_requests_total',
    'Total requests by model',
    ['model_name', 'status']  # status: success, error, timeout
)

# Fallback tracking
fallback_events_total = Counter(
    'llm_fallback_events_total',
    'Count of fallback events',
    ['from_model', 'to_model', 'reason']  # reason: unhealthy, timeout, error
)

# Health status
model_health_status = Gauge(
    'llm_model_health',
    'Health status (1=healthy, 0=unhealthy)',
    ['model_name']
)

# Latency by model
model_latency_seconds = Histogram(
    'llm_inference_latency_seconds',
    'Inference latency by model',
    ['model_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)
```

### 5.2 Alerting Rules

```yaml
# prometheus/alerts.yml

groups:
  - name: llm_health
    interval: 30s
    rules:
      # Alert if primary model unhealthy for >5 minutes
      - alert: PrimaryModelDown
        expr: llm_model_health{model_name="kimi_k2_self_hosted"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Primary LLM (Kimi K2) is unhealthy"
          description: "Kimi K2 self-hosted has been unhealthy for >5min. Falling back to Claude API."

      # Alert if fallback is being used
      - alert: UsingFallbackModel
        expr: rate(llm_requests_total{model_name=~"claude.*|gpt.*"}[5m]) > 0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Using API fallback models"
          description: "Traffic is being routed to API models (Claude/GPT). Check primary model health."

      # Alert if API usage exceeds daily budget
      - alert: APIUsageHigh
        expr: sum(increase(llm_requests_total{model_name=~"claude.*|gpt.*"}[24h])) > 1000
        labels:
          severity: warning
        annotations:
          summary: "API usage exceeds daily budget"
          description: "Claude/GPT usage exceeded 1000 requests/day. Cost control threshold reached."

      # Alert if all models unhealthy
      - alert: AllModelsDown
        expr: sum(llm_model_health) == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "ALL LLM models are unhealthy"
          description: "No healthy LLM providers available. System cannot process requests."
```

### 5.3 Grafana Dashboard

**Panels**:
1. **Model Health Status** (gauge)
   - Green: Kimi K2 healthy
   - Yellow: Using fallback
   - Red: All models down

2. **Request Distribution** (pie chart)
   - % requests by model (Kimi K2 vs Claude vs GPT)

3. **Fallback Event Timeline** (graph)
   - When did fallbacks occur? How long?

4. **Latency Comparison** (histogram)
   - Latency distribution per model (compare Kimi K2 vs Claude vs GPT)

5. **API Usage** (counter)
   - Daily Claude requests (track toward budget limit)
   - Daily GPT requests

**Dashboard JSON**: (Available in `/monitoring/grafana-dashboard.json`)

---

## 6. Operational Procedures

### 6.1 Primary Model Failure Response

**Detection**: Prometheus alert `PrimaryModelDown` fires

**Automated Response**:
1. Health check fails → Router automatically routes to Claude API
2. Alert sent to Slack/PagerDuty
3. Metrics logged (fallback event)

**Manual Response** (on-call engineer):
1. Check Kimi K2 logs: `kubectl logs deployment/kimi-k2-inference -n executive-brain`
2. Common issues:
   - **OOM**: Reduce `--gpu-memory-utilization`, reduce batch size, restart
   - **GPU hang**: Check `nvidia-smi`, may need hard reboot
   - **vLLM crash**: Check logs for errors, restart deployment
3. Restart: `kubectl rollout restart deployment/kimi-k2-inference -n executive-brain`
4. Wait for health check to pass (~5-10min for model reload)
5. Traffic automatically routes back to Kimi K2 when healthy

**Reasoning**: Automate fallback (no manual intervention needed), but require human to fix root cause.

### 6.2 API Quota Management

**Problem**: Claude/GPT APIs have rate limits and cost money.

**Strategy**:
1. **Daily Budget**: Max 1000 Claude requests/day (configurable)
2. **Rate Limiting**: Max 10 Claude req/s (avoid hitting Anthropic rate limits)
3. **Circuit Breaker**: If budget exceeded, disable API fallback (fail requests instead)

**Implementation**:
```python
class APIQuotaManager:
    def __init__(self, daily_limit: int = 1000):
        self.daily_limit = daily_limit
        self.request_count = 0
        self.reset_time = datetime.now() + timedelta(days=1)

    def check_quota(self, model_name: str) -> bool:
        """Return True if quota available."""
        # Only apply to API models
        if "api" not in model_name:
            return True

        # Check if new day (reset counter)
        if datetime.now() > self.reset_time:
            self.request_count = 0
            self.reset_time = datetime.now() + timedelta(days=1)

        # Check quota
        if self.request_count >= self.daily_limit:
            logger.error(f"API quota exceeded: {self.request_count}/{self.daily_limit}")
            return False

        return True

    def increment(self, model_name: str):
        """Increment request counter."""
        if "api" in model_name:
            self.request_count += 1
```

**Reasoning**: Prevent runaway API costs if primary model is down for extended period.

---

## 7. A/B Testing & Benchmarking

### 7.1 Comparing Kimi K2 vs Claude

**Scenario**: Validate Kimi K2 performs as well as Claude on our workload

**Method**:
1. **Shadow Traffic**: Send 10% of requests to BOTH Kimi K2 AND Claude
2. **Compare Outputs**: Log both responses
3. **Measure Quality**: Ragas evaluation on both
4. **Analyze Results**: After 1 week, compare:
   - Quality scores (faithfulness, relevance)
   - Latency
   - User feedback (if applicable)

**Implementation**:
```python
class ABTestRouter:
    def __init__(self, router: HybridModelRouter, ab_test_ratio: float = 0.1):
        self.router = router
        self.ab_test_ratio = ab_test_ratio

    def route_with_ab_test(self, task, user_id):
        """Route with A/B testing."""
        # Normal routing
        primary_model = self.router.route(task)

        # A/B test: X% of requests also sent to comparison model
        if random.random() < self.ab_test_ratio:
            comparison_model = "claude_opus_api"

            # Send to BOTH models (parallel)
            primary_result = self.router.providers[primary_model].generate(task)
            comparison_result = self.router.providers[comparison_model].generate(task)

            # Log both for comparison
            log_ab_test_result(
                task=task,
                primary_model=primary_model,
                primary_result=primary_result,
                comparison_model=comparison_model,
                comparison_result=comparison_result
            )

            # Return primary result to user
            return primary_result
        else:
            # Normal path (no A/B test)
            return self.router.providers[primary_model].generate(task)
```

**Reasoning**: Empirically validate Kimi K2 performance against SOTA proprietary model.

---

## 8. Future Extensions

### 8.1 Multi-Model Routing

**Scenario**: Different models for different task types

**Example**:
- **Deep reasoning**: Kimi K2 (best BrowseComp, GPQA scores)
- **Code generation**: Qwen Coder 32B (self-hosted, specialized)
- **Writing**: Claude Opus (best at long-form content)

**Implementation**: Extend router with task-type detection and model specialization mapping.

### 8.2 Load Balancing Multiple Self-Hosted Instances

**Scenario**: Deploy 2x Kimi K2 instances for redundancy and throughput

```yaml
models:
  kimi_k2_instance_1:
    endpoint: http://kimi-k2-pod-1:8000/v1
    priority: 1

  kimi_k2_instance_2:
    endpoint: http://kimi-k2-pod-2:8000/v1
    priority: 1  # Same priority = load balance

routing:
  load_balancing:
    strategy: round_robin  # or least_connections
    health_aware: true  # Skip unhealthy instances
```

**Reasoning**: High availability and horizontal scaling of self-hosted.

---

## 9. Decision Rationale Summary

| Decision | Choice | Rationale | Evidence |
|----------|--------|-----------|----------|
| **Primary Model** | Kimi K2 Self-Hosted | Cost, privacy, control | Self-hosting eliminates per-token costs |
| **Fallback Model** | Claude Opus | Reliability, performance | Beats GPT-5 on benchmarks |
| **Health Checking** | 30s interval | Balance freshness vs overhead | Industry standard (K8s liveness probes) |
| **API Budget** | 1000 req/day | Cost control | Assumption based on 5% failure rate |
| **Fallback Strategy** | Automatic | Minimize downtime | No manual intervention required |
| **A/B Testing** | 10% shadow traffic | Empirical validation | Standard A/B test ratio |

**All decisions** traceable to specific reasoning or evidence.

---

## 10. Assumptions & Validation Required

| Assumption | Validation Method | Risk if Wrong |
|------------|-------------------|---------------|
| **Kimi K2 latency <500ms** | Load test in staging | Poor UX if >2s |
| **Claude API reliable** | Monitor uptime over 30 days | Fallback may also fail |
| **1000 Claude req/day sufficient** | Measure actual fallback frequency | May need higher budget |
| **Health checks don't impact performance** | Benchmark with/without health checks | May add latency |
| **vLLM compatible with Kimi K2** | Deploy and test | May need TGI or custom serving |

**ACTION**: Validate all assumptions in staging before production.

---

**Document Version**: 1.0
**Status**: Production specification
**Next Review**: After 30 days of production operation
