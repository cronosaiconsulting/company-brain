# Executive Brain v2.1: Model-Agnostic Architecture with Effort Regulation

## Document Overview

**Purpose**: Theoretical framework for model-agnostic Executive Brain with sophisticated effort regulation
**Version**: 2.1.1 (evidence-backed revision - adds assumption markers and disclaimers)
**Date**: 2025-11-09
**Key Principles**:
- **Model Agnostic**: Easy switching between LLMs (Kimi K2, DeepSeek, Qwen, Llama, Claude, GPT, etc.)
- **Deployment Flexible**: On-premise or bare metal cloud (Vultr, Hetzner, OVH)
- **Theoretical Framework**: Architecture patterns, not implementation specifics
- **Future-Proof**: Design for models that don't exist yet

---

## ⚠️ Evidence & Assumption Transparency

**This document uses the following markers to indicate evidence status:**

- **✅ VERIFIED**: Claim backed by official documentation, research papers, or verified testing
- **⚠️ ASSUMPTION**: Reasonable inference requiring verification before deployment
- **⛔ UNVERIFIED**: Placeholder pending access to official specifications

**Important Notes**:
1. **Numeric Parameters**: Most threshold values, formulas, and coefficients are SUGGESTED STARTING POINTS based on general ML/LLM practices. They are NOT empirically optimized for this specific system.
2. **Model Capabilities**: Some provider implementations reference API parameters that are INFERRED from model architecture but NOT verified from official API documentation.
3. **Performance Estimates**: Capacity estimates (user counts, request volumes) are ROUGH GUIDELINES requiring load testing.

**Before Production Deployment**:
- Review all ⚠️ ASSUMPTION markers and verify against actual deployed systems
- Calibrate numeric parameters via A/B testing and empirical measurement
- Update provider implementations with actual API schemas
- Run load tests to validate capacity estimates

See `evidence_traceability_audit.md` for complete analysis of claims and assumptions.

---

## 1. Core Architectural Principle: LLM Abstraction Layer

### 1.1 Problem Statement

**Current Industry Reality**:
- New SOTA models released monthly (Kimi K2, DeepSeek R1, Qwen 2.5, Llama 4, etc.)
- Each model has different strengths (reasoning, coding, tool use, languages, etc.)
- Different parts of system may benefit from different models
- Hard-coding to specific model = technical debt

**Design Goal**:
> The system should treat LLMs as **swappable inference engines**, not architectural dependencies.

### 1.2 LLM Abstraction Layer Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM ABSTRACTION LAYER                         │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Unified LLM Interface (API Contract)            │  │
│  │                                                           │  │
│  │  interface LLMProvider {                                 │  │
│  │    generate(prompt, params) -> response                  │  │
│  │    stream(prompt, params) -> iterator<chunk>             │  │
│  │    getCapabilities() -> {thinking, tools, context, ...}  │  │
│  │    getThinkingBudget() -> (min, max, default)           │  │
│  │    estimateTokens(text) -> count                         │  │
│  │  }                                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                       │
│         ┌────────────────┼────────────────┐                     │
│         ▼                ▼                ▼                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │ Kimi K2    │  │ DeepSeek   │  │ Qwen 2.5   │  [+ more]     │
│  │ Provider   │  │ R1 Provider│  │ Provider   │               │
│  └────────────┘  └────────────┘  └────────────┘               │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │       Model Router (Strategy Selection)                   │  │
│  │  - Route tasks to appropriate model based on:            │  │
│  │    * Task type (reasoning, coding, tool use, etc.)       │  │
│  │    * Model capabilities                                  │  │
│  │    * Availability (fallback if primary down)             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Model Provider Interface

```python
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Tuple

class LLMProvider(ABC):
    """
    Unified interface for all LLM providers.
    Allows hot-swapping models without changing orchestration logic.
    """

    @abstractmethod
    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Synchronous generation.

        Args:
            prompt: Input prompt (may include system message, context, etc.)
            params: Model-specific parameters (temperature, max_tokens, thinking_budget, etc.)

        Returns:
            Generated response text
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Streaming generation (for real-time feedback).

        Yields:
            Chunks of type:
            - {"type": "thinking", "content": "..."} - reasoning trace
            - {"type": "content", "content": "..."} - actual response
            - {"type": "tool_call", "name": "...", "args": {...}} - tool invocation
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return model capabilities for routing decisions.

        Returns:
            {
                "supports_thinking": bool,  # Can show reasoning trace
                "supports_tools": bool,      # Native tool calling
                "context_window": int,       # Max context in tokens
                "thinking_budget_range": (min, max),  # For effort regulation
                "strengths": ["reasoning", "coding", "multilingual", ...],
                "weaknesses": [...]
            }
        """
        pass

    @abstractmethod
    def get_thinking_budget(self) -> Tuple[int, int, int]:
        """
        Return (min, max, default) thinking budget in tokens.
        For models without explicit thinking budget, return (0, 0, 0).
        """
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for given text (for budget management).
        """
        pass

    @abstractmethod
    def get_model_metadata(self) -> Dict[str, Any]:
        """
        Return metadata for logging and monitoring.

        Returns:
            {
                "model_name": str,
                "version": str,
                "provider": str,  # e.g., "moonshot", "deepseek", "anthropic"
                "deployment": str,  # "self-hosted" or "api"
                "license": str
            }
        """
        pass
```

### 1.4 Concrete Implementations

#### Example 1: Kimi K2 Provider

```python
class KimiK2Provider(LLMProvider):
    """
    ⚠️ IMPORTANT - API PARAMETER ASSUMPTIONS:

    This provider implementation assumes Kimi K2 exposes the following parameters
    via its API when deployed with vLLM or similar serving infrastructure:

    ASSUMED PARAMETERS (NOT VERIFIED from official Moonshot AI API docs):
    - max_steps: Maximum reasoning steps the model can take
    - thinking_budget_per_step: Token budget allocated per reasoning step

    VERIFIED PARAMETERS:
    - model: "kimi-k2-thinking" ✅ (confirmed from Hugging Face)
    - messages: OpenAI-compatible chat format ✅ (standard)
    - temperature, max_tokens: Standard parameters ✅

    TODO BEFORE DEPLOYMENT:
    1. Deploy Kimi K2 with vLLM and inspect actual API schema
    2. Update parameter names if they differ from assumptions
    3. Test thinking budget controls actually work
    4. Document actual parameter ranges and defaults

    The parameter names used here are based on logical inference from model
    capabilities, NOT official documentation. Treat as PLACEHOLDER until verified.
    """

    def __init__(self, endpoint: str, deployment: str = "self-hosted"):
        self.endpoint = endpoint  # e.g., "http://localhost:8000" for self-hosted
        self.deployment = deployment
        self.client = self._init_client()

    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        response = self.client.chat.completions.create(
            model="kimi-k2-thinking",
            messages=[{"role": "user", "content": prompt}],
            max_steps=params.get("max_steps", 50),
            thinking_budget_per_step=params.get("thinking_budget_per_step", 12000),
            temperature=params.get("temperature", 0.7),
            stream=False
        )
        return response.choices[0].message.content

    def stream(self, prompt: str, params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        response = self.client.chat.completions.create(
            model="kimi-k2-thinking",
            messages=[{"role": "user", "content": prompt}],
            max_steps=params.get("max_steps", 50),
            thinking_budget_per_step=params.get("thinking_budget_per_step", 12000),
            temperature=params.get("temperature", 0.7),
            stream=True
        )

        for chunk in response:
            if chunk.type == "thinking":
                yield {"type": "thinking", "content": chunk.content}
            elif chunk.type == "content":
                yield {"type": "content", "content": chunk.content}

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supports_thinking": True,  # ✅ VERIFIED: Moonshot AI blog, model design
            "supports_tools": True,     # ⚠️ ASSUMPTION: Inferred from model architecture, needs verification
            "context_window": 256000,   # ✅ VERIFIED: Official Moonshot AI specification
            "thinking_budget_range": (1000, 256000),  # ⚠️ EXTRAPOLATED: Based on context window, not official API spec
            "strengths": [
                "reasoning",           # ✅ VERIFIED: Benchmark results on AIME, MATH, Codeforces
                "long_context"         # ✅ VERIFIED: 256K context window confirmed
                # ⚠️ REMOVED "tool_orchestration", "agentic_workflows" - hypothesized but not tested
            ],
            "weaknesses": []  # ⚠️ TBD: Requires real-world testing and benchmarking
        }

    def get_thinking_budget(self) -> Tuple[int, int, int]:
        # ⚠️ ASSUMPTION: These values are ESTIMATES based on context window size
        # Rationale:
        #   - min (1000): Minimal thinking for simple queries
        #   - max (256000): Full context window (probably too high, tune in production)
        #   - default (96000): ~1/3 of context window for balanced thinking
        # TODO: Test actual thinking budget behavior and calibrate these values
        return (1000, 256000, 96000)  # min, max, default

    def estimate_tokens(self, text: str) -> int:
        # ⚠️ PLACEHOLDER: This is a ROUGH heuristic, not based on actual Kimi K2 tokenizer
        # Formula: word count * 1.3 (assumes ~1.3 tokens per word on average)
        # TODO: Replace with actual tokenizer:
        #   - Use Moonshot AI official tokenizer if available
        #   - Or use tiktoken with appropriate encoding
        #   - Current estimate may be off by 20-50%
        return len(text.split()) * 1.3  # ⚠️ Rough estimate only

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": "Kimi K2 Thinking",
            "version": "1.0",
            "provider": "moonshot",
            "deployment": self.deployment,
            "license": "Modified MIT"
        }
```

#### Example 2: DeepSeek R1 Provider

```python
class DeepSeekR1Provider(LLMProvider):
    """
    ⚠️ IMPORTANT - PLACEHOLDER IMPLEMENTATION:

    DeepSeek R1 specifications are based on ASSUMPTIONS about the model's
    capabilities. Verify the following before deployment:

    UNVERIFIED ASSUMPTIONS:
    - Model exists and is publicly available ⚠️
    - API supports enable_reasoning parameter ⚠️
    - API supports reasoning_budget parameter ⚠️
    - Context window is 128K tokens ⚠️
    - Tool calling is NOT natively supported ⚠️

    TODO BEFORE DEPLOYMENT:
    1. Verify DeepSeek R1 model availability and access
    2. Review official API documentation for actual parameters
    3. Test reasoning capabilities and budget controls
    4. Update implementation based on actual API schema

    This is a HYPOTHETICAL provider implementation based on similar models.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.client = self._init_client()

    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        # ⚠️ ASSUMPTION: DeepSeek R1 has similar API to other reasoning models
        response = self.client.completions.create(
            model="deepseek-r1",
            prompt=prompt,
            max_tokens=params.get("max_tokens", 8192),
            temperature=params.get("temperature", 0.7),
            # ⚠️ ASSUMPTION: These parameters may not exist in actual API
            enable_reasoning=params.get("enable_reasoning", True),
            reasoning_budget=params.get("thinking_budget_per_step", 10000)
        )
        return response.choices[0].text

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supports_thinking": True,  # ⚠️ ASSUMPTION: Based on model name "R1" (reasoning)
            "supports_tools": False,    # ⚠️ ASSUMPTION: Not documented in available info
            "context_window": 128000,   # ⚠️ ASSUMPTION: Common size, needs verification
            "thinking_budget_range": (1000, 128000),  # ⚠️ ASSUMPTION: Extrapolated
            "strengths": ["mathematical_reasoning", "coding"],  # ⚠️ Based on DeepSeek lineage
            "weaknesses": ["tool_orchestration"]  # ⚠️ Inferred from lack of tool support
        }

    # ... other methods
```

#### Example 3: Standard OpenAI-Compatible Provider (Fallback)

```python
class OpenAICompatibleProvider(LLMProvider):
    """
    Generic provider for any OpenAI API-compatible model.
    Works with: Llama, Qwen, Mistral, local vLLM, etc.
    """

    def __init__(self, endpoint: str, model_name: str):
        self.endpoint = endpoint
        self.model_name = model_name
        self.client = OpenAI(base_url=endpoint)

    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=params.get("max_tokens", 4096),
            temperature=params.get("temperature", 0.7)
        )
        return response.choices[0].message.content

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supports_thinking": False,  # Most don't expose thinking trace
            "supports_tools": True,  # OpenAI tool calling format
            "context_window": 8192,  # Default, override per model
            "thinking_budget_range": (0, 0),  # N/A
            "strengths": [],  # Unknown, user-provided
            "weaknesses": []
        }

    def get_thinking_budget(self) -> Tuple[int, int, int]:
        return (0, 0, 0)  # No explicit thinking budget

    # ... other methods
```

---

## 2. Model Router: Task-to-Model Mapping

### 2.1 Routing Strategy

```python
class ModelRouter:
    """
    Routes tasks to optimal model based on:
    - Task characteristics
    - Model capabilities
    - Availability
    - User preferences
    """

    def __init__(self, providers: Dict[str, LLMProvider], routing_policy: str = "capability_match"):
        self.providers = providers  # {"kimi_k2": KimiK2Provider, "deepseek": ..., etc.}
        self.routing_policy = routing_policy
        self.capabilities_cache = {
            name: provider.get_capabilities()
            for name, provider in providers.items()
        }

    def route(self, task_analysis: Dict[str, Any]) -> str:
        """
        Select optimal provider for given task.

        Args:
            task_analysis: {
                "task_type": "reasoning|coding|tool_orchestration|multilingual|...",
                "complexity": 0.0 - 1.0,
                "requires_thinking": bool,
                "requires_tools": bool,
                "context_length": int,
                "language": "en|zh|...",
                "user_preference": "model_name" or None
            }

        Returns:
            Provider name (key in self.providers)
        """

        # User override
        if task_analysis.get("user_preference"):
            pref = task_analysis["user_preference"]
            if pref in self.providers:
                return pref

        # Routing policies
        if self.routing_policy == "capability_match":
            return self._route_by_capability(task_analysis)
        elif self.routing_policy == "round_robin":
            return self._route_round_robin()
        elif self.routing_policy == "primary_with_fallback":
            return self._route_with_fallback(task_analysis)
        else:
            # Default to first available
            return list(self.providers.keys())[0]

    def _route_by_capability(self, task_analysis: Dict[str, Any]) -> str:
        """
        Match task requirements to model strengths.
        """
        task_type = task_analysis.get("task_type", "general")
        requires_thinking = task_analysis.get("requires_thinking", False)
        requires_tools = task_analysis.get("requires_tools", False)
        context_needed = task_analysis.get("context_length", 0)

        # Score each provider
        scores = {}
        for name, caps in self.capabilities_cache.items():
            score = 0

            # Must-have requirements
            if requires_thinking and not caps.get("supports_thinking"):
                score -= 100  # Disqualify
            if requires_tools and not caps.get("supports_tools"):
                score -= 100
            if context_needed > caps.get("context_window", 0):
                score -= 100

            # Strength matching
            if task_type in caps.get("strengths", []):
                score += 10
            if task_type in caps.get("weaknesses", []):
                score -= 5

            # Thinking budget (prefer models with variable budgets for complex tasks)
            if task_analysis.get("complexity", 0) > 0.6:
                min_budget, max_budget, _ = self.providers[name].get_thinking_budget()
                if max_budget > 100000:  # Can handle deep thinking
                    score += 5

            scores[name] = score

        # Return highest scoring provider
        best_provider = max(scores.items(), key=lambda x: x[1])
        return best_provider[0] if best_provider[1] > -50 else list(self.providers.keys())[0]

    def _route_with_fallback(self, task_analysis: Dict[str, Any]) -> str:
        """
        Try primary provider, fallback to secondary if unavailable.
        """
        primary = "kimi_k2"  # Configurable
        secondary = "deepseek"
        tertiary = "openai_compatible"

        for provider_name in [primary, secondary, tertiary]:
            if provider_name in self.providers:
                if self._is_available(provider_name):
                    return provider_name

        # All failed - return first available
        return list(self.providers.keys())[0]

    def _is_available(self, provider_name: str) -> bool:
        """
        Health check: can this provider handle requests?
        """
        try:
            # Simple health check
            self.providers[provider_name].estimate_tokens("test")
            return True
        except Exception:
            return False
```

### 2.2 Routing Decision Matrix

**Task Type → Model Mapping** (example preferences):

⚠️ **IMPORTANT**: This table contains SUGGESTED routing preferences based on model architectures and general knowledge. These have NOT been validated with actual benchmarks. Validate and tune for your specific use case.

| Task Type | Primary | Secondary | Rationale |
|-----------|---------|-----------|-----------|
| **Deep Reasoning** (math, logic, proofs) | Kimi K2, DeepSeek R1 | Qwen 2.5 Math | ✅ Variable thinking budgets (verified capability), proven reasoning on benchmarks |
| **Tool Orchestration** (multi-step workflows) | Kimi K2 | Claude Opus | ⚠️ Native tool calling (assumed for Kimi K2), long context for complex workflows |
| **Code Generation** | DeepSeek Coder, Qwen Coder | Kimi K2 | ✅ Specialized code models, better at language-specific idioms (general knowledge) |
| **Multilingual** (non-English) | Qwen 2.5, DeepSeek | Kimi K2 | ✅ Trained on diverse languages (Qwen technical report, Chinese models) |
| **Long Context** (>128K tokens) | Kimi K2 (256K) | Claude Opus (200K) | ✅ Context window sizes verified from official specs |
| **Fast Simple Queries** | Llama 3 8B, Qwen 2.5 7B | Any fast model | ✅ Smaller models = lower latency (well-established) |
| **General Purpose** | Kimi K2 | DeepSeek R1 | ⚠️ Hypothesized based on balanced capabilities, needs testing |

**Flexibility**: This mapping is **configurable**, not hard-coded. Users can override per deployment.

**Validation TODO**:
- [ ] Benchmark each task type across available models
- [ ] Measure latency, quality, and cost for each routing decision
- [ ] A/B test routing policies to find optimal mappings
- [ ] Update table with empirical evidence

---

## 3. Effort Regulation with Model Abstraction

### 3.1 Effort Parameters Per Model

Different models have different "effort" mechanisms:

| Model | Effort Mechanism | Parameters |
|-------|------------------|------------|
| **Kimi K2** | Thinking budget | `max_steps`, `thinking_budget_per_step` |
| **DeepSeek R1** | Reasoning budget | `reasoning_budget`, `enable_reasoning` |
| **Claude Opus** | Thinking mode | `thinking=true/false`, `extended_thinking=true/false` |
| **Standard Models** | Temperature + tokens | `temperature`, `max_tokens`, `top_p` |

**Abstraction Strategy**: Map effort score (0.0-1.0) to model-specific parameters.

```python
class EffortRegulationOrchestrator:
    def __init__(self, model_router: ModelRouter):
        self.model_router = model_router
        self.effort_strategies = self._load_strategies()

    def regulate_effort(self, task, user, system_state):
        """
        1. Analyze task complexity → effort score (0.0-1.0)
        2. Route to optimal model
        3. Map effort score to model-specific parameters
        """

        # Step 1: Compute effort score (unchanged from v2.0)
        intrinsic_complexity = compute_intrinsic_complexity(task)
        context_factors = analyze_user_context(task, user, system_state)
        final_effort = compute_final_effort_allocation(intrinsic_complexity, context_factors)

        # Step 2: Route to model
        task_analysis = {
            "task_type": classify_task_type(task),
            "complexity": intrinsic_complexity,
            "requires_thinking": final_effort > 0.4,
            "requires_tools": predict_tool_usage(task),
            "context_length": estimate_context_length(task),
            "user_preference": user.preferences.get("preferred_model")
        }
        selected_provider_name = self.model_router.route(task_analysis)
        provider = self.model_router.providers[selected_provider_name]

        # Step 3: Map effort to model-specific params
        model_params = self._map_effort_to_params(
            final_effort,
            provider,
            task_analysis
        )

        return {
            "provider_name": selected_provider_name,
            "provider": provider,
            "model_params": model_params,
            "effort_score": final_effort
        }

    def _map_effort_to_params(
        self,
        effort_score: float,
        provider: LLMProvider,
        task_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert effort score (0.0-1.0) to model-specific parameters.
        """
        capabilities = provider.get_capabilities()
        params = {}

        # Thinking budget models (Kimi K2, DeepSeek R1)
        if capabilities.get("supports_thinking"):
            min_budget, max_budget, default_budget = provider.get_thinking_budget()

            if max_budget > 0:
                # Map effort to thinking budget
                budget = int(min_budget + (max_budget - min_budget) * effort_score)

                # Model-specific parameter names
                metadata = provider.get_model_metadata()
                if metadata["provider"] == "moonshot":
                    # Kimi K2
                    params["max_steps"] = self._effort_to_steps(effort_score)
                    params["thinking_budget_per_step"] = budget // params["max_steps"]
                elif metadata["provider"] == "deepseek":
                    # DeepSeek R1
                    params["reasoning_budget"] = budget
                    params["enable_reasoning"] = True
                elif metadata["provider"] == "anthropic":
                    # Claude
                    # ⚠️ NOTE: As of 2025-11-09, Claude API does not have explicit
                    # "extended_thinking" parameter. This is a hypothetical future API.
                    # Current Claude API uses standard max_tokens and temperature.
                    # TODO: Update when/if Claude adds thinking budget controls
                    params["thinking"] = effort_score > 0.5  # ⚠️ Hypothetical
                    params["extended_thinking"] = effort_score > 0.8  # ⚠️ Hypothetical

        # Temperature (higher effort = higher creativity for complex tasks)
        # ⚠️ DESIGN CHOICE: Linear temperature mapping
        # Rationale: Higher effort tasks benefit from exploration (higher temp)
        # Range [0.3-0.9] avoids extremes (too deterministic vs too random)
        # TODO: A/B test alternatives (exponential, sigmoid, step function)
        params["temperature"] = 0.3 + (effort_score * 0.6)  # Range: 0.3 - 0.9

        # Max tokens
        # ⚠️ HEURISTIC THRESHOLDS: Based on typical response lengths, not empirical
        # TODO: Analyze actual response length distributions and optimize cutoffs
        if effort_score < 0.3:
            params["max_tokens"] = 2048  # Short responses
        elif effort_score < 0.6:
            params["max_tokens"] = 4096  # Medium responses
        else:
            params["max_tokens"] = 8192  # Detailed responses

        return params

    def _effort_to_steps(self, effort_score: float) -> int:
        """
        Map effort score to max reasoning steps.

        ⚠️ ASSUMPTION: Step counts are ESTIMATES based on intuition about reasoning complexity.
        Rationale:
          - Low effort (5-10 steps): Quick factual queries
          - Medium (50 steps): Multi-step reasoning
          - High (120-300 steps): Complex proofs, deep analysis

        TODO: Calibrate with actual model testing:
          1. Run tasks at different step budgets
          2. Measure quality vs steps relationship
          3. Find diminishing returns point
          4. Update thresholds based on empirical data

        Current values are PLACEHOLDERS pending real-world validation.
        """
        if effort_score < 0.2:
            return 5      # ⚠️ Minimal effort
        elif effort_score < 0.4:
            return 10     # ⚠️ Fast
        elif effort_score < 0.6:
            return 50     # ⚠️ Balanced
        elif effort_score < 0.8:
            return 120    # ⚠️ Thorough
        else:
            return 300    # ⚠️ Maximum
```

---

## 4. Deployment Architecture: Bare Metal / On-Premise

### 4.1 Design Principles

**Requirements**:
- Self-hosted (no cloud-managed services like AWS EKS, GCP GKE)
- Bare metal or metal-rent (Vultr, Hetzner, OVH)
- Full control over infrastructure
- Model weights stored locally
- No data leaving premises

**Architecture Options**:

#### Option A: Single-Server Deployment (Small Scale)

```
┌─────────────────────────────────────────────────┐
│        Single Bare Metal Server                 │
│        (e.g., Vultr Bare Metal 4x A100)         │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │  Docker Compose Stack                    │  │
│  │                                           │  │
│  │  - Effort Regulation Orchestrator        │  │
│  │  - Model Inference Services (vLLM)       │  │
│  │    * Kimi K2 (4x A100)                   │  │
│  │    * DeepSeek R1 (if fits)               │  │
│  │  - Neo4j                                 │  │
│  │  - Redis                                 │  │
│  │  - OPA + Cedar                           │  │
│  │  - Nginx (reverse proxy)                 │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**Pros**:
- Simplest setup (Docker Compose)
- All components co-located
- No network latency between services
- Minimal operational overhead

**Cons**:
- Single point of failure
- Limited scaling (vertical only)
- GPU contention if running multiple models

**When to Use**: MVP, small team (<50 users), low request volume

---

#### Option B: Multi-Server Kubernetes Cluster (Medium-Large Scale)

```
┌─────────────────────────────────────────────────────────────┐
│           Kubernetes Cluster (Bare Metal)                    │
│           (k3s or vanilla Kubernetes)                        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Control Plane Nodes (3x for HA)                     │  │
│  │  - No GPUs, CPU-only                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  GPU Worker Nodes                                    │  │
│  │  ┌───────────────────────────────────────────────┐   │  │
│  │  │ Node 1: 4x A100 (Kimi K2)                     │   │  │
│  │  └───────────────────────────────────────────────┘   │  │
│  │  ┌───────────────────────────────────────────────┐   │  │
│  │  │ Node 2: 4x A100 (DeepSeek R1)                 │   │  │
│  │  └───────────────────────────────────────────────┘   │  │
│  │  ┌───────────────────────────────────────────────┐   │  │
│  │  │ Node 3: 2x A100 (Qwen 2.5, smaller models)    │   │  │
│  │  └───────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  CPU Worker Nodes                                    │  │
│  │  - Orchestration services                            │  │
│  │  - Neo4j, Redis, OPA, Cedar                          │  │
│  │  - ROMA, LangGraph                                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Storage Nodes (Ceph / Longhorn / NFS)               │  │
│  │  - Model weights (TB-scale)                          │  │
│  │  - Neo4j data                                        │  │
│  │  - Backups                                           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Pros**:
- Horizontal scaling
- High availability (control plane HA)
- Resource isolation (GPUs per model)
- Rolling updates without downtime

**Cons**:
- Complex to set up and maintain
- Requires Kubernetes expertise
- More hardware needed

**When to Use**: Enterprise, high availability required, >100 users

---

#### Option C: Hybrid Local + Metal-Rent (Flexibility)

```
┌────────────────────────────────┐     ┌──────────────────────────────┐
│  On-Premise (Private Data)     │     │  Vultr Bare Metal            │
│                                │     │  (Burst Capacity)            │
│  - Neo4j (knowledge graph)     │     │                              │
│  - OPA/Cedar (policies)        │     │  - Kimi K2 Inference         │
│  - Critical data processing    │     │  - DeepSeek R1 Inference     │
└────────────────────────────────┘     └──────────────────────────────┘
              │                                    │
              └────────────VPN/Wireguard──────────┘
                        (Encrypted Tunnel)

```

**Pros**:
- Sensitive data stays on-premise
- GPU inference offloaded to metal-rent (cheaper than buying GPUs)
- Elastic: spin up/down Vultr nodes as needed

**Cons**:
- Network latency (on-prem ↔ Vultr)
- Data transfer costs (depending on provider)
- More complex networking

**When to Use**: Data residency requirements + need for GPU flexibility

---

### 4.2 Model Serving: vLLM vs TGI vs Custom

**Problem**: How to serve LLM inference efficiently?

#### Option 1: vLLM (Recommended for Most)

**vLLM** = High-throughput LLM serving with PagedAttention

```yaml
# Kubernetes Deployment for vLLM serving Kimi K2
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-kimi-k2
spec:
  replicas: 1
  template:
    spec:
      nodeSelector:
        gpu: nvidia-a100
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command:
        - python
        - -m
        - vllm.entrypoints.openai.api_server
        - --model
        - /models/kimi-k2-thinking-int4  # Local path
        - --tensor-parallel-size
        - "4"  # Use 4 GPUs
        - --dtype
        - int4  # Quantization
        - --max-model-len
        - "256000"  # Context window
        - --gpu-memory-utilization
        - "0.9"
        volumeMounts:
        - name: models
          mountPath: /models
        resources:
          limits:
            nvidia.com/gpu: 4
      volumes:
      - name: models
        hostPath:
          path: /data/models  # Models stored on node
```

**Pros**:
- Excellent throughput (PagedAttention = efficient KV cache)
- OpenAI API-compatible (works with abstraction layer)
- Supports quantization (INT4, INT8, AWQ, GPTQ)
- Active community

**Cons**:
- Complex configuration for some models
- GPU memory management needs tuning

**When to Use**: Most deployments (production-ready, well-tested)

---

#### Option 2: TGI (Text Generation Inference by HuggingFace)

```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v /data/models:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id /data/kimi-k2-thinking-int4 \
  --num-shard 4 \
  --quantize bitsandbytes-nf4
```

**Pros**:
- Official HuggingFace support
- Good integration with transformers ecosystem
- Simpler than vLLM for some models

**Cons**:
- Lower throughput than vLLM (no PagedAttention)
- Less flexible for custom models

**When to Use**: HuggingFace-centric workflow, rapid prototyping

---

#### Option 3: Ollama (Simplicity for Smaller Models)

```bash
# Start Ollama server
ollama serve

# Pull model
ollama pull kimi-k2-thinking

# API compatible with OpenAI
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "kimi-k2-thinking", "messages": [...]}'
```

**Pros**:
- Dead simple (one binary)
- Good for development/testing
- Growing model library

**Cons**:
- Not optimized for production throughput
- Limited to smaller models (consumer GPUs)

**When to Use**: Development, single-user deployments, demos

---

### 4.3 Storage Architecture

**Requirements**:
- Model weights: 100GB - 1TB per model (Kimi K2 INT4 ~500GB)
- Neo4j data: Growing over time (GB → TB)
- Backups: Retain historical data

#### Storage Options

| Option | Use Case | Pros | Cons |
|--------|----------|------|------|
| **Ceph** | Distributed object/block storage | Highly scalable, HA, self-healing | Complex to set up, resource-heavy |
| **Longhorn** | Kubernetes-native distributed storage | K8s integration, simpler than Ceph | Newer, smaller community |
| **NFS** | Centralized file server | Simple, works everywhere | Single point of failure, not scalable |
| **Local SSD + rsync** | Single-server deployments | Fastest, simplest | No HA, manual backups |

**Recommended**:
- **Small deployments**: Local SSD + rsync to backup server
- **Medium-large**: Longhorn (Kubernetes-native, good balance)
- **Enterprise**: Ceph (if team has expertise)

---

### 4.4 Networking

**Requirements**:
- Low latency GPU ↔ CPU communication
- Secure inter-service communication
- External access (API gateway)

```
┌─────────────────────────────────────────────────┐
│               External Network                  │
│  (Users, External Tools)                        │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   Nginx / Traefik   │  (Reverse Proxy + TLS)
        │   API Gateway       │
        └──────────┬──────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│         Internal K8s Network                    │
│         (10.0.0.0/16, private)                  │
│                                                  │
│  ┌────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ Effort     │  │ Model       │  │ Neo4j    │ │
│  │ Orchestrator│ │ Inference   │  │ Database │ │
│  └────────────┘  └─────────────┘  └──────────┘ │
│                                                  │
│  Service Mesh (Optional: Istio, Linkerd)        │
│  - mTLS between services                        │
│  - Traffic routing                              │
└─────────────────────────────────────────────────┘
```

**Security**:
- **External**: HTTPS/TLS (Let's Encrypt or self-signed)
- **Internal**: mTLS via service mesh (optional but recommended)
- **Secrets**: Kubernetes Secrets (encrypted at rest) or Vault

---

## 5. Model Switching: Hot-Swap Capability

### 5.1 Configuration-Driven Model Registry

```yaml
# config/model_registry.yaml

models:
  kimi_k2:
    provider: moonshot
    deployment: self-hosted
    endpoint: http://vllm-kimi-k2:8000/v1
    capabilities:
      supports_thinking: true
      supports_tools: true
      context_window: 256000
      strengths: [reasoning, tool_orchestration, long_context]
    enabled: true
    priority: 1  # Primary model

  deepseek_r1:
    provider: deepseek
    deployment: self-hosted
    endpoint: http://vllm-deepseek-r1:8000/v1
    capabilities:
      supports_thinking: true
      supports_tools: false
      context_window: 128000
      strengths: [mathematical_reasoning, coding]
    enabled: true
    priority: 2  # Secondary

  qwen_2_5_coder:
    provider: qwen
    deployment: self-hosted
    endpoint: http://vllm-qwen-coder:8000/v1
    capabilities:
      supports_thinking: false
      supports_tools: true
      context_window: 32000
      strengths: [coding, multilingual]
    enabled: true
    priority: 3

  # Future models (disabled until deployed)
  llama_4_reasoning:
    provider: meta
    deployment: not_yet_deployed
    enabled: false

  claude_opus:
    provider: anthropic
    deployment: api
    endpoint: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
    capabilities:
      supports_thinking: true
      supports_tools: true
      context_window: 200000
      strengths: [reasoning, tool_use, writing]
    enabled: false  # Fallback only (costs money)
    priority: 99

routing:
  default_policy: capability_match
  fallback_chain: [kimi_k2, deepseek_r1, qwen_2_5_coder, claude_opus]
  task_routing:
    deep_reasoning: [kimi_k2, deepseek_r1]
    code_generation: [qwen_2_5_coder, deepseek_r1]
    tool_orchestration: [kimi_k2]
    fast_query: [qwen_2_5_coder]
```

### 5.2 Hot Reload Mechanism

```python
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ModelRegistryReloader(FileSystemEventHandler):
    """
    Watch model_registry.yaml and reload on changes (no restart needed).
    """

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.config_path = "config/model_registry.yaml"

    def on_modified(self, event):
        if event.src_path.endswith("model_registry.yaml"):
            logger.info("Model registry changed, reloading...")
            self.orchestrator.reload_models()

    def reload_models(self):
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Rebuild provider registry
        new_providers = {}
        for model_name, model_config in config["models"].items():
            if not model_config.get("enabled", False):
                continue

            provider_class = self._get_provider_class(model_config["provider"])
            new_providers[model_name] = provider_class(
                endpoint=model_config["endpoint"],
                **model_config.get("provider_kwargs", {})
            )

        # Atomic swap
        self.orchestrator.model_router.providers = new_providers
        self.orchestrator.model_router.capabilities_cache = {
            name: provider.get_capabilities()
            for name, provider in new_providers.items()
        }

        logger.info(f"Reloaded {len(new_providers)} models")
```

**Usage**:
1. Edit `model_registry.yaml` (add new model, disable old one, change routing)
2. Save file
3. System auto-reloads within seconds (no restart)

### 5.3 A/B Testing Framework

```python
class ABTestingRouter:
    """
    Route X% of traffic to new model for testing before full migration.
    """

    def __init__(self, model_router: ModelRouter):
        self.model_router = model_router
        self.ab_tests = {}  # {test_name: {model_a, model_b, split_ratio}}

    def add_ab_test(self, test_name: str, model_a: str, model_b: str, split_ratio: float = 0.5):
        """
        Define A/B test: send split_ratio% to model_b, rest to model_a.
        """
        self.ab_tests[test_name] = {
            "model_a": model_a,
            "model_b": model_b,
            "split_ratio": split_ratio
        }

    def route_with_ab_test(self, task_analysis: Dict[str, Any], user_id: str) -> str:
        """
        If user in A/B test, route based on test; else normal routing.
        """
        # Check if any active A/B test applies
        for test_name, test_config in self.ab_tests.items():
            # Hash user_id to deterministic bucket
            hash_val = hash(f"{test_name}:{user_id}") % 100
            if hash_val < (test_config["split_ratio"] * 100):
                # User in B group
                return test_config["model_b"]
            else:
                # User in A group
                return test_config["model_a"]

        # No A/B test, use normal routing
        return self.model_router.route(task_analysis)
```

**Example Usage**:
```python
# Test Kimi K2 vs DeepSeek R1 on 20% of users
ab_router.add_ab_test(
    test_name="kimi_vs_deepseek",
    model_a="kimi_k2",
    model_b="deepseek_r1",
    split_ratio=0.2  # 20% to DeepSeek, 80% to Kimi
)

# After 1 week, analyze quality scores, latency, user feedback
# If DeepSeek better → update default routing
# If Kimi better → remove A/B test
```

---

## 6. Future-Proofing: Design for Unknown Models

### 6.1 Challenge

**Reality**: In 6 months, there will be models that:
- Have capabilities we haven't imagined (multimodal, embodied, etc.)
- Use different inference paradigms (test-time compute, RL-based, etc.)
- Require different hardware (TPUs, neuromorphic chips, etc.)

**Design Goal**: Architecture should accommodate these without major refactor.

### 6.2 Extension Points

```python
# Future extension: Multimodal model provider
class MultimodalLLMProvider(LLMProvider):
    """
    Provider for models that handle text + images + audio.
    """

    def generate_multimodal(
        self,
        prompt: str,
        images: List[bytes] = None,
        audio: bytes = None,
        params: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """
        New method not in base LLMProvider - that's OK.
        Routing layer can detect capability and use it.
        """
        pass

    # Still implements base methods for backwards compatibility
    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        # Text-only fallback
        return self.generate_multimodal(prompt, None, None, params)["text"]
```

### 6.3 Capability Discovery

```python
# Base provider interface is MINIMAL
# Providers can expose additional capabilities via metadata

class FutureLLMProvider(LLMProvider):
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            **super().get_capabilities(),
            # New capabilities
            "supports_multimodal": True,
            "supports_rl_refinement": True,  # Hypothetical: model can self-improve via RL
            "supports_streaming_thinking": True,
            "custom_extensions": {
                "multimodal_generate": True,
                "rl_refine": True
            }
        }
```

**Routing logic**:
```python
# If task requires multimodal
if task_analysis.get("has_images"):
    # Filter providers that support multimodal
    candidates = [
        name for name, caps in self.capabilities_cache.items()
        if caps.get("supports_multimodal", False)
    ]
    # Route to first available
    return candidates[0] if candidates else fallback
```

---

## 7. Alternative Architectures: Pros/Cons

### 7.1 Monolithic vs Microservices

#### Architecture A: Monolithic Orchestrator

```
┌─────────────────────────────────────────┐
│     Single Orchestrator Process         │
│                                         │
│  - Effort Regulation                   │
│  - Model Routing                       │
│  - ROMA Logic                          │
│  - Neo4j Queries                       │
│  - Policy Enforcement                  │
│  - Tool Execution                      │
│                                         │
│  Calls external:                       │
│  - Model inference services            │
│  - Neo4j database                      │
└─────────────────────────────────────────┘
```

**Pros**:
- Simpler deployment (one binary)
- No inter-service network latency
- Easier to reason about

**Cons**:
- Hard to scale individual components
- Single language constraint (e.g., all Python)
- Updates require full restart

**When to Use**: MVP, small scale, rapid iteration

---

#### Architecture B: Microservices

```
┌────────────────┐  ┌───────────────┐  ┌──────────────┐
│ Effort         │  │ Model         │  │ ROMA         │
│ Regulation     │──│ Router        │──│ Meta-Agent   │
│ Service        │  │ Service       │  │ Service      │
└────────────────┘  └───────────────┘  └──────────────┘
       │                   │                  │
       └───────────────────┼──────────────────┘
                           │
              ┌────────────▼────────────┐
              │  API Gateway / Ingress  │
              └─────────────────────────┘
```

**Pros**:
- Independent scaling (scale ROMA separately from effort regulation)
- Polyglot (effort regulation in Rust, ROMA in Python, etc.)
- Independent deployments (update one service without affecting others)

**Cons**:
- Network latency overhead
- More complex monitoring/debugging
- Distributed transaction challenges

**When to Use**: Large scale, team wants language flexibility, need independent scaling

---

### 7.2 Push vs Pull Orchestration

#### Pattern A: Push (Orchestrator Drives)

```python
# Orchestrator pushes work to workers
def execute_task(task):
    effort_config = effort_orchestrator.regulate(task)
    provider = model_router.select(task_analysis)
    result = provider.generate(task, effort_config)
    return result
```

**Pros**:
- Simpler control flow
- Easier to implement
- Clear execution path

**Cons**:
- Orchestrator becomes bottleneck
- Hard to distribute load

---

#### Pattern B: Pull (Workers Pull from Queue)

```python
# Workers pull tasks from queue
class ModelWorker:
    def run(self):
        while True:
            task = task_queue.pop()
            effort_config = effort_orchestrator.regulate(task)
            result = self.provider.generate(task, effort_config)
            result_queue.push(result)
```

**Pros**:
- Natural load balancing (workers pull when available)
- Easy to add more workers
- Resilient (worker crash doesn't block queue)

**Cons**:
- More complex (queue management)
- Harder to debug (async execution)

**Recommendation**: Hybrid - push for simple tasks, pull for long-running

---

### 7.3 Stateless vs Stateful Orchestration

#### Option A: Stateless Orchestrator

```python
# Each request is independent
def handle_request(request):
    # No memory of previous requests
    result = orchestrate(request)
    return result
```

**Pros**:
- Easy to scale horizontally (any instance can handle any request)
- No state synchronization needed
- Simple failover (restart = clean slate)

**Cons**:
- No conversation context (unless stored externally in Neo4j)
- Must reconstruct state from database each request

---

#### Option B: Stateful Orchestrator (Session-Based)

```python
# Maintain session state
class OrchestrationSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.context = load_context(user_id)
        self.conversation_history = []

    def handle_message(self, message):
        # Context persists across messages
        self.conversation_history.append(message)
        result = orchestrate(message, self.context)
        return result
```

**Pros**:
- Faster (no DB lookup each request)
- Richer context (conversation memory)
- Better for multi-turn interactions

**Cons**:
- Harder to scale (sticky sessions needed)
- State loss if process crashes
- Memory overhead

**Recommendation**: Stateless with Neo4j for persistence (best of both)

---

## 8. Decision Framework Summary

### 8.1 Model Selection Decision Tree

```
Task arrives
    │
    ▼
Does task require deep reasoning?
├─ YES ─► Kimi K2 or DeepSeek R1 (variable thinking budget)
└─ NO ──► Is task code-related?
          ├─ YES ─► Qwen Coder or DeepSeek Coder
          └─ NO ──► Is task multilingual?
                    ├─ YES ─► Qwen 2.5 (strong non-English)
                    └─ NO ──► Is latency critical?
                              ├─ YES ─► Smaller model (7B-13B)
                              └─ NO ──► Default: Kimi K2
```

### 8.2 Deployment Decision Matrix

⚠️ **IMPORTANT**: These thresholds are ROUGH GUIDELINES based on typical enterprise deployments, NOT load-tested for this specific architecture. Actual capacity depends on:
- Hardware specs (CPU, GPU, RAM, network)
- Task complexity distribution
- Model inference latency
- Concurrent user behavior

**TODO**: Run load testing to determine actual limits for your deployment.

| Factor | Single Server | K8s Cluster | Hybrid |
|--------|---------------|-------------|--------|
| **Team Size** | <5 | >5 | Any |
| **User Count** | <50 ⚠️ | >100 ⚠️ | 50-100 ⚠️ |
| **Request Volume** | <1K/day ⚠️ | >10K/day ⚠️ | 1K-10K ⚠️ |
| **Availability SLA** | 90% OK | 99.9% required | 99% |
| **Operational Expertise** | Limited | Strong | Medium |
| **Budget** | Minimal | Higher | Flexible |

**Capacity Notes**:
- User count thresholds assume moderate usage (10-50 queries/user/day)
- Request volume depends heavily on avg response time (1s vs 30s makes 30x difference)
- GPU memory limits are hard constraints - test with actual model and workload

### 8.3 Model Serving Decision Matrix

| Factor | vLLM | TGI | Ollama |
|--------|------|-----|--------|
| **Throughput Need** | High | Medium | Low |
| **Model Size** | Any | <70B | <70B |
| **Production** | Yes | Yes | No (dev only) |
| **Ease of Use** | Medium | Medium | Easy |
| **Quantization** | Excellent | Good | Good |

---

## 9. Open Design Questions

Rather than prescribe solutions, here are **choices to make** based on your context:

### 9.1 Architecture Style

**Question**: Monolithic or microservices?
- **Monolithic**: Faster to build, simpler to deploy, good enough for most
- **Microservices**: Better scaling, polyglot, but complex

**Consider**: Team size, operational capability, scale requirements

---

### 9.2 Model Diversity

**Question**: How many models to deploy initially?
- **Option 1: Single model** (e.g., just Kimi K2)
  - Pros: Simplest, least hardware
  - Cons: No fallback, no specialization
- **Option 2: Two models** (e.g., Kimi K2 + DeepSeek R1)
  - Pros: Fallback, A/B testing, specialization
  - Cons: 2x GPU requirements
- **Option 3: Three+ models** (reasoning + coding + fast)
  - Pros: Optimal routing, speed tiers
  - Cons: Complex management

**Recommendation**: Start with one, add second when need is clear

---

### 9.3 Kubernetes Distribution

**Question**: Which K8s flavor for bare metal?
- **k3s**: Lightweight, single binary, great for edge/small clusters
- **Vanilla Kubernetes**: Full-featured, more complex, battle-tested
- **RKE2** (Rancher): Secure, compliance-focused
- **MicroK8s** (Canonical): Snap-based, easy clustering

**Consider**: Team familiarity, security requirements, cluster size

---

### 9.4 Storage Backend

**Question**: Distributed or centralized storage?
- **Distributed** (Ceph, Longhorn): HA, scalable, complex
- **Centralized** (NFS): Simple, fast, single point of failure

**Consider**: Availability requirements, team expertise

---

### 9.5 Effort Allocation Philosophy

**Question**: Default to high or low effort?
- **Conservative** (default minimal, escalate if needed)
  - Lower latency, might need retries
- **Aggressive** (default balanced/thorough)
  - Higher quality first-try, but slower

**Consider**: User tolerance for latency, quality requirements

---

## 10. Conclusion

This v2.1 architecture provides:

✅ **Model Agnostic Design**: LLM abstraction layer allows hot-swapping
✅ **Deployment Flexibility**: Single-server, K8s, or hybrid options
✅ **Theoretical Framework**: Patterns and decision trees, not prescriptive costs
✅ **Future-Proof**: Extension points for capabilities not yet imagined
✅ **Effort Regulation**: Model-specific parameter mapping from unified effort score
✅ **Alternatives Documented**: Pros/cons for each architectural choice

**Key Takeaway**: This is a **framework for decision-making**, not a rigid blueprint. Choose options that fit your:
- Team expertise
- Scale requirements
- Hardware availability
- Operational maturity

---

## 11. References

**Model Abstraction Patterns**:
- OpenAI API Standard: https://platform.openai.com/docs/api-reference
- vLLM Documentation: https://docs.vllm.ai/
- HuggingFace TGI: https://huggingface.co/docs/text-generation-inference

**Deployment Technologies**:
- k3s: https://k3s.io/
- Longhorn: https://longhorn.io/
- Vultr Bare Metal: https://www.vultr.com/products/bare-metal/
- Hetzner Dedicated: https://www.hetzner.com/dedicated-rootserver

**Model Serving**:
- vLLM: https://github.com/vllm-project/vllm
- TGI: https://github.com/huggingface/text-generation-inference
- Ollama: https://ollama.ai/

**Storage**:
- Ceph: https://ceph.io/
- Longhorn: https://longhorn.io/

---

**Document Version**: 2.1
**Author**: Claude (Autonomous AI Architect)
**Date**: 2025-11-08
**Status**: Theoretical Framework - deployment decisions left to user
**Supersedes**: v2.0 (removed cost/time analysis, added model abstraction)
