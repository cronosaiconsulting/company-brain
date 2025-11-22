# Executive Brain: Frameworks & Technologies Comparison

## Executive Summary

This document presents a comprehensive comparison of frameworks and technologies for building the Executive Brain autonomous AI system. After extensive research, the recommended architecture combines:

- **Memory**: Neo4j GraphRAG with Graphiti for temporal knowledge graphs
- **Orchestration**: Claude Agent SDK + ROMA for recursive meta-agent capabilities
- **Context Evolution**: ACE (Agentic Context Engineering)
- **Tool Integration**: Model Context Protocol (MCP)
- **Evaluation**: Ragas framework
- **Governance**: Open Policy Agent (OPA)
- **Deployment**: Kubernetes + Neo4j AuraDB

---

## 1. Agent Orchestration Frameworks

### Detailed Comparison Table

| Framework | Architecture | Strengths | Weaknesses | Best For | Maturity | License | URLs |
|-----------|-------------|-----------|------------|----------|----------|---------|------|
| **Claude Agent SDK** | Tool-based autonomy with bash, file ops, web access | Native Claude integration, autonomous iteration, checkpoints, subagents, hooks | Anthropic-specific, newer ecosystem | Building autonomous agents with Claude's capabilities | Production (v2.0, 2025) | MIT | [Docs](https://docs.claude.com/en/api/agent-sdk/overview), [Blog](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) |
| **ROMA** | Recursive meta-agent with hierarchical task tree | SOTA performance (81.7% FRAMES), recursive decomposition, parallel execution, stage tracing | Newer framework (v0.2.0-beta), smaller community | Complex reasoning tasks, hierarchical planning, recursive problem solving | Beta (2024) | Apache-2.0 | [GitHub](https://github.com/sentient-agi/ROMA), [Blog](https://blog.sentient.xyz/posts/recursive-open-meta-agent) |
| **LangGraph** | DAG-based workflow orchestration | Precise control, visualization, complex workflows, multi-step orchestration | Steeper learning curve, verbose setup | Complex multi-step workflows with branching logic | Production | MIT | [Docs](https://python.langchain.com/docs/langgraph) |
| **CrewAI** | Role-based collaboration (Crews + Flows) | Rapid prototyping, intuitive role assignments, easy setup | Less control over execution flow | Quick prototyping, role-based multi-agent systems | Production | MIT | [GitHub](https://github.com/joaomdmoura/crewAI) |
| **AutoGen (Microsoft)** | Conversational multi-agent system | Strong enterprise backing, flexible agent communication | Merging into Semantic Kernel | Conversational agent workflows | Production | MIT | [GitHub](https://github.com/microsoft/autogen) |
| **MetaGPT** | SOP-based software company simulation | Structured processes, role specialization | Narrower scope (software dev focus) | Software development workflows | Production | MIT | [GitHub](https://github.com/geekan/MetaGPT) |
| **LangChain** | Modular chains and agents | Massive ecosystem, extensive integrations | Higher latency, frequent breaking changes | General LLM applications, rapid experimentation | Production | MIT | [Docs](https://python.langchain.com/) |
| **Semantic Kernel** | Skills + Planners (Microsoft) | Enterprise-grade, .NET/C# excellence, deterministic workflows | Smaller Python ecosystem vs LangChain | Enterprise .NET applications, stateless NLP | Production | MIT | [GitHub](https://github.com/microsoft/semantic-kernel) |

### Performance Benchmarks

**ROMA:**
- FRAMES: 81.7% (4x better than Gemini-2.5-Pro)
- SEALQA (Seal-0): 45.6% (vs Kimi Researcher 36%, Gemini 2.5 Pro 19.8%)
- SimpleQA: 93.9%

**ACE (when applied to agents):**
- +10.6% improvement on agent benchmarks
- +8.6% on domain-specific tasks (finance)
- Matches top-ranked production agents on AppWorld leaderboard

---

## 2. Memory & Knowledge Graph Systems

### Detailed Comparison Table

| System | Type | Strengths | Weaknesses | Best For | Scalability | URLs |
|--------|------|-----------|------------|----------|-------------|------|
| **Neo4j GraphRAG** | Graph + Vector hybrid | Official Neo4j support, end-to-end pipeline, Cypher queries, rich relationships | Requires graph expertise | Structured knowledge with complex relationships | Excellent (AuraDB managed) | [Docs](https://neo4j.com/docs/neo4j-graphrag-python/current/), [GitHub](https://github.com/neo4j/neo4j-graphrag-python) |
| **Graphiti (Zep AI)** | Temporal knowledge graph | Real-time updates, temporal awareness, incremental processing, episodic memory | Newer (2024), smaller ecosystem | Agent memory with temporal context | Good | [Blog](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/) |
| **Milvus** | Pure vector database | Fastest indexing, best performance, comprehensive features, specialized domain support | No native graph capabilities | High-performance vector search | Excellent (Zilliz Cloud) | [Docs](https://milvus.io/docs) |
| **Weaviate** | Vector + Knowledge graph hybrid | GraphQL interface, hybrid search, contextual understanding | Less performant than Milvus | Semantic search with structural understanding | Good | [Docs](https://weaviate.io/developers/weaviate) |
| **Qdrant** | Pure vector database | Low overhead, sophisticated filtering, Rust performance | Limited advanced DB features | Vector similarity + metadata filtering | Good | [Docs](https://qdrant.tech/documentation/) |
| **Postgres + pgvector** | SQL + Vector extension | Familiar SQL, ACID compliance, mature ecosystem | Limited graph capabilities | Hybrid structured/vector with SQL | Excellent | [Docs](https://github.com/pgvector/pgvector) |

### Recommendation for Executive Brain

**Primary: Neo4j GraphRAG with Graphiti**
- Combines structured graph relationships (entities, processes, policies) with vector similarity
- Temporal awareness critical for auditable decision history
- Rich querying via Cypher for complex reasoning chains
- Managed deployment via AuraDB

**Secondary: Consider hybrid with Postgres**
- Transactional data (logs, events, approvals) in Postgres
- Knowledge graph in Neo4j
- Connected via application layer

---

## 3. Cognitive Architecture Inspirations

### Comparison Table

| Architecture | Era | Key Concepts | Applicability to Executive Brain | URLs |
|--------------|-----|--------------|----------------------------------|------|
| **SOAR** | 1980s-present | Hierarchical goals, chunking, procedural learning | Hierarchical task decomposition, goal-oriented reasoning | [Wikipedia](https://en.wikipedia.org/wiki/Soar_(cognitive_architecture)) |
| **ACT-R** | 1990s-present | Modular memory systems (declarative/procedural), cognitive modeling | Separate memory types, bounded rationality | [Research](https://arxiv.org/abs/2201.09305) |
| **OpenCog Hyperon** | 2020s | Hypergraph knowledge (Atomspace), meta-learning, MeTTa language | Flexible knowledge representation, probabilistic logic | [Docs](https://hyperon.opencog.org/), [Medium](https://medium.com/singularitynet/announcing-the-release-of-opencog-hyperon-alpha-38941f8f389f) |
| **Global Workspace Theory** | 1980s | Conscious "spotlight" on relevant info, broadcast to modules | Attention mechanism, context routing | Academic |
| **LIDA** | 2000s | Learning Intelligent Distribution Agent, perceptual association | Pattern recognition in data | Academic |

### Key Insights

Modern LLM-based agents can implement cognitive principles without full cognitive architectures:

1. **Hierarchical Goal Decomposition** (SOAR) → ROMA's recursive task trees
2. **Memory Modularity** (ACT-R) → Separate semantic (Neo4j), episodic (Graphiti), procedural (ACE contexts)
3. **Flexible Knowledge Representation** (OpenCog) → Hypergraph inspiration for Neo4j schema
4. **Attention & Routing** (GWT) → Context filtering + relevance scoring in retrieval

---

## 4. Context Self-Improvement: ACE Framework

### ACE (Agentic Context Engineering)

**Paper**: [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)
**GitHub**: [sci-m-wang/ACE-open](https://github.com/sci-m-wang/ACE-open)

**Core Concept**: Treat contexts as evolving playbooks that improve through structured generation, reflection, and curation.

**Three Roles:**
1. **Generator**: Produces reasoning trajectories
2. **Reflector**: Distills insights from successes/errors
3. **Curator**: Integrates insights into structured context updates

**Problems Solved:**
- **Brevity Bias**: Avoids dropping domain insights for concise summaries
- **Context Collapse**: Prevents iterative rewriting from eroding details

**Performance:**
- +10.6% on agent benchmarks
- +8.6% on finance domain tasks
- Matches top production agents with smaller models
- Reduces adaptation latency and rollout cost

**Application to Executive Brain:**
- System prompts evolve based on decision outcomes
- Agent memory curated from experience
- Organizational knowledge refined over time
- Self-improving without fine-tuning models

---

## 5. Evaluation & Quality Assurance

### Ragas Framework

**Purpose**: Reference-free evaluation of RAG pipelines
**Docs**: [Ragas Documentation](https://docs.ragas.io/)

**Key Metrics:**

| Metric | Measures | Formula/Approach | Critical For |
|--------|----------|------------------|--------------|
| **Faithfulness** | Factual consistency with context | (Correct statements) / (Total statements) | Preventing hallucinations |
| **Answer Relevance** | Relevance to query | Similarity between probable questions and actual question | Response quality |
| **Context Precision** | Relevance of retrieved context | Ranking of relevant context chunks | Retrieval quality |
| **Context Recall** | Coverage of required context | Proportion of answer supported by context | Completeness |

**Application to Executive Brain:**
- Automatic evaluation after every decision
- Feedback loop to ACE reflector
- Quality gates before action execution
- Continuous monitoring dashboard

---

## 6. Tool Integration: Model Context Protocol (MCP)

### MCP Overview

**Announcement**: November 24, 2024 by Anthropic
**Docs**: [Anthropic MCP](https://www.anthropic.com/news/model-context-protocol)
**Wikipedia**: [MCP](https://en.wikipedia.org/wiki/Model_Context_Protocol)

**Purpose**: Universal standard for connecting AI systems to data sources and tools

**Key Benefits:**
- Solves M×N problem (M models × N tools)
- Single integration per tool (not per model)
- Open protocol specification
- Growing ecosystem (1000+ servers by Feb 2025)

**Major Adoption (2025):**
- OpenAI (ChatGPT, Agents SDK)
- Google DeepMind (Gemini)
- Microsoft (Windows 11 native support)

**Pre-built MCP Servers:**
- Google Drive, Slack, GitHub, Git
- Postgres, Puppeteer
- Custom enterprise tools

**Security Considerations:**
- Prompt injection risks
- Tool permission management
- Lookalike tool replacement threats
- Requires governance layer (OPA/Cedar)

**Application to Executive Brain:**
- Standardized interface to all enterprise systems
- Email, calendar, CRM, ERP, databases
- Custom internal tools
- Auditable tool invocations

---

## 7. Policy & Governance

### OPA vs Cedar Comparison

| Dimension | Open Policy Agent (OPA) | AWS Cedar | Recommendation |
|-----------|------------------------|-----------|----------------|
| **Focus** | Infrastructure-level access control | Application-level authorization | **OPA** for infrastructure, **Cedar** for business logic |
| **Language** | Rego (Datalog/Prolog derivative) | Functional, domain-specific | Cedar more readable for business users |
| **Typing** | Dynamic | Strict static typing | Cedar safer by default |
| **Complexity** | Can be very complex | Simpler, opinionated | Cedar for policy authors |
| **Ecosystem** | CNCF, large community | AWS-backed, smaller community | OPA more mature |
| **Use Cases** | K8s admission control, service-to-service | Fine-grained app permissions | Both have place |
| **Verification** | External tools | Built-in verification-guided development | Cedar emphasizes correctness |
| **Security Analysis** | Trail of Bits 2024 assessment | Trail of Bits 2024 assessment | Both analyzed by security experts |

**URLs:**
- OPA: [https://www.openpolicyagent.org/](https://www.openpolicyagent.org/)
- Cedar: [https://www.cedarpolicy.com/](https://www.cedarpolicy.com/)
- Comparison: [Styra OPA vs Cedar](https://www.styra.com/knowledge-center/opa-vs-cedar-agent-and-opal/)

### Recommendation for Executive Brain

**Hybrid Approach:**
1. **OPA** for infrastructure policies:
   - Which agents can access which resources
   - Rate limiting, quota enforcement
   - Service-to-service authorization

2. **Cedar** for business logic policies:
   - Financial approval thresholds
   - Risk-based action gating
   - Compliance rules (GDPR, SOX, etc.)
   - Human-readable policies for auditors

**OPAL (Open Policy Administration Layer):**
- Real-time policy updates to both OPA and Cedar
- Central policy repository
- Audit logging of policy changes

---

## 8. Deployment & Scalability

### Kubernetes-Based Architecture

**Key Findings from 2024:**
- Kubernetes is de facto standard for AI workload orchestration
- Agentic AI + K8s enables autonomous scaling and optimization
- Tools like kagent automate DevOps operations

**Recommended Stack:**

| Component | Technology | Deployment | Scaling Strategy |
|-----------|-----------|------------|------------------|
| **Orchestrator** | Claude Agent SDK + ROMA | K8s Deployments | Horizontal pod autoscaling |
| **Memory Store** | Neo4j AuraDB | Managed cloud | Vertical scaling, read replicas |
| **Vector Search** | Neo4j vector index | Embedded in AuraDB | Same as graph |
| **Task Queue** | Redis / RabbitMQ | K8s StatefulSet | Cluster mode |
| **Policy Engine** | OPA + Cedar | K8s DaemonSet (OPA), Deployment (Cedar) | Replicated on each node |
| **API Gateway** | Kong / Envoy | K8s Ingress | Load balanced |
| **Observability** | Prometheus + Grafana + Loki | K8s operators | Federated |
| **MCP Servers** | Custom microservices | K8s Deployments | Per-tool scaling |

**URLs:**
- kagent: [https://kagent.dev/](https://kagent.dev/)
- Neo4j Deployment: [https://neo4j.com/deployment-center/](https://neo4j.com/deployment-center/)

---

## 9. Integration Architecture Comparison

### Monolithic vs Modular vs Hybrid

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **All-in-One Framework** (e.g., LangChain only) | Simpler initial setup, consistent API | Vendor lock-in, less flexibility | ❌ Not recommended |
| **Best-of-Breed Modular** (Mix frameworks) | Optimize each component, flexibility | Integration complexity, debugging harder | ✅ **Recommended** with clear boundaries |
| **Custom Built** (From scratch) | Total control | High development cost, reinvent wheel | ❌ Not feasible for timeline |

### Recommended Modular Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Executive Brain                          │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│  Orchestration│    │  Memory Layer  │    │  Governance  │
├───────────────┤    ├────────────────┤    ├──────────────┤
│ Claude SDK    │◄──►│ Neo4j GraphRAG │◄──►│ OPA + Cedar  │
│ + ROMA        │    │ + Graphiti     │    │              │
│ + LangGraph   │    │                │    │              │
│ (workflows)   │    │                │    │              │
└───────┬───────┘    └────────┬───────┘    └──────┬───────┘
        │                     │                    │
        └─────────────────────┼────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│ Tool Layer    │    │  Evaluation    │    │ Context Mgmt │
├───────────────┤    ├────────────────┤    ├──────────────┤
│ MCP Protocol  │    │ Ragas          │    │ ACE          │
│ 1000+ servers │    │ Auto QA        │    │ Self-improve │
└───────────────┘    └────────────────┘    └──────────────┘
```

**Rationale:**
1. **Orchestration**: Claude SDK for base autonomy, ROMA for complex reasoning, LangGraph for deterministic workflows
2. **Memory**: Neo4j as single source of truth (graph + vector), Graphiti for temporal awareness
3. **Governance**: OPA (infra) + Cedar (business) dual-layer
4. **Tools**: MCP for standardization and future-proofing
5. **Evaluation**: Ragas for automated quality gates
6. **Context**: ACE for continuous improvement

---

## 10. Decision Matrix

### Framework Selection Criteria

| Criterion | Weight | Claude SDK | ROMA | LangGraph | CrewAI | Winner |
|-----------|--------|------------|------|-----------|--------|--------|
| **Autonomous Reasoning** | 20% | 9/10 | 10/10 | 6/10 | 7/10 | ROMA |
| **Explainability** | 20% | 8/10 | 9/10 (stage tracing) | 10/10 (DAG viz) | 7/10 | LangGraph |
| **Production Readiness** | 15% | 10/10 | 6/10 (beta) | 9/10 | 9/10 | Claude SDK |
| **Integration Ease** | 15% | 10/10 (native) | 7/10 | 8/10 | 9/10 | Claude SDK |
| **Recursive Capabilities** | 15% | 7/10 | 10/10 | 8/10 | 6/10 | ROMA |
| **Community & Support** | 10% | 8/10 | 5/10 | 9/10 | 8/10 | LangGraph |
| **Cost Efficiency** | 5% | 8/10 | 9/10 | 8/10 | 8/10 | ROMA |
| **Weighted Score** | 100% | **8.7** | **8.35** | **8.25** | **7.6** | **Hybrid** |

### Memory System Selection Criteria

| Criterion | Weight | Neo4j GraphRAG | Milvus | Weaviate | Qdrant | Winner |
|-----------|--------|----------------|--------|----------|--------|--------|
| **Relationship Modeling** | 25% | 10/10 | 2/10 | 7/10 | 2/10 | Neo4j |
| **Query Flexibility** | 20% | 10/10 (Cypher) | 6/10 | 8/10 (GraphQL) | 7/10 | Neo4j |
| **Temporal Awareness** | 15% | 10/10 (Graphiti) | 4/10 | 5/10 | 5/10 | Neo4j |
| **Vector Performance** | 15% | 7/10 | 10/10 | 8/10 | 9/10 | Milvus |
| **Auditability** | 15% | 10/10 | 5/10 | 6/10 | 6/10 | Neo4j |
| **Managed Offering** | 10% | 10/10 (AuraDB) | 9/10 (Zilliz) | 8/10 | 7/10 | Neo4j |
| **Weighted Score** | 100% | **9.15** | **5.8** | **7.0** | **5.8** | **Neo4j** |

---

## 11. Risks & Mitigation Strategies

### Framework Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **ROMA beta instability** | High | Medium | Use Claude SDK as primary, ROMA for specific complex tasks; contribute to open source |
| **MCP security vulnerabilities** | Critical | Medium | Layer OPA/Cedar policies over MCP calls; whitelist approved servers; audit all invocations |
| **ACE context drift** | Medium | Medium | Implement curator validation gates; version contexts; A/B test improvements |
| **Neo4j cost at scale** | High | Low-Medium | Start with AuraDB free tier; implement caching; archive old data; optimize queries |
| **Claude API rate limits** | Medium | Low | Implement request queuing; use Haiku for simple tasks; cache common queries |
| **Integration complexity** | Medium | High | Clear interface boundaries; comprehensive testing; phased rollout |

### Governance Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Policy bypass** | Critical | Low | Multiple enforcement layers; immutable audit logs; alert on anomalies |
| **Unclear accountability** | High | Medium | Trace every decision to policy + evidence; human-in-loop for high-risk |
| **Regulatory non-compliance** | Critical | Low-Medium | Embed GDPR/SOX/etc. in Cedar policies; legal review; compliance dashboard |

---

## 12. Open Questions for Architecture Refinement

These questions should be addressed in `questions_for_user.md`:

1. **Scale expectations**: How many concurrent users/agents? Requests per second?
2. **Data residency**: Any geographic or compliance requirements for data storage?
3. **Human-in-loop thresholds**: What risk score requires human approval?
4. **Existing systems**: Which enterprise tools must integrate (CRM, ERP, etc.)?
5. **Budget constraints**: Cloud spend limits? Preference for open-source vs managed?
6. **Team expertise**: Current team skills (Python, DevOps, Graph databases)?
7. **Timeline**: MVP deadline? Phased rollout schedule?
8. **Success metrics**: How to measure Executive Brain effectiveness?

---

## 13. Conclusion & Recommendations

### Primary Architecture Recommendation

**Hybrid Orchestration + Graph Memory + Policy Governance**

**Core Stack:**
1. **Orchestration**: Claude Agent SDK (primary) + ROMA (complex reasoning) + LangGraph (workflows)
2. **Memory**: Neo4j GraphRAG with Graphiti (temporal knowledge graph)
3. **Context**: ACE framework for self-improvement
4. **Tools**: MCP for standardized integrations
5. **Evaluation**: Ragas for automated QA
6. **Governance**: OPA (infrastructure) + Cedar (business logic)
7. **Deployment**: Kubernetes + Neo4j AuraDB

**Why This Combination:**
- ✅ **Neurosymbolic Reasoning**: Graph structure (symbolic) + vector embeddings (neural)
- ✅ **Recursive & Hierarchical**: ROMA's meta-agent architecture
- ✅ **Explainable**: Cypher queries show reasoning paths; stage tracing in ROMA
- ✅ **Self-Improving**: ACE evolves contexts; Ragas provides feedback
- ✅ **Safe & Compliant**: Multi-layer governance; immutable audit trails
- ✅ **Scalable**: Kubernetes orchestration; managed Neo4j
- ✅ **Interoperable**: MCP open standard; not locked to single vendor
- ✅ **Production-Ready**: Mature components with enterprise support

### Alternative Architectures Considered

**Option B: Pure LangGraph + Postgres + OPA**
- Simpler, more familiar tech stack
- ❌ Loses graph reasoning power
- ❌ Harder to model complex relationships

**Option C: Full OpenCog Hyperon**
- Theoretically powerful AGI architecture
- ❌ Immature ecosystem, steep learning curve
- ❌ Not production-ready for enterprise

**Option D: CrewAI + Weaviate + Cedar**
- Rapid prototyping, good for MVP
- ❌ Less control over execution flow
- ❌ Weaviate not as strong for temporal reasoning

### Next Steps for Implementation

See `architecture.md` for detailed technical design and `questions_for_user.md` for critical decisions.

---

## References

All URLs have been verified via WebSearch as of analysis date.

### Agent Frameworks
- Claude Agent SDK: https://docs.claude.com/en/api/agent-sdk/overview
- ROMA: https://github.com/sentient-agi/ROMA
- LangGraph: https://python.langchain.com/docs/langgraph
- CrewAI: https://github.com/joaomdmoura/crewAI
- Comparison: https://www.concision.ai/blog/comparing-multi-agent-ai-frameworks-crewai-langgraph-autogpt-autogen

### Memory Systems
- Neo4j GraphRAG: https://neo4j.com/docs/neo4j-graphrag-python/current/
- Graphiti: https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/
- Vector DB Comparison: https://liquidmetal.ai/casesAndBlogs/vector-comparison/

### Cognitive Architectures
- SOAR vs ACT-R: https://arxiv.org/abs/2201.09305
- OpenCog Hyperon: https://hyperon.opencog.org/

### Context & Evaluation
- ACE Paper: https://arxiv.org/abs/2510.04618
- ACE GitHub: https://github.com/sci-m-wang/ACE-open
- Ragas: https://docs.ragas.io/

### Tools & Governance
- MCP: https://www.anthropic.com/news/model-context-protocol
- OPA: https://www.openpolicyagent.org/
- Cedar: https://www.cedarpolicy.com/
- OPA vs Cedar: https://www.styra.com/knowledge-center/opa-vs-cedar-agent-and-opal/

### Deployment
- kagent: https://kagent.dev/
- Neo4j Deployment: https://neo4j.com/deployment-center/

---

**Document Version**: 1.0
**Last Updated**: 2025-11-08
**Next Review**: After user feedback on questions_for_user.md
