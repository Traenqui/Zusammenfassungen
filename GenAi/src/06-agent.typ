#import "../../template_zusammenf.typ": *

= Agents

An _agent_ is a system where an LLM can *plan*, *choose actions*, call *tools*, observe results, and iterate (often with memory/state), instead of being only “text in → text out”.

== Tools
- Pure LLM: stateless; produces text based on prompt/context.
- *Tools* let the system interact with the outside world:
  - query databases / search indexes
  - call APIs (calendar, email, CRMs)
  - run code, calculators, retrieval, file operations
- Typical agent loop (conceptual):
  - _Plan_ -> _Select tool_ -> _Call tool_ -> _Observe_ -> repeat -> _Answer_

== M×N Integration Problem
- With *M* LLM apps and *N* tool providers, custom integrations scale as $M times N$.
- Goal: reduce integration effort to $M + N$ via a shared protocol/interface.

== MCP (Model Context Protocol) — Core Idea
*MCP* provides a standardized interface for connecting LLM applications to tool/data servers:
- Each AI application implements the *MCP client* once.
- Each tool/data provider exposes an *MCP server* once.
- Result: fewer bespoke adapters, easier reuse across apps and tools.

=== Roles (typical terminology)
- *Host*: the LLM application (or agent runtime) that wants to use tools/resources.
- *Client*: the connector inside the host that speaks MCP.
- *Server*: exposes capabilities (tools, resources, prompts) via MCP.

== MCP Flow (high-level)
1. Host connects to MCP server
2. Server advertises available capabilities (e.g., tools/resources)
3. LLM selects a tool based on the user task
4. Host invokes tool call via MCP
5. Server executes tool and returns structured results
6. Host uses the result to continue reasoning / produce final answer

#hinweis[
Key agent takeaway: MCP standardizes *how* tools are described and invoked; the LLM still decides *when* and *which* tools to use.
]

== Transport Options (MCP)
#table(
  columns: (1fr, 1.6fr, 1.6fr, 1.6fr),
  table.header([*Transport*], [*Pros*], [*Cons*], [*When to Use*]),

  [gRPC],
  [Fast, streaming-friendly, strongly typed, low latency],
  [More setup; Protobuf tooling; less convenient for browsers],
  [Internal systems, high-performance multi-client deployments],

  [HTTP/HTTPS (Streamable HTTP)],
  [Universal, easy to debug, works well across networks],
  [More overhead than gRPC; streaming needs conventions (e.g., SSE)],
  [Exposing tools across services, web-friendly deployments],

  [stdio],
  [Minimal setup; great for local processes],
  [Local only; typically single-client; process lifecycle handling],
  [Teaching, prototypes, local tool servers],
)

#hinweis[
For HTTP-based transport, typical pattern is client -> server via POST and server -> client streams/notifications via GET (often with SSE for streaming).
]

== Small Models & ToolRAG
- Question: can smaller LMs emulate function-calling / tool-use reliably?
- *ToolRAG* (“RAG for tools”): retrieve relevant tools by embedding similarity.
- Failure mode: auxiliary tools are missed because the query does not semantically match them.
  - Example: “schedule meeting with Alex” may also require `get_email_address`.
- Fix idea: treat tool selection as *classification* (which tools are required), not only similarity search.
  - Use tool schemas, dependencies, and multi-step planning to include required helper tools.
