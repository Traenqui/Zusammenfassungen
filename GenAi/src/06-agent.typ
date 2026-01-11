
= Agents

An agent is a system where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks. Need Tools to execute, memory to remember, and reasoning ability to plan.

== Tools
- Pure LLM is stateless, text in → text out
- Tools give ability to interact with external environment or extract data and use external applications

== M×N Problem
- With M LLM-Models and N Tools, integration is $M times N$
- *MCP reduces to* $M + N$ by providing standard interface
- Each AI-Application implements client side of MCP once
- Each tool/data implements server side once
- Standardized interface prevents repeated custom integrations
- *MCP:* open protocol enabling 2-way communication between LLM apps and tool servers

== MCP Flow
*MCP Server:* Provides context, tools and capabilities to LLM\
*MCP Host:* LLM application that manages connection\
*MCP Client:* Maintains 1-to-1 connection with MCP Server\
*Resources:* The tools, data, or services provided locally or remotely

== Transport
*Streamable HTTP* is default transport for remote MCP servers
- Supported out-of-the-box by official MCP SDKs
- MCP servers run as independent processes and handle multiple clients
- Communication uses HTTP methods (GET, PUT)
- Optional Server-Sent Events (SSE) enable streaming responses

#table(
  columns: (1fr, 1.5fr, 1.5fr, 1.5fr),
  table.header([*Transport*], [*Pros*], [*Cons*], [*When to Use*]),
  [gRPC], [Fast, streaming, strongly typed, low latency], [Higher setup, requires Protobuf], [Internal MCP, high-performance multi-client],
  [HTTP/HTTPS], [Universal, easy to debug, browser-friendly], [Higher latency, no streaming by default], [Exposing MCP tools to external clients, web APIs],
  [stdio], [Minimal setup], [Local only, single client], [Teaching, local experiments],
)

== Small Models & Tool RAG
- Question: can small language models emulate function-calling well?
- ToolRAG ("RAG for tools") can miss auxiliary tools if embeddings don't match query
- Example: scheduling may also require `get_email_address`
- Fix idea: tool selection as *classification* (which tools needed), not just similarity search
