# Telegram-RAG-Bot

<h1>Automated Customer Service System Architecture</h1>
<p>This document outlines the theoretical justification, component selection, and structural synthesis of the automated customer service system. The system integrates a Telegram interface, a Large Language Model (LLM), and a hybrid database architecture using MongoDB and ChromaDB.</p>


<img width="841" height="401" alt="image" src="https://github.com/user-attachments/assets/fc71f2a5-aba8-443d-8836-abe11520dbb3" />


<h2>1. Theoretical Basis and Component Selection</h2>
<p>The system architecture is driven by the need to balance stochastic generation (AI) with deterministic business logic, ensuring high reliability for critical operations like order processing while maintaining flexibility for general inquiries.</p>

<h3>1.1. Large Language Model (LLM) Selection (Outdated. Need to review current benchmarks and models)</h3>
<p>The core generative engine selected for this system is <strong>Gemini 2.5</strong> (Pro/Flash family). The selection process evaluated GPT-5, Gemini 2.5, and LLaMA 4 based on performance benchmarks and business constraints.</p>

<ul>
  <li><strong>Performance Metrics:</strong> Benchmarks such as MMLU (General Knowledge), GSM8K (Math), and various coding evaluations were used. GPT-5 showed marginally higher accuracy (>88.7% on MMLU), while Gemini 2.5 offered comparable performance (~88.6%) suitable for the target domain.</li>
  <li><strong>Context Window:</strong> Gemini 2.5 supports ~1 million tokens, suitable for large technical documentation and price lists.</li>
  <li><strong>Cost-Efficiency:</strong> Gemini offers flexibility between "Pro" (high reasoning) and "Flash" (low latency/cost) models.</li>
</ul>

<h3>1.2. Database Architecture</h3>
<p>The system utilizes a Hybrid Storage Architecture to handle semi-structured corporate data (product cards, warranty terms, logs).</p>

<h4>Primary Data Store (MongoDB)</h4>
<p>Selected over Relational (SQL) and Column-family stores due to its document-oriented nature. MongoDB allows flexible schema evolution without expensive table migrations. Benchmarks indicate superior throughput for mixed read/write workloads.</p>

<h4>Vector Store (ChromaDB)</h4>
<p>ChromaDB supports semantic search by storing vector embeddings of document fragments. It integrates seamlessly with LLM frameworks for Retrieval-Augmented Generation (RAG) tasks.</p>

<p><strong>Data Flow:</strong> MongoDB is the "source of truth," while ChromaDB functions as an auxiliary index, synchronized to reflect the current document state.</p>

<h3>1.3. Retrieval-Augmented Generation (RAG) Strategy</h3>
<p>The system implements a Hybrid RAG Pipeline to reduce hallucinations and provide up-to-date information:</p>

<ul>
  <li><strong>Vector Retrieval:</strong> Semantic search for unstructured queries (e.g., "how to use a warranty").</li>
  <li><strong>Text-to-NoSQL:</strong> LLM generates structured database queries for queries like "price of item X" or "order status."</li>
  <li><strong>Routing:</strong> A semantic router determines whether to trigger vector search, direct DB query, or pure LLM generation.</li>
</ul>

<h2>2. System Synthesis and Mathematical Modeling</h2>
<p>The system is modeled as a network of mass service systems (Queuing Theory) to predict performance and optimize resources.</p>

<h3>2.1. Information Flow and Logic Control</h3>
<p>Data processing branches based on request type:</p>

<h4>Finite State Machine (FSM)</h4>
<p>Used for regulated processes like placing an order, ensuring deterministic behavior. Sequence: $s_0$ → $s_5$.</p>
<p><strong>Cycle:</strong> Category Selection → Product Selection → Price Confirmation → Quantity Input → Finalization.</p>

<h4>Generative Pipeline (RAG)</h4>
<p>Used for open-ended consultations. Modeled as a Directed Acyclic Graph (DAG): normalization → classification → retrieval → context augmentation → generation.</p>


![rag-pipe](https://github.com/user-attachments/assets/155744bc-8932-44c1-96ee-5a7f3a7ee337)



<h3>2.2. Queuing Network Model (M/M/m)</h3>
<p>The architecture is represented as an open network of service nodes to analyze response times ($T_{resp}$).</p>

<ul>
  <li><strong>Nodes ($v$):</strong>
    <ul>
      <li>$v_{bot}$: Telegram Bot application (Interface and Logic)</li>
      <li>$v_{LLM}$: External LLM Service (Gemini)</li>
      <li>$v_{DB}$: Hybrid Knowledge Base (MongoDB + ChromaDB)</li>
    </ul>
  </li>
  <li><strong>Characteristics:</strong> Each node is a multiserver queue ($M/M/m$) with Poisson input ($\lambda$) and exponential service ($\mu$).</li>
  <li><strong>Load Balancing:</strong> Interactions defined by transition probability matrix $R$. Total response time:
    <p style="text-align:center;">$$T_{resp} = \alpha_{bot}T_{bot} + \alpha_{LLM}T_{LLM} + \alpha_{DB}T_{DB}$$</p>
  </li>
</ul>

<p>This model allows calculation of system stability ($\rho < 1$) and optimization of channel capacity ($m$) to meet the target response time of under 10 seconds.</p>
