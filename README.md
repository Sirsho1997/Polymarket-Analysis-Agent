# Polymarket-Analysis-Agent
This project combines the **Model Context Protocol (MCP)** with **LLM models** and **LangGraph** agents for analyzing **Polymarket** (prediction market that allows users to gain/lose on the outcome of world events)

## Features

- **Fetch Live Polymarket Events**  
  Pulls open markets via the [Polymarket Gamma API](https://polymarket.com).

-  **Classify Event category with Claude**  
  Automatically categorizes events into topics like *Politics*, *Tech*, *Crypto*, etc.

- **View Price History**  
  Uses CLOB API to fetch historical pricing data for each market, and also appends relevant internet search context for the event question using the [Tavily API](https://www.tavily.com/).

- **Persistent Resource Storage**  
  Each tool run saves data as versioned `.csv` files (e.g., `events_1687628300000.csv`). Where 1687628300000 is the current timestamp in ms.

- **LangGraph REACT Agent Interface**  
  Use a conversational agent that can invoke Model Context Protocol (MCP) tools and resources to answer questions interactively.


##  Project Structure

```
├── polymarket_server.py # Main FastMCP tool and resource server
├── polymarket_client.py # LangGraph + LangChain chat agent
├── trade_events/ # Auto-generated folder to store market event CSVs
│ ├── events_*.csv # Timestamped event snapshots
├── .env # API keys. I experimented with both Anthropic(claude-3-7-sonnet-20250219) and OPENAI key (gpt-4o)
├── requirements.txt
└── README.md

```

##  Quickstart

### 1. Clone the Repo

```bash
git clone https://github.com/Sirsho1997/polymarket-agent.git
cd polymarket-agent
```

### 2. Install Requirements
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
TAVILY_API_KEY=your_tavily_key
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

### 4. Chat with the Agent
```bash
python polymarket_client.py
```

##  Key MCP Tools & Resources
- download_recent_events(n: int)	: Downloads & categorized active Polymarket events based on category.
- get_price_history_clob(event_id: str): Fetches historical price data for an event and also appends search context for the event question AI agents using the **Tavily** API.
- extract_events(): Returns a markdown summary of all saved events.

## Example Conversation
<img width="1400" alt="image" src="https://github.com/user-attachments/assets/7a19a067-468e-4080-ae81-3469402c0218" />
<img width="1400" alt="image" src="https://github.com/user-attachments/assets/e392c7d5-73eb-4421-b2e7-655f8e089786" />
<img width="1400" alt="image" src="https://github.com/user-attachments/assets/22d3bcaa-df64-455d-8a8f-84e4a3aae615" />


## Acknowledgements
[Polymarket ](https://polymarket.com/) — for open prediction market data

[FastMCP](https://gofastmcp.com/getting-started/welcome?gad_source=1&gad_campaignid=22521620347&gbraid=0AAAAACeCpg-oisgSP5zF9Q49q4n0LlCHJ&gclid=CjwKCAjw6NrBBhB6EiwAvnT_rkyzVxseBsYTLbJvZzQ2WOV7-2i9h23VAHBfNKYOY2g1lLZ272uZGRoCY50QAvD_BwE) — Pythonic way to build Model Context Protocol servers and clients

[Tavily API](https://www.tavily.com/) — a specialized search engine designed for Large Language Models (LLMs) and AI agents

[LangGraph](https://www.langchain.com/langgraph) — multi-step agent framework
