# Polymarket-Analysis-Agent
This project combines the Model Context Protocol with LLM model and LangGraph agents for analyzing Polymarket


## Features

- **Fetch Live Polymarket Events**  
  Pulls open markets via the [Polymarket Gamma API](https://polymarket.com).

-  **Classify Event category with Claude**  
  Automatically categorizes events into topics like *Politics*, *Tech*, *Crypto*, etc.

- **View Price History**  
  Uses CLOB API to fetch historical pricing data for each market.

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

