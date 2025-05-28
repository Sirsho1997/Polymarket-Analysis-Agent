
import json
import os
from dotenv import load_dotenv
import anthropic
import pandas as pd
from pandas import DataFrame
import requests
import datetime
from mcp.server.fastmcp import FastMCP
from typing import List
from glob import glob

load_dotenv()
client = anthropic.Anthropic()

EVENT_DIR = "trade_events"

# Initialize FastMCP server
mcp = FastMCP("polymarket_server")

def classify_descriptions_with_claude(questions: List[str], batch_size: int = 10) -> List[str]:
    """
    Classifies event titles/questions into predefined categories using Claude model.

    Args:
        questions (List[str]): A list of event description strings.
        batch_size (int, optional): # of descriptions to send in each prompt batch. Defaults to 10.

    Returns:
        List[str]: A list of category labels corresponding to each description.
    """
    all_categories = []

    # List of allowed category labels
    CATEGORY_LIST = ["Politics", "Sports", "Crypto", "Tech", "Culture", "Economy", "Entertainment"]

    # Process descriptions in batches
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]

        # Format each description with an index number
        formatted_descriptions = "\n\n".join(
            [f"{idx + 1}. {desc.strip()}" for idx, desc in enumerate(batch)]
        )

        # Prompt to instruct Claude on how to classify the descriptions
        prompt = f"""Classify each of the following event descriptions into one of the following categories: {', '.join(CATEGORY_LIST)}

        Respond with a numbered list where each number corresponds to the description number, and each line contains ONLY the category name.
        
        Descriptions:
        {formatted_descriptions}
        
        Example output:
        1. Politics
        2. Tech
        3. Economy
        ...
        
        Now classify:
        """

        try:
            # Send the prompt to Claude model for classification
            message = client.messages.create(
                model='claude-3-7-sonnet-20250219',
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse and validate Claude's response
            lines = message.content[0].text.strip().splitlines()
            for line in lines:
                parts = line.strip().split(". ", 1)
                if len(parts) == 2 and parts[1] in CATEGORY_LIST:
                    # Append valid category
                    all_categories.append(parts[1])
                else:
                    # Fallback if unexpected format or unknown category
                    all_categories.append("Uncategorized")

        except Exception as e:
            print(f"Claude error: {e}")
            # On error, mark all items in the batch with "Error"
            all_categories.extend(["Error"] * len(batch))

    return all_categories

@mcp.tool()
def download_recent_events(n: int = 2) -> str:
    """
    Downloads recent active events from Polymarket.

    Args:
        n (int): # of events to return Default is 2 (reduce claude API call).

    Returns:
        str: A summary of DataFrame containing events details.
    """

    # API endpoint to fetch all currently open events
    url = f"https://gamma-api.polymarket.com/events?closed=false&limit=5000"
    response = requests.get(url)
    events = response.json()

    # Initialize lists to store flattened market data
    question, conditionId, description, volume = [], [], [], []
    outcomePrice_0, outcomePrice_1 = [], []
    bestBid, bestAsk, spread = [], [], []
    startDateIso, endDateIso, createdAt = [], [], []
    yes_token, no_token = [], []

    # Parse through the JSON response to extract required fields
    for event in events:
        for market in event.get("markets", []):
            try:
                question.append(market.get('question'))
                conditionId.append(market.get('conditionId'))
                description.append(market.get('description'))
                volume.append(float(market.get('volume', 0)))

                # Parse outcome prices; use eval to convert string representation of list
                outcome_prices = eval(market.get('outcomePrices', '["0", "0"]'))
                outcomePrice_0.append(float(outcome_prices[0]))
                outcomePrice_1.append(float(outcome_prices[1]))

                # Parse token IDs
                token_ids = eval(market.get('clobTokenIds', '["0", "0"]'))
                yes_token.append(token_ids[0])
                no_token.append(token_ids[1])

                # Append additional market data
                bestBid.append(float(market.get('bestBid', 0)))
                bestAsk.append(float(market.get('bestAsk', 0)))
                spread.append(float(market.get('spread', 0)))
                startDateIso.append(market.get('startDateIso'))
                endDateIso.append(market.get('endDateIso'))
                createdAt.append(market.get('createdAt'))
            except Exception as e:
                print(f"Error parsing market: {e}")
                continue

    print("Data downloaded")

    # Create DataFrame from parsed lists
    event_df = pd.DataFrame({
        "event_id": [idx for idx in range(len(question))],
        "question": question,
        "conditionId": conditionId,
        "description": description,
        "volume": volume,
        "outcomePrice_yes": outcomePrice_0,
        "token_id": yes_token,
        "bestBid": bestBid,
        "spread": spread,
        "startDateIso": startDateIso,
        "endDateIso": endDateIso,
        "createdAt": createdAt
    })

    current_time_ms = int(datetime.datetime.utcnow().timestamp() * 1000)

    event_df["event_id"] += current_time_ms

    # Convert end dates to datetime objects and filter for future events
    event_df["endDate"] = pd.to_datetime(event_df["endDateIso"], errors="coerce")
    current_time = datetime.datetime.utcnow()

    event_df = event_df[event_df["endDate"] > current_time]

    # Filter out resolved markets where the price is already 1.0 (fully priced in) or 0.0
    event_df = event_df[event_df["outcomePrice_yes"] != 1.0]
    event_df = event_df[event_df["outcomePrice_yes"] != 0.0]

    print("Data Filtered based on current time")

    # Randomly sample 'n' events to reduce cost/processing
    event_df = event_df.sample(n, random_state=27)

    print(f"Data Filtered based on random {n} events")

    # Classify each sampled event using Claude model
    categories = classify_descriptions_with_claude(event_df['question'])

    # Add categories as a new column
    event_df['categories'] = categories

    # Ensure directory exists before saving
    os.makedirs(EVENT_DIR, exist_ok=True)

    print(event_df)
    # Save DataFrame to CSV file
    event_df.to_csv(f"{EVENT_DIR}/events_{current_time_ms}.csv", index=False)

    content =''

    for _, row in event_df.iterrows():
        content += f"## {row['question']}\n"
        content += f"- **Event ID**: {row['event_id']}\n"
        content += f"- **Category**: {row.get('categories', 'Unknown')}\n"
        content += f"- **Volume**: ${row['volume']:.2f}\n"
        content += f"- **Yes Price**: {row['outcomePrice_yes']:.2f}\n"
        content += f"- **End Date**: {row['endDateIso']}\n\n"
        content += "---\n\n"

    return content


@mcp.resource("events://latest")
def extract_events() -> str:
    """
    Load the recently saved Polymarket events.

    Returns:
        str: string of event summaries.
    """

    # Find all files like events_*.csv
    csv_files = glob(os.path.join(EVENT_DIR, "events_*.csv"))

    if not csv_files:
        return "# No saved events found.\n\nTry running `download_recent_events` first."

    try:
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)

        print(df)
        content = "# Latest Polymarket Events\n\n"
        content += f"Total events: {len(df)}\n\n"

        for _, row in df.iterrows():
            content += f"## {row['question']}\n"
            content += f"- **Event ID**: {row['event_id']}\n"
            content += f"- **Category**: {row.get('categories', 'Unknown')}\n"
            content += f"- **Volume**: ${row['volume']:.2f}\n"
            content += f"- **Yes Price**: {row['outcomePrice_yes']:.2f}\n"
            content += f"- **End Date**: {row['endDateIso']}\n\n"
            content += "---\n\n"

        return content

    except Exception as e:
        return f"# Error loading events:\n\n{str(e)}"

@mcp.tool()
def get_price_history_clob(event_id : str) -> DataFrame:
    """
    Fetches the historical price data an event using event ID.

    Args:
        event_id (str): ID of the event.

    Returns:
        DataFrame: A DataFrame containing the price history.
    """

    # Find all files like events_*.csv
    csv_files = glob(os.path.join(EVENT_DIR, "events_*.csv"))

    event_df = [pd.read_csv(f) for f in csv_files]
    event_df = pd.concat(event_df, ignore_index=True)

    # Filter for the event with the given ID
    event_df = event_df[event_df['event_id'] == int(event_id)]

    # Extract the token ID used to query the CLOB price history
    token_id = event_df['token_id'].values[0]

    # Define the API endpoint and query parameters
    url = 'https://clob.polymarket.com/prices-history'
    params = {
        'market': token_id,
        'fidelity': 60,     # Fidelity level of price data (60 min = default granularity)
        'interval': 'all'   # Get the entire history
    }

    # Send GET request to the Polymarket CLOB price history API
    r = requests.get(url, params=params)

    if r.status_code == 200:
        # Parse the response JSON
        response = json.loads(r.text)

        # Convert the 'history' field into a DataFrame
        price_history_df = pd.DataFrame(response['history'])

        if price_history_df.empty:
            print("No price history data available.")
            return price_history_df

        # Convert UNIX timestamp to datetime
        price_history_df['t'] = pd.to_datetime(price_history_df['t'], unit='s', utc=True)

        # Rename columns for clarity
        price_history_df.rename(columns={'t': 'timestamp', 'p': 'price'}, inplace=True)

        # Drop the last row, which may contain unrefreshed or partial data
        price_history_df = price_history_df.iloc[:-1]

        # Set timestamp as index for time-series operations
        price_history_df.set_index('timestamp', inplace=True)

        # Reset index if you want 'timestamp' as a column again
        price_history_df.reset_index(inplace=True)

        return price_history_df

    else:
        # Handle failed request
        print(f"Failed to fetch data: {r.status_code}")
        return pd.DataFrame()

# Starting the server
if __name__ == "__main__":
    mcp.run(transport="stdio")
