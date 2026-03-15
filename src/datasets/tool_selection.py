import json
import random
import os
from pathlib import Path

TOOL_A = {
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "Gets the current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. London",
                },
            },
            "required": ["location"],
        },
    }
}

TOOL_B = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluates a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '2 + 2'",
                },
            },
            "required": ["expression"],
        },
    }
}

TOOL_C = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Gets the current stock price for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol, e.g. AAPL",
                },
            },
            "required": ["ticker"],
        },
    }
}

TOOL_D = {
    "type": "function",
    "function": {
        "name": "get_flight_status",
        "description": "Gets the current status of a flight.",
        "parameters": {
            "type": "object",
            "properties": {
                "flight_number": {
                    "type": "string",
                    "description": "The flight number, e.g. BA123",
                },
            },
            "required": ["flight_number"],
        },
    }
}

# 25 diverse temperature/weather queries for TOOL_A
QUERIES_A = [
    "What's the temperature in London?",
    "How cold is it in New York right now?",
    "Tell me the current weather in Tokyo.",
    "Is it freezing in Berlin today?",
    "Get the temperature for Paris, France.",
    "What's the current temperature in Sydney?",
    "How hot is it in Dubai right now?",
    "Check the temperature in Moscow for me.",
    "What is the temperature outside in Toronto?",
    "Is it warm in Barcelona today?",
    "Tell me the temperature reading for Amsterdam.",
    "How cold is Chicago right now?",
    "What's the weather like in Seoul?",
    "Get me the current temperature for Mumbai.",
    "Is it hot in Cairo at the moment?",
    "What's the temperature in Buenos Aires?",
    "How warm is it in Singapore today?",
    "Tell me the current temperature in Los Angeles.",
    "What's the temperature reading for Istanbul?",
    "How chilly is it in Stockholm today?",
    "Get the temperature for Cape Town, South Africa.",
    "Is it cold in Helsinki right now?",
    "What's the temperature in Nairobi?",
    "How warm is Miami today?",
    "Tell me the current temperature in Mexico City.",
]

# 25 diverse math/calculator queries for TOOL_B
QUERIES_B = [
    "What is 25 multiplied by 4?",
    "Calculate 134 + 288.",
    "Evaluate the expression 15 / 3.",
    "How much is 90 minus 12?",
    "Find the result of 7 times 8.",
    "What is 500 divided by 25?",
    "Compute 18 squared.",
    "What is the sum of 456 and 789?",
    "Multiply 33 by 6 for me.",
    "What is 1000 minus 347?",
    "Calculate the product of 12 and 15.",
    "Evaluate 2 to the power of 10.",
    "What is 144 divided by 12?",
    "Add 256 and 512 together.",
    "What is 75 percent of 200?",
    "Compute 9 factorial.",
    "What is 3.14 times 100?",
    "Find the remainder when 17 is divided by 5.",
    "What is the square root of 169?",
    "Calculate 888 minus 444.",
    "Multiply 0.5 by 1000.",
    "What is 2024 divided by 4?",
    "Evaluate 100 plus 200 plus 300.",
    "What is 15 cubed?",
    "Compute the average of 10, 20, and 30.",
]

QUERIES_C = [
    "What's the current price of AAPL?",
    "How is TSLA doing today in the market?",
    "Check the stock price for Microsoft.",
    "Is GOOGL up or down right now?",
    "Get me the latest quote for Amazon stock.",
    "What is the stock price of Meta?",
    "How much is one share of NVIDIA?",
    "Tell me the current trading price of NFLX.",
    "Check the value of Berkshire Hathaway shares.",
    "What's the stock ticker for Disney at right now?",
    "How is the S&P 500 performing today?",
    "Get the current price for IBM.",
    "Is AMD stock gaining today?",
    "What is the market price of Intel?",
    "Check the stock value of Coca-Cola.",
    "How much is Walmart stock trading for?",
    "Tell me the stock price for Pfizer.",
    "What is Johnson & Johnson's stock at?",
    "Check the current price of Visa stock.",
    "How is Mastercard stock doing?",
    "Get the stock quote for JPMorgan Chase.",
    "What's the price of Bank of America shares?",
    "Check the stock price for ExxonMobil.",
    "How is Chevron stock performing?",
    "Tell me the current price of AT&T stock.",
]

QUERIES_D = [
    "Is flight BA123 on time?",
    "Check the status of DL456.",
    "Has AA789 landed yet?",
    "What's the estimated arrival time for UA101?",
    "Is my flight AF202 delayed?",
    "Get the status for LH303.",
    "Check if flight EK404 has departed.",
    "What gate is QF505 arriving at?",
    "Is AC606 still scheduled for 3 PM?",
    "Tell me the flight status for VS707.",
    "Check the departure time for flight JAL808.",
    "Has CX909 been cancelled?",
    "What is the status of SQ111?",
    "Is flight MH222 boarding now?",
    "Check the arrival time for TG333.",
    "Has flight PR444 arrived in Manila?",
    "What's the status of my flight VN555?",
    "Is flight BR666 delayed?",
    "Check the status for flight CI777.",
    "Has KE888 departed Seoul yet?",
    "What is the status of flight OZ999?",
    "Check if flight AI1111 is on schedule.",
    "Is 9W2222 delayed due to weather?",
    "Has flight G83333 taken off?",
    "Check the status of flight 6E4444.",
]

def generate_tool_selection_dataset(num_samples=1000, output_dir=None, seed=42):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "tool_selection"
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    dataset = []

    tools = [TOOL_A, TOOL_B, TOOL_C, TOOL_D]
    queries_map = {
        "get_current_temperature": QUERIES_A,
        "calculator": QUERIES_B,
        "get_stock_price": QUERIES_C,
        "get_flight_status": QUERIES_D
    }

    for _ in range(num_samples):
        # Sample two distinct tools
        clean_tool_obj, corrupted_tool_obj = random.sample(tools, 2)
        
        clean_tool_name = clean_tool_obj["function"]["name"]
        corrupted_tool_name = corrupted_tool_obj["function"]["name"]
        
        query_clean = random.choice(queries_map[clean_tool_name])
        query_corrupted = random.choice(queries_map[corrupted_tool_name])

        clean_messages = [
            {"role": "system", "content": "You are a model that can do function calling with the following functions"},
            {"role": "user", "content": query_clean}
        ]

        corrupted_messages = [
            {"role": "system", "content": "You are a model that can do function calling with the following functions"},
            {"role": "user", "content": query_corrupted}
        ]

        dataset.append({
            "tools": tools, # ALL tools in context
            "clean_messages": clean_messages,
            "corrupted_messages": corrupted_messages,
            "target_clean_tool": clean_tool_name,
            "target_corrupted_tool": corrupted_tool_name
        })

    file_path = os.path.join(output_dir, "dataset.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print(f"Generated {len(dataset)} Tool Selection samples at {file_path}")
    return file_path

if __name__ == "__main__":
    generate_tool_selection_dataset()
