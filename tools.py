import random

def get_weather_forecast(location: str) -> str:
    """
    Mock tool to get a weather forecast for a given location.
    In a real scenario, this would call a weather API.
    """
    print(f"TOOL USED: get_weather_forecast(location='{location}')")
    if "City" in location: # Assume cities are more prone to disruption
        return random.choice(["Clear skies", "Heavy rain expected", "Potential storm warning"])
    return random.choice(["Clear skies", "Partly cloudy"])

def get_traffic_status(origin: str, destination: str) -> str:
    """
    Mock tool to get traffic status for a lane.
    In a real scenario, this would call a traffic data API.
    """
    print(f"TOOL USED: get_traffic_status(origin='{origin}', destination='{destination}')")
    return random.choice(["Light traffic", "Moderate congestion", "Heavy delays reported"])

def get_available_tools() -> list:
    """Returns a list of available tool definitions for the agent."""
    tool_defs = [
        {
            "name": "get_weather_forecast",
            "description": "Provides the weather forecast for a specific city or supplier location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city or location name."
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "get_traffic_status",
            "description": "Reports the current traffic conditions between an origin and a destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "The starting point of the lane."
                    },
                    "destination": {
                        "type": "string",
                        "description": "The ending point of the lane."
                    }
                },
                "required": ["origin", "destination"]
            }
        }
    ]
    return tool_defs

# Mapping tool names to their actual functions
AVAILABLE_FUNCTIONS = {
    "get_weather_forecast": get_weather_forecast,
    "get_traffic_status": get_traffic_status,
}
