"""Modules for MCP tools related to FF Assistant."""

import asyncio
import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# Initialize the MCP server
app = Server("sleeper-fantasy-football")

# Base URL for Sleeper API
SLEEPER_API_BASE = "https://api.sleeper.app/v1"

async def fetch_sleeper_data(endpoint: str):
    """Helper function to fetch data from Sleeper API"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{SLEEPER_API_BASE}{endpoint}")
        response.raise_for_status()
        return response.json()

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for interacting with Sleeper API"""
    return [
        Tool(
            name="get_user",
            description="Get Sleeper user information by username",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Sleeper username to look up"
                    }
                },
                "required": ["username"]
            }
        ),
        Tool(
            name="get_user_leagues",
            description="Get all leagues for a user in a specific season",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Sleeper user ID"
                    },
                    "season": {
                        "type": "string",
                        "description": "NFL season year (e.g., '2024')"
                    },
                    "sport": {
                        "type": "string",
                        "description": "Sport type (default: 'nfl')",
                        "default": "nfl"
                    }
                },
                "required": ["user_id", "season"]
            }
        ),
        Tool(
            name="get_league_rosters",
            description="Get all rosters in a league",
            inputSchema={
                "type": "object",
                "properties": {
                    "league_id": {
                        "type": "string",
                        "description": "Sleeper league ID"
                    }
                },
                "required": ["league_id"]
            }
        ),
        Tool(
            name="get_league_matchups",
            description="Get matchups for a specific week in a league",
            inputSchema={
                "type": "object",
                "properties": {
                    "league_id": {
                        "type": "string",
                        "description": "Sleeper league ID"
                    },
                    "week": {
                        "type": "integer",
                        "description": "NFL week number (1-18)"
                    }
                },
                "required": ["league_id", "week"]
            }
        ),
        Tool(
            name="get_league_info",
            description="Get detailed information about a league",
            inputSchema={
                "type": "object",
                "properties": {
                    "league_id": {
                        "type": "string",
                        "description": "Sleeper league ID"
                    }
                },
                "required": ["league_id"]
            }
        ),
        Tool(
            name="get_nfl_players",
            description="Get all NFL players data (cached, updates daily)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    
    try:
        if name == "get_user":
            username = arguments["username"]
            data = await fetch_sleeper_data(f"/user/{username}")
            return [TextContent(
                type="text",
                text=f"User found: {data.get('display_name')} (ID: {data.get('user_id')})\n"
                     f"Avatar: {data.get('avatar', 'None')}\n"
                     f"Full data: {data}"
            )]
        
        elif name == "get_user_leagues":
            user_id = arguments["user_id"]
            season = arguments["season"]
            sport = arguments.get("sport", "nfl")
            data = await fetch_sleeper_data(f"/user/{user_id}/leagues/{sport}/{season}")
            
            if not data:
                return [TextContent(type="text", text="No leagues found for this user in this season.")]
            
            leagues_summary = f"Found {len(data)} league(s):\n\n"
            for league in data:
                leagues_summary += f"- {league.get('name')} (ID: {league.get('league_id')})\n"
                leagues_summary += f"  Status: {league.get('status')}, Teams: {league.get('total_rosters')}\n"
            
            leagues_summary += f"\nFull data: {data}"
            return [TextContent(type="text", text=leagues_summary)]
        
        elif name == "get_league_rosters":
            league_id = arguments["league_id"]
            data = await fetch_sleeper_data(f"/league/{league_id}/rosters")
            
            rosters_summary = f"Found {len(data)} roster(s):\n\n"
            for roster in data:
                rosters_summary += f"Roster {roster.get('roster_id')}: "
                rosters_summary += f"Owner ID: {roster.get('owner_id')}, "
                rosters_summary += f"Wins: {roster.get('settings', {}).get('wins', 0)}, "
                rosters_summary += f"Losses: {roster.get('settings', {}).get('losses', 0)}\n"
            
            rosters_summary += f"\nFull data: {data}"
            return [TextContent(type="text", text=rosters_summary)]
        
        elif name == "get_league_matchups":
            league_id = arguments["league_id"]
            week = arguments["week"]
            data = await fetch_sleeper_data(f"/league/{league_id}/matchups/{week}")
            
            matchups_summary = f"Week {week} matchups:\n\n"
            matchup_dict = {}
            for matchup in data:
                matchup_id = matchup.get('matchup_id')
                if matchup_id not in matchup_dict:
                    matchup_dict[matchup_id] = []
                matchup_dict[matchup_id].append(matchup)
            
            for matchup_id, teams in matchup_dict.items():
                matchups_summary += f"Matchup {matchup_id}:\n"
                for team in teams:
                    matchups_summary += f"  Roster {team.get('roster_id')}: {team.get('points', 0)} points\n"
            
            matchups_summary += f"\nFull data: {data}"
            return [TextContent(type="text", text=matchups_summary)]
        
        elif name == "get_league_info":
            league_id = arguments["league_id"]
            data = await fetch_sleeper_data(f"/league/{league_id}")
            
            info = f"League: {data.get('name')}\n"
            info += f"Season: {data.get('season')}\n"
            info += f"Status: {data.get('status')}\n"
            info += f"Teams: {data.get('total_rosters')}\n"
            info += f"Scoring: {data.get('scoring_settings', {})}\n"
            info += f"\nFull data: {data}"
            return [TextContent(type="text", text=info)]
        
        elif name == "get_nfl_players":
            data = await fetch_sleeper_data("/players/nfl")
            player_count = len(data)
            sample_players = list(data.items())[:5]
            
            result = f"Retrieved {player_count} NFL players.\n\nSample players:\n"
            for player_id, player_data in sample_players:
                result += f"- {player_data.get('full_name')} ({player_data.get('position')})\n"
            
            result += f"\nNote: Full player data contains {player_count} players. Use this data to look up player info by ID."
            return [TextContent(type="text", text=result)]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except httpx.HTTPError as e:
        return [TextContent(type="text", text=f"Error fetching data from Sleeper API: {str(e)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
