# ff-assistant

Read-only fantasy football reports for all active NFL leagues on a configured Sleeper account. Each league gets its own Tuesday waiver, Thursday injury, and Sunday lineup report in one private Discord channel, scheduled for 8:00 AM America/Chicago. No roster moves are submitted.

## Setup

1. Create a private Discord channel visible only to you, then create a webhook for that channel. A Discord webhook posts to a channel; it cannot send a direct message.
2. In GitHub repository **Settings → Secrets and variables → Actions**, create `DISCORD_WEBHOOK_URL`, `GEMINI_API_KEY`, and `SLEEPER_USERNAME` secrets. Get the Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
3. Push this workflow to the repository's default branch, then use **Actions → Fantasy Football Agent → Run workflow** to test each mode. Scheduled workflows only run from the default branch.

Run local commands from the repository root. The app reads settings from its `.env` file. Copy `.env.example` to `.env` and fill in `SLEEPER_USERNAME`, `DISCORD_WEBHOOK_URL`, and `GEMINI_API_KEY` for reports. The app automatically selects all `in_season` leagues in Sleeper's current NFL season. Add `SLEEPER_LEAGUE_IDS` to `.env` as a comma-separated list to select specific leagues. `GEMINI_MODEL` defaults to `gemini-3.5-flash-lite`; add it to `.env` to override the default. No Sleeper API key is needed. GitHub Actions creates `.env` from repository secrets and workflow settings for each run.

Set `LOG_LEVEL=DEBUG` in `.env` to log Gemini return details and its full output text. For a manual GitHub Actions run, select **Log Gemini response details and full model output**. Scheduled runs use INFO logging. DEBUG logs can contain private roster advice, so limit access to those logs.

`GEMINI_RETRY_DELAY_SECONDS` defaults to `5`. A Gemini turn is retried up to two times after a transient HTTP 408, 429, or 5xx error, timeout, or connection error, with this many seconds between attempts. Other API errors fail immediately. Set the value in `.env` locally or as a GitHub Actions repository variable to change the delay. The Gemini SDK may also retry transient failures internally before this delay applies.

## Local commands

```bash
python -m pip install -e '.[dev]'
python -m ff_assistant.main --list-leagues
python -m ff_assistant.main --mode waivers --preview-context
python -m ff_assistant.main --mode lineup
pytest
```

The app loads player metadata from the bundled `src/ff_assistant/resources/sleeper-players.json.gz` snapshot. Normal runs and GitHub Actions do not request Sleeper's large `players/nfl` endpoint. The snapshot was fetched on 2026-10-16 and is static until manually refreshed, so its teams, status, and injury fields can become stale. League settings, rosters, and trending additions still come from Sleeper at run time. The model receives a compact roster-specific context, never the entire player database.

To refresh the bundled snapshot from the repository root, run:

```bash
curl --fail --location --compressed https://api.sleeper.app/v1/players/nfl --output src/ff_assistant/resources/sleeper-players.json
gzip -n -9 -f src/ff_assistant/resources/sleeper-players.json
date -u +%Y-%m-%dT%H:%M:%SZ > src/ff_assistant/resources/sleeper-players-fetched-at.txt
```

The model can request regular-season player statistics through `nflreadpy` for rostered players and available trending candidates in its report context. The tool returns season and recent-game attempts, carries, targets, catches, yards, touchdowns, and derived efficiency rates. Offensive snap percentages are included when nflverse player IDs can be matched to the separate PFR snap-count data. Data loads on the first tool call and is reused across leagues in the same run. Tool results state the latest week available; missing or stale data should not be treated as current-week performance.

The model can also request current-week NFL matchups for those players. The `get_player_matchups` tool uses Sleeper's current season and week with the nflverse schedule to return opponents, byes, and kickoff times in Eastern and UTC. It loads on first use and reuses the schedule across leagues in the same run. A missing player team or unavailable schedule is reported explicitly. Matchups do not include a fantasy-point projection or opponent strength rating.

The model can request current dynasty trade values and overall or position rankings for report players through `get_player_values`. It uses [TheTradeCalc's public values API](https://thetradecalc.com/values-api) and fetches data only when requested, caching the board for 10 minutes across leagues. The feed requires no API key. Players are matched by name and position because TheTradeCalc does not provide Sleeper IDs; ambiguous or missing matches are reported as unavailable. Values use a 12-team Superflex PPR baseline without TE premium, so interpret them in light of your league settings. Reports using these values include [Values by TheTradeCalc](https://thetradecalc.com), as required by the source.

## Data limits

Sleeper's injury status is a snapshot, not an official NFL practice report. Injury and lineup runs also check the [NFL injury report](https://www.nfl.com/injuries/) and reject pages that show a different week; when the current report is unavailable, the output says so. This version has no verified weekly projections or historical injury snapshots. Reports avoid numerical projected-point claims. Waiver candidates come from available trending additions, so quiet breakout candidates may be missed. If a league fetch or report fails, the workflow fails and logs the league name without printing credentials.

To preview data without a Gemini call or Discord post, run `--preview-context`.
