# ff-assistant

Read-only fantasy football reports for all active NFL leagues on a configured Sleeper account. Each league gets its own Tuesday waiver, Thursday injury, and Sunday lineup report in one private Discord channel, scheduled for 8:00 AM America/Chicago. No roster moves are submitted.

## Setup

1. Create a private Discord channel visible only to you, then create a webhook for that channel. A Discord webhook posts to a channel; it cannot send a direct message.
2. In GitHub repository **Settings → Secrets and variables → Actions**, create `DISCORD_WEBHOOK_URL`, `GEMINI_API_KEY`, and `SLEEPER_USERNAME` secrets. Get the Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
3. Push this workflow to the repository's default branch, then use **Actions → Fantasy Football Agent → Run workflow** to test each mode. Scheduled workflows only run from the default branch.

Run local commands from the repository root. The app reads settings from its `.env` file. Copy `.env.example` to `.env` and fill in `SLEEPER_USERNAME`, `DISCORD_WEBHOOK_URL`, and `GEMINI_API_KEY` for reports. The app automatically selects all `in_season` leagues in Sleeper's current NFL season. Add `SLEEPER_LEAGUE_IDS` to `.env` as a comma-separated list to select specific leagues. `GEMINI_MODEL` defaults to `gemini-3.8-flash`; add it to `.env` to override the default. No Sleeper API key is needed. GitHub Actions creates `.env` from repository secrets and workflow settings for each run.

## Local commands

```bash
python -m pip install -e '.[dev]'
python -m ff_assistant.main --list-leagues
python -m ff_assistant.main --mode waivers --preview-context
python -m ff_assistant.main --mode lineup
pytest
```

The `.cache/sleeper-players.json` file is refreshed at most once per day. GitHub Actions caches it between runs. Reports use Sleeper league settings, rosters, player metadata, and trending additions. The model receives a compact roster-specific snapshot, never the entire player database.

## Data limits

Sleeper's injury status is a snapshot, not an official NFL practice report. Injury and lineup runs also check the [NFL injury report](https://www.nfl.com/injuries/) and reject pages that show a different week; when the current report is unavailable, the output says so. This version has no verified weekly projections, opponent schedule, kickoff times, or historical injury snapshots. Reports avoid numerical projected-point claims. Waiver candidates come from available trending additions, so quiet breakout candidates may be missed. If a league fetch or report fails, the workflow fails and logs the league name without printing credentials.

To preview data without a Gemini call or Discord post, run `--preview-context`.
