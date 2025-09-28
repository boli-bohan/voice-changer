# Repository Guidelines

## Project Structure & Module Organization
- Core Python lives at the repo root: `server.py` exposes the FastAPI WebSocket interface, `shift_pitch.py` owns audio transforms, and `client.py` provides CLI helpers. Keep shared utilities near these modules.
- The React + Vite frontend sits in `frontend/`; source files are under `src/`, public assets under `public/`, and build artefacts in `dist/`.
- `temp_audio/` stores throwaway WAV buffers created during local runs—do not commit its contents. Tooling metadata is centralised in `pyproject.toml`, `uv.lock`, `frontend/package.json`, and the project-wide `Justfile`.

## Build, Test, and Development Commands
- `just install` installs backend dependencies with `uv sync` and runs `npm install` for the frontend.
- `just up` starts `uv run uvicorn server:app --reload` alongside `npm run dev`.
- `just down` stops both processes; pair with `just clean` to clear `temp_audio/` and build outputs.
- `just test` runs `uv run pytest` then `npm run test` (Vitest). Run it before every PR, and rerun targeted commands while iterating.

## Coding Style & Naming Conventions
- Use Python 3.11, four-space indents, and keep lines below 100 characters. Functions and modules follow `snake_case`; classes use `PascalCase`. Prefer explicit imports and type annotations on new interfaces.
- Ruff enforces linting (rules `E,F,I,N,W,UP`); fix issues via `uv run ruff check .` or auto-format with `uv run ruff format .`.
- Frontend code is TypeScript-first. Components and files are `PascalCase` (`PushToTalkButton.tsx`), hooks are `use`-prefixed camelCase, and shared utilities live in `frontend/src/lib/` (create as needed).
- Run `npm run format` to apply Prettier. ESLint is temporarily disabled; when touching configuration, aim to restore `npm run lint`.

## Testing Guidelines
- Backend tests should live under a `tests/` package (add it if absent) and rely on `pytest` plus `pytest-asyncio` to cover WebSocket flows with async fixtures.
- Frontend tests can sit beside components or in `frontend/src/__tests__/`, using Vitest and React Testing Library patterns. Mock microphones and WebSocket events to keep tests deterministic.
- Add regression tests alongside new features; document any skipped coverage in the PR description.

## Commit & Pull Request Guidelines
- Existing history only contains `Initial commit`, so continue writing concise, imperative summaries (e.g. `Implement streaming retry`).
- PRs should explain intent, list verification steps (`just test`, manual PTT run), and reference tracking tickets. Attach screenshots or screen recordings for UI adjustments.
- Call out configuration changes, required migrations, or ops follow-up so reviewers can coordinate quickly.
