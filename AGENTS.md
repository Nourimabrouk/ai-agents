# Repository Guidelines

## Project Structure & Module Organization
- `agents/`: Framework and domain agents (`claude-code/`, `microsoft/`, `langchain/`, `accountancy/`).
- `frameworks/`: Framework-specific code and examples (e.g., `claude-code/`).
- `tools/`: Reusable tools and the tool registry (`tools/tool_framework.py`).
- `templates/`: Base patterns (`templates/base_agent.py`).
- `orchestrator.py`: Multi-agent coordination entry point and patterns.
- `projects/`: End-to-end apps (`invoice-automation/`, etc.).
- `utils/`, `assets/`, `docs/`, `planning/`: Utilities, outputs, docs, learning materials.
- Config: `.env` (use `.env.example`), GitHub Actions in `.github/workflows/`.

## Build, Test, and Development Commands
- Install deps: `npm run setup` (runs `npm install` + `pip install -r requirements.txt`).
- Dev servers: `npm run dev` (Claude MCP + monitoring dashboard).
- Build: `npm run build` (`build:claude`, `build:monitoring`).
- Lint/format: `npm run lint` | `npm run lint:fix` | `npm run format`.
- Tests (JS/TS): `npm test` or `npm run test:agents`.
- Python env (Windows): `python -m venv ai-agents-env && ai-agents-env\Scripts\activate && pip install -r requirements.txt`.

## Coding Style & Naming Conventions
- Python: 4 spaces, type hints, snake_case; format with Black; lint with Flake8; type-check with MyPy.
- JS/TS: ESLint + Prettier; 2 spaces; camelCase for variables/functions, PascalCase for classes; files kebab-case (e.g., `mcp-server.js`).
- Paths: group by framework/domain under `agents/`; shared logic in `tools/` or `templates/`.

## Testing Guidelines
- JS/TS: Jest. Name tests `*.test.ts`/`*.test.js`; colocate with sources or under `agents/**/__tests__/`.
- Python: Pytest available; use `tests/` mirror structure or colocate as `test_*.py`.
- Run: `npm test` for Node; `pytest` for Python components.
- Include edge cases and basic async flows; mock external APIs.

## Commit & Pull Request Guidelines
- Commits: Imperative, concise; prefer Conventional Commits (e.g., `feat(agents): add invoice parser`, `fix(tools): handle empty input`).
- PRs: Clear description, linked issues, steps to test, logs/screenshots if applicable; update docs (`README.md`, `docs/`) when behavior changes.
- CI: Ensure `npm run lint`, `npm test`, and type checks pass before requesting review.

## Security & Configuration Tips
- Never commit secrets; copy `.env.example` to `.env` and fill keys (`ANTHROPIC_API_KEY`, etc.).
- Use `NODE_ENV=development` locally; avoid logging secrets; rotate keys when needed.
- Follow least-privilege for any service credentials and prefer local mocks in tests.
