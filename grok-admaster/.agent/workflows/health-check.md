---
description: A workflow to verify the health and correctness of the codebase.
---

// turbo-all
# System Health & Verification Workflow

Use this workflow to ensure that the project is in a stable state after making changes.

## 1. Backend Verification
1. **Linting Check**: Run `flake8` or `ruff` if installed to check for Python style issues.
2. **Startup Check**: Attempt to start the FastAPI server to ensure there are no import errors.
   - Command: `cd server && python -m uvicorn app.main:app --reload --port 8000` (Wait for "Application startup complete").

## 2. Frontend Verification
1. **Build Check**: Run `npm run build` in the `client` directory to ensure there are no production build errors.
2. **Type Check**: Run `npx tsc --noEmit` in the `client` directory to catch TypeScript errors.
3. **Linting Check**: Run `npm run lint` to enforce styling rules.

## 3. Visual Verification
1. **Open Browser**: Launch the development server URL to manually inspect changes.
2. **Interact**: Test the specific feature path to ensure buttons click and data loads as expected.
