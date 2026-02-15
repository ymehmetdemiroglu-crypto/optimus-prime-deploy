# Coding Standards Document

## 1. General Principles
*   **KISS:** Keep It Simple, Stupid.
*   **DRY:** Don't Repeat Yourself.
*   **Comments:** Comment *why*, not *what*.

## 2. Frontend (React/TS)
*   **Functional Components:** Use React Functional Components (FC) with Hooks. No Class components.
*   **Exports:** Named exports preferred over default exports for better refactoring support.
*   **Imports:** Group imports:
    1. React / External Libraries
    2. Internal Components
    3. Utils / Hooks / Types
    4. CSS / Assets
*   **Linting:** ESLint + Prettier (Standard config).
*   **State:** Use `useState` for local UI state. Use `Context` for global app state (User, Theme).

## 3. Backend (Python/FastAPI)
*   **Formatting:** Use `black` for code formatting.
*   **Imports:** Sort using `isort`.
*   **Type Hints:** Essential for all route handlers and core logic.
*   **Async:** Use `async def` for route handlers to utilize ASGI benefits.

## 4. File Structure (Convention)

```
/src
  /components
    /ui          # Generic UI kit
    /layout      # Shell, Header, Sidebar
    /dashboard   # Specific dashboard widgets
  /pages         # Route targets
  /api           # Axios/Fetch wrappers
  /types         # Shared TS interfaces
  /utils         # Helpers (formatting, math)
```

## 5. Version Control
*   **Commit Messages:** Imperative mood.
    *   *Good:* "Add chart component"
    *   *Bad:* "Added chart component" or "Fixing stuff"
