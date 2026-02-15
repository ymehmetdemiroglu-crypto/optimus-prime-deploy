# Component Library Rules

## 1. Component Philosophy
*   **Atomic Design:** Build from small `atoms` to complex `organisms`.
*   **Composition:** Prefer composition over massive prop drilling.
*   **Strict Typing:** All components must have a `Props` interface.

## 2. Naming Conventions
*   **Files:** PascalCase (e.g., `StatCard.tsx`, `ChatBubble.tsx`).
*   **Folders:** If a component has sub-parts or styles, group in a folder: `components/Dashboard/StatCard/index.tsx`.
*   **Props:** CamelCase (e.g., `isOpen`, `onClick`).

## 3. Reusable Components (The "UiKit")
We will build a set of base components in `src/components/ui`:

*   `Button`: Variants (`primary` (Grok Blue), `secondary`, `danger`, `ghost`).
*   `Card`: The base container for all dashboard panels. Standard padding and rounded corners.
*   `Badge`: For status indicators (Active, Paused).
*   `Input`: Standard text inputs with consistent focus states.
*   `Loader`: A "Grok" themed loading spinner.

## 4. Specific Patterns
*   **Data Display:** Use `Intl.NumberFormat` for all currency.
    *   Example: `$1,234.56`
*   **Charts:** Wrap `Recharts` components in a `ResponsiveContainer` always.
*   **Empty States:** Every list/table must have a visual "No Data" state with a Call to Action.
