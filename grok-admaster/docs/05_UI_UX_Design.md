# UI/UX Design Document

## 1. Design System

### Colors (Dark Mode Theme)
*   **Backgrounds:**
    *   `bg-obsidian`: `#0f1115` (Main App Background)
    *   `bg-panel`: `#181b21` (Card/Panel Background)
*   **Accents:**
    *   `text-primary`: `#ffffff`
    *   `text-secondary`: `#9ca3af` (Gray-400)
    *   `brand-amazon`: `#ff9900` (Amazon Orange - for specific highlights)
    *   `brand-grok`: `#00e5ff` (Cyan/Neon Blue - for AI elements)
*   **Status:**
    *   `success`: `#10b981` (Green - Low ACoS / High Sales)
    *   `danger`: `#ef4444` (Red - High ACoS / Inventory Alert)

### Typography
*   **Font:** Inter (San-serif). Clean, legible, tech-oriented.
*   **Headings:** Bold, often uppercase for "War Room" headers.
*   **Numbers:** Monospace for tabular data (e.g., `font-mono`).

## 2. Layout Structure

### The Shell
*   **Sidebar (Left):** Navigation (Dashboard, Campaigns, Products, Settings). Collapsible.
*   **Header (Top):** User Profile, Global Time/Date, Sync Status indicator.
*   **Main Content (Center):** The active view.
*   **Grok Assistant (Floating/Right Panel):** A persistent chat drawer or floating button that expands.

## 3. Key Screens

### 3.1. Dashboard ("The War Room")
*   **Top Row:** 4 KPI Cards (Sales, Spend, ACoS, Profit).
*   **Middle:** Large "Velocity" Chart. Dual axis: Sales Bars vs. ACoS Line.
*   **Bottom:** "Live Feed" of AI actions (e.g., "Grok lowered bid on 'blue widgets' @ 10:42 AM").

### 3.2. Campaign Manager
*   Table view.
*   Columns: Status, Campaign Name, Strategy (Dropdown), Budget, ACoS.
*   "Grok Mode" toggle switch for every row.

### 3.3. Chat Interface
*   Conversation history scroll.
*   Input field with "Quick Actions" chips (e.g., "Analyze last 7 days", "Audit SEO").
*   Grok's responses should render Markdown (bolding key insights).
