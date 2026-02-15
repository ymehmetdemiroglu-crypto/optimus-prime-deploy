---
description: A comprehensive workflow for autonomous feature development from planning to implementation and verification.
---

// turbo-all
# Autonomous Feature Development Workflow

This workflow guides the agent through the complete lifecycle of adding a new feature to the Grok AdMaster project.

## 1. Discovery & Planning
1. **Analyze Requirements**: Thoroughly read the task description and examine existing code related to the request.
2. **Identify Impact**: Determine which frontend components and backend endpoints need modifications.
3. **Draft Plan**: Create an implementation plan (usually stored in `docs/plans/` or shared as a message) outlining:
    - Backend changes (models, services, API)
    - Frontend changes (types, components, pages)
    - Testing strategy

## 2. Backend Implementation
1. **Define Data Models**: Update or create Pydantic models in `server/app/models/`.
2. **Implement Logic**: Add core business logic or AI simulations in `server/app/services/`.
3. **Expose Endpoints**: Create or modify route handlers in `server/app/api/`.
4. **Verify Backend**: Run a quick connectivity test or use `curl` to verify the new endpoint if possible.

## 3. Frontend Implementation
1. **Update Types**: Ensure TypeScript interfaces in `client/src/types/` match backend models.
2. **Build Components**: Create or update UI components in `client/src/components/`.
3. **Connect API**: Update API calling logic in `client/src/api/`.
4. **Assemble Pages**: Integrate components into the relevant pages in `client/src/pages/`.

## 4. Verification & Polish
1. **Lint & Type Check**: Run `npm run lint` and `tsc` in the client directory.
2. **Style Check**: Ensure the UI matches the "Cyber-Professional" aesthetic (Glow effects, neon accents, dark mode).
3. **Final Review**: Perform a "self-review" of the code to ensure no debug logs or placeholders are left behind.
