# Frontend (Loveable) UX and API Changes

This document specifies changes to make in the Loveable UI project so it aligns with reviewer feedback and uses the new backend APIs. The backend in this repo is already updated; deploy the Loveable app with these changes and point it at this backend.

---

## 1. Layout and UX

### 1.1 Tabs instead of scrolling panel

- Replace the single long scrolling panel on the left (problem description area) with **tabs**.
- Suggested tab labels: **Description** | **Examples** | **Rubric** | **Allowed libraries** | **Stats** (see section 3).
- Each tab shows one section only; no long single scroll. Reference: [doraemon123.lovable.app/challenges/lcp/lcp-001](https://doraemon123.lovable.app/challenges/lcp/lcp-001).

### 1.2 Redundant information

- Show the **problem title once** (e.g. only in the main problem header or as the first heading in the Description tab).
- Remove duplicate titles from the description body, examples header, or sidebar so the same title does not appear multiple times.

### 1.3 Task description format

- Use clear headings and structure: Description, Objective, Dataset, Your Task, Tips (or equivalent).
- The API already returns structured data: `GET /api/challenges/<id>` includes `description`, `examples`, `scoring`, `allowed_libraries`. Render these in the corresponding tabs with consistent typography and spacing (LeetCode-style).

### 1.4 "AI/ML Challenges" left-aligned

- Left-align the "AI/ML Challenges" heading (and any sibling navigation) so it is not centered.

### 1.5 "... and more is coming"

- Do not let this text occupy a full line when there is space above or beside it.
- Place it inline (e.g. next to the challenge count or as a small note) so it shares the line or sits in a compact block.

---

## 2. API usage (no change to endpoints you already use)

- **List challenges:** `GET /api/challenges` → `{ challenges: [...] }`
- **Challenge details (now includes stats):** `GET /api/challenges/<id>` → includes `stats` (see below).
- **Run code:** `POST /api/challenges/<id>/run` with `{ "code": "..." }`
- **Submit code:** `POST /api/challenges/<id>/submit` with `{ "code": "...", "prompt_text": "..." }`  
  - `prompt_text` is optional; send the user’s prompt (e.g. concatenated chat messages that led to this code) for Prompt Golf.
- **Agent chat:** `POST /api/agent/chat` (unchanged)
- **Stats only (optional):** `GET /api/challenges/<id>/stats` if you prefer to load stats in a separate request.

---

## 3. Social metrics (Stats tab or block)

For each challenge, show the following. Data comes from the `stats` object on `GET /api/challenges/<id>` (or from `GET /api/challenges/<id>/stats`).

### 3.1 Stats object shape

```json
{
  "acceptance_rate": 0.82,
  "total_attempts": 100,
  "passed_attempts": 82,
  "shortest_passing_prompt_length": 80,
  "shortest_passing_prompt_preview": "First 50 chars of shortest passing prompt or null"
}
```

### 3.2 What to display

- **Acceptance rate:** e.g. `82%` (format `acceptance_rate` as percentage; show "—" or "N/A" when `total_attempts === 0`).
- **Prompt Golf:** e.g. "Shortest passing prompt: 80 chars" (use `shortest_passing_prompt_length`; show "—" when null). Optionally show `shortest_passing_prompt_preview` (e.g. in a tooltip or subtitle).
- **Your current prompt:** Character count of the **current** prompt in the chat (the text the user would send for this solution). Compute this on the client (length of the current prompt text); no new API. Display e.g. "Your current prompt: 0 chars".

### 3.3 Where to put it

- Add a **Stats** (or **Social**) tab in the problem left panel, or a compact block (e.g. above or below the tabs) so it does not clutter the description.

---

## 4. Submit request: send `prompt_text`

When the user clicks Submit, send the prompt that led to the current code so the backend can update Prompt Golf and acceptance rate:

- If the user has a single prompt (e.g. one chat message), send that string.
- If the user has a conversation, send a concatenation of the user messages (e.g. joined with newlines) or the last user message; keep it consistent so "shortest passing prompt" is comparable.

Example request body:

```json
{
  "code": "...",
  "prompt_text": "Write a sentiment classifier using TF-IDF and LogisticRegression."
}
```

---

## 5. Summary checklist

- [ ] Replace scrolling panel with tabs: Description | Examples | Rubric | Allowed libraries | Stats
- [ ] Show problem title once; remove duplicates
- [ ] Refine description formatting (headings, lists)
- [ ] Left-align "AI/ML Challenges"
- [ ] Put "... and more is coming" inline
- [ ] Add Stats (or Social) section with Acceptance rate, Prompt Golf, Your current prompt (chars)
- [ ] Include `prompt_text` in `POST /api/challenges/<id>/submit` when available
- [ ] Use `stats` from `GET /api/challenges/<id>` (or fetch `/stats`) for the Stats UI
