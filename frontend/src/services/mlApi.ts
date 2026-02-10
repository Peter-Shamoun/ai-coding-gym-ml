/**
 * ML Backend API Service
 * ======================
 * Typed functions for all Flask backend endpoints.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Summary returned by GET /api/challenges for each challenge. */
export interface MLChallengeSummary {
  id: string;
  title: string;
  difficulty: string;
  tags: string[];
  objective: string;
}

/** Dataset column metadata. */
export interface DatasetColumn {
  name: string;
  type?: string;
  description?: string;
}

/** Dataset information from challenge details. */
export interface DatasetInfo {
  description?: string;
  columns?: DatasetColumn[];
  train_file?: string;
  test_file?: string;
  [key: string]: unknown;
}

/** Scoring category from challenge details. */
export interface ScoringCategory {
  name: string;
  max_score: number;
  description?: string;
}

/** Scoring info from challenge details. */
export interface ScoringInfo {
  max_score?: number;
  pass_threshold?: number;
  categories?: ScoringCategory[];
}

/** Full challenge details returned by GET /api/challenges/:id. */
export interface MLChallengeDetails {
  id: string;
  title: string;
  difficulty: string;
  tags: string[];
  objective: string;
  description: string;
  starter_code: string;
  dataset: DatasetInfo;
  scoring: ScoringInfo;
  examples: Array<Record<string, unknown>>;
  allowed_libraries: string[];
}

/** Grading category result. */
export interface GradingCategory {
  name: string;
  score: number;
  max_score: number;
  feedback: string[];
  accuracy?: number | null;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  per_class_f1?: Record<string, number> | null;
  r2?: number | null;
  rmse?: number | null;
  [key: string]: unknown;
}

/** Grading result included in submit responses. */
export interface GradingResult {
  total_score: number;
  max_score: number;
  passed: boolean;
  categories: GradingCategory[];
  [key: string]: unknown;
}

/** Response from POST /api/challenges/:id/run or /submit. */
export interface ExecutionResult {
  success: boolean;
  mode: "run" | "submit";
  stdout: string;
  stderr: string;
  execution_time: number;
  timed_out: boolean;
  error?: string;
  traceback?: string;
  // Run-mode extras
  sample_accuracy?: number | null;
  train_time?: number | null;
  // Submit-mode extras
  grading?: GradingResult | null;
}

/** Conversation turn for agent history. */
export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

/** Response from POST /api/agent/chat. */
export interface AgentResponse {
  code: string;
  message: string;
  error?: string;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

const API_BASE = "/api";

async function apiFetch<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (networkErr) {
    throw new Error(
      "Cannot reach the backend server. Is Flask running on port 5000?"
    );
  }

  // Read response text first — .json() crashes on empty/HTML bodies
  const text = await res.text();

  let body: Record<string, unknown>;
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    // Response wasn't JSON (likely an HTML error page or empty body)
    throw new Error(
      res.ok
        ? "Backend returned an invalid response"
        : `Backend error ${res.status}: ${text.slice(0, 200)}`
    );
  }

  if (!res.ok) {
    throw new Error(
      (body.error as string) ?? `API error ${res.status}`
    );
  }
  return body as T;
}

/** GET /api/challenges — list all ML challenges. */
export async function fetchMLChallenges(): Promise<MLChallengeSummary[]> {
  const data = await apiFetch<{ challenges: MLChallengeSummary[] }>(
    `${API_BASE}/challenges`
  );
  return data.challenges;
}

/** GET /api/challenges/:id — full challenge details. */
export async function fetchMLChallengeDetails(
  challengeId: string
): Promise<MLChallengeDetails> {
  return apiFetch<MLChallengeDetails>(
    `${API_BASE}/challenges/${encodeURIComponent(challengeId)}`
  );
}

/** POST /api/challenges/:id/run — quick test on sample data. */
export async function runCode(
  challengeId: string,
  code: string
): Promise<ExecutionResult> {
  return apiFetch<ExecutionResult>(
    `${API_BASE}/challenges/${encodeURIComponent(challengeId)}/run`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    }
  );
}

/** POST /api/challenges/:id/submit — full grading. */
export async function submitCode(
  challengeId: string,
  code: string
): Promise<ExecutionResult> {
  return apiFetch<ExecutionResult>(
    `${API_BASE}/challenges/${encodeURIComponent(challengeId)}/submit`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    }
  );
}

/** POST /api/agent/chat — AI coding agent. */
export async function agentChat(
  challengeId: string,
  message: string,
  currentCode?: string,
  history?: ChatTurn[]
): Promise<AgentResponse> {
  return apiFetch<AgentResponse>(`${API_BASE}/agent/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      challenge_id: challengeId,
      message,
      current_code: currentCode ?? "",
      history: history ?? [],
    }),
  });
}

/** Build a dataset download URL. */
export function getDatasetDownloadUrl(
  challengeId: string,
  filename: string
): string {
  return `${API_BASE}/challenges/${encodeURIComponent(challengeId)}/dataset/${encodeURIComponent(filename)}`;
}

/** Challenge social metrics. */
export interface ChallengeMetrics {
  acceptance_rate: number;
  shortest_prompt: number;
  total_submissions: number;
}

/** GET /api/challenges/:id/metrics — social metrics. */
export async function fetchChallengeMetrics(
  challengeId: string
): Promise<ChallengeMetrics> {
  return apiFetch<ChallengeMetrics>(
    `${API_BASE}/challenges/${encodeURIComponent(challengeId)}/metrics`
  );
}

/** GET /api/health — health check. */
export async function healthCheck(): Promise<{ status: string }> {
  return apiFetch<{ status: string }>(`${API_BASE}/health`);
}
