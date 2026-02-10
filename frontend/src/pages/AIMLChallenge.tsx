import { useState, useEffect, useCallback, useRef } from "react";
import { useParams, Link } from "react-router-dom";
import {
  ArrowLeft,
  Brain,
  Loader2,
  Download,
  BarChart3,
  Package,
  Database,
  FileText,
  History,
  Send,
  Play,
  CheckCircle2,
  XCircle,
  StickyNote,
  Terminal,
  Clock,
  AlertTriangle,
  Trophy,
  MessageSquare,
  Bot,
  Square,
} from "lucide-react";
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable";
import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { getProblems } from "@/data/load-problems";
import ThemeToggle from "@/components/ThemeToggle";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import CodeEditor from "@/components/CodeEditor";
import {
  fetchMLChallengeDetails,
  fetchChallengeMetrics,
  runCode,
  submitCode,
  agentChat,
  getDatasetDownloadUrl,
  type ExecutionResult,
  type MLChallengeDetails,
  type ChallengeMetrics,
  type ChatTurn,
} from "@/services/mlApi";

// ---------------------------------------------------------------------------
// Constants & types
// ---------------------------------------------------------------------------

const difficultyColor: Record<string, string> = {
  Easy: "text-code-green",
  Medium: "text-code-yellow",
  Hard: "text-destructive",
};

type RightTab = "editor" | "notes" | "submissions" | "submit";

interface SubmissionRecord {
  timestamp: string;
  mode: "run" | "submit";
  result: ExecutionResult;
}

// ---------------------------------------------------------------------------
// Score bar component for grading results
// ---------------------------------------------------------------------------

const ScoreBar = ({
  label,
  score,
  maxScore,
  feedback,
}: {
  label: string;
  score: number;
  maxScore: number;
  feedback?: string[];
}) => {
  const pct = maxScore > 0 ? (score / maxScore) * 100 : 0;
  const color =
    pct >= 80 ? "bg-code-green" : pct >= 50 ? "bg-code-yellow" : "bg-destructive";

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs font-mono">
        <span className="text-code-foreground">{label}</span>
        <span className="text-code-muted">
          {score}/{maxScore}
        </span>
      </div>
      <div className="w-full h-2 rounded-full bg-code-border/40 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {feedback && feedback.length > 0 && (
        <ul className="space-y-0.5 mt-1">
          {feedback.map((f, i) => (
            <li key={i} className="text-xs text-code-muted leading-snug pl-3 relative">
              <span className="absolute left-0">·</span>
              {f}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main page component
// ---------------------------------------------------------------------------

const AIMLChallenge = () => {
  const { id } = useParams<{ id: string }>();

  // ── Load challenge list to find the matching challenge ────
  const {
    data: challenges = [],
    isLoading: listLoading,
    error: listError,
  } = useQuery({
    queryKey: ["challenges"],
    queryFn: getProblems,
  });

  const challenge = challenges.find(
    (c) => c.type === "AI-ML" && c.title === id
  );
  const backendId = challenge?.backendId;

  // ── Fetch live details from backend ──────────────────────
  const {
    data: backendDetails,
    isLoading: detailsLoading,
  } = useQuery<MLChallengeDetails>({
    queryKey: ["mlChallengeDetails", backendId],
    queryFn: () => fetchMLChallengeDetails(backendId!),
    enabled: !!backendId,
    retry: 1,
    staleTime: 5 * 60 * 1000,
  });

  // ── Fetch social metrics ──────────────────────────────────
  const { data: metrics } = useQuery<ChallengeMetrics>({
    queryKey: ["challengeMetrics", backendId],
    queryFn: () => fetchChallengeMetrics(backendId!),
    enabled: !!backendId,
    retry: 1,
    staleTime: 5 * 60 * 1000,
  });

  // ── UI state ─────────────────────────────────────────────
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [codeTheme, setCodeTheme] = useState<"dark" | "light">("light");
  const [rightTab, setRightTab] = useState<RightTab>("editor");
  type LeftTab = "description" | "examples" | "rubric" | "data";
  const [leftTab, setLeftTab] = useState<LeftTab>("description");
  const [code, setCode] = useState("");
  const [codeInitialised, setCodeInitialised] = useState(false);
  const [notes, setNotes] = useState("");
  const [prompt, setPrompt] = useState("");

  // Execution state
  const [isRunning, setIsRunning] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [runResult, setRunResult] = useState<ExecutionResult | null>(null);
  const [submitResult, setSubmitResult] = useState<ExecutionResult | null>(null);
  const [execError, setExecError] = useState<string | null>(null);

  // Agent state
  const [isAgentWorking, setIsAgentWorking] = useState(false);
  const [agentMessage, setAgentMessage] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatTurn[]>([]);

  // Past submissions history (kept in memory)
  const [submissions, setSubmissions] = useState<SubmissionRecord[]>([]);

  // Ref for scrolling results into view
  const resultsRef = useRef<HTMLDivElement>(null);

  // ── Initialize code from backend starter code ────────────
  useEffect(() => {
    if (!codeInitialised && backendDetails?.starter_code) {
      setCode(backendDetails.starter_code);
      setCodeInitialised(true);
    } else if (!codeInitialised && challenge?.codeStub && !backendDetails) {
      // Fallback to static stub if backend not available yet
      setCode(challenge.codeStub);
    }
  }, [backendDetails, challenge, codeInitialised]);

  // ── Handlers ─────────────────────────────────────────────

  const handleRun = useCallback(async () => {
    if (!backendId || !code.trim()) return;
    setIsRunning(true);
    setExecError(null);
    setRunResult(null);
    setRightTab("submit");

    try {
      const result = await runCode(backendId, code);
      setRunResult(result);
      setSubmissions((prev) => [
        { timestamp: new Date().toISOString(), mode: "run", result },
        ...prev,
      ]);
    } catch (err) {
      setExecError(err instanceof Error ? err.message : "Run failed");
    } finally {
      setIsRunning(false);
    }
  }, [backendId, code]);

  const handleSubmit = useCallback(async () => {
    if (!backendId || !code.trim()) return;
    setIsSubmitting(true);
    setExecError(null);
    setSubmitResult(null);
    setRightTab("submit");

    try {
      const result = await submitCode(backendId, code);
      setSubmitResult(result);
      setSubmissions((prev) => [
        { timestamp: new Date().toISOString(), mode: "submit", result },
        ...prev,
      ]);
    } catch (err) {
      setExecError(err instanceof Error ? err.message : "Submission failed");
    } finally {
      setIsSubmitting(false);
    }
  }, [backendId, code]);

  const handleAgentSend = useCallback(async () => {
    if (!backendId || !prompt.trim()) return;
    setIsAgentWorking(true);
    setAgentMessage(null);

    const userTurn: ChatTurn = { role: "user", content: prompt };
    setChatHistory((prev) => [...prev, userTurn]);
    const currentPrompt = prompt;
    setPrompt("");

    try {
      const resp = await agentChat(backendId, currentPrompt, code, chatHistory);

      if (resp.error) {
        setAgentMessage(`Error: ${resp.error}`);
      } else {
        if (resp.code) {
          setCode(resp.code);
        }
        setAgentMessage(resp.message || "Code updated in editor.");
        setChatHistory((prev) => [
          ...prev,
          { role: "assistant", content: resp.message || resp.code },
        ]);
      }
    } catch (err) {
      setAgentMessage(
        `Agent error: ${err instanceof Error ? err.message : "Unknown error"}`
      );
    } finally {
      setIsAgentWorking(false);
    }
  }, [backendId, prompt, code, chatHistory]);

  const handlePromptKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        handleAgentSend();
      }
    },
    [handleAgentSend]
  );

  // ── Loading / error states ───────────────────────────────

  if (listLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading challenge...</p>
        </div>
      </div>
    );
  }

  if (listError || !challenge) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">
            {listError ? "Failed to load challenge" : "Challenge not found"}
          </h1>
          <Link to="/challenges" className="text-primary hover:underline">
            &larr; Back to challenges
          </Link>
        </div>
      </div>
    );
  }

  // Merge static challenge data with live backend details
  const description =
    backendDetails?.description || challenge.problemStatement || challenge.description;
  const starterCode = backendDetails?.starter_code || challenge.codeStub || "# Write your solution here\n\npass";
  const allowedLibs = backendDetails?.allowed_libraries || challenge.allowedLibraries || [];
  const scoring = backendDetails?.scoring;
  const dataset = backendDetails?.dataset;

  // Build dataset download links from backend
  const trainFile = dataset?.train_file;

  const activeResult = submitResult || runResult;
  const isBusy = isRunning || isSubmitting;

  // ── Right panel tabs definition ──────────────────────────

  const rightTabs: { id: RightTab; label: string; icon: React.ReactNode }[] = [
    { id: "editor", label: "Editor", icon: <FileText className="w-3.5 h-3.5" /> },
    { id: "notes", label: "Notes", icon: <StickyNote className="w-3.5 h-3.5" /> },
    {
      id: "submissions",
      label: "Past Submissions",
      icon: <History className="w-3.5 h-3.5" />,
    },
    {
      id: "submit",
      label: "Submit & Results",
      icon: <Send className="w-3.5 h-3.5" />,
    },
  ];

  // ── Render ───────────────────────────────────────────────

  return (
    <div
      className={`h-screen flex flex-col bg-code text-code-foreground overflow-hidden ${
        codeTheme === "light" ? "code-light" : ""
      }`}
    >
      {/* Top bar */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-code-border bg-code shrink-0">
        <div className="flex items-center gap-3">
          <Link
            to="/challenges"
            className="flex items-center gap-1.5 text-xs text-code-muted hover:text-code-foreground transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Challenges
          </Link>
          <span className="text-code-border">|</span>
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-code-accent" />
            <span className="text-sm font-medium">{challenge.title}</span>
          </div>
          <span
            className={`text-xs font-mono font-semibold ${difficultyColor[challenge.difficulty]}`}
          >
            {challenge.difficulty}
          </span>
          {detailsLoading && (
            <Loader2 className="w-3 h-3 animate-spin text-code-muted" />
          )}
        </div>
        <div className="flex items-center gap-2">
          <ThemeToggle
            codeTheme={codeTheme}
            onToggle={() =>
              setCodeTheme(codeTheme === "dark" ? "light" : "dark")
            }
          />
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="flex items-center gap-1.5 text-xs text-code-muted hover:text-code-foreground transition-colors"
          >
            {sidebarOpen ? "Hide" : "Show"} Description
          </button>
        </div>
      </div>

      {/* Main content */}
      <ResizablePanelGroup direction="horizontal" className="flex-1 min-h-0">
        {/* Left panel — challenge description */}
        {sidebarOpen && (
          <>
            <ResizablePanel defaultSize={35} minSize={20} maxSize={60}>
              <div className="h-full flex flex-col bg-code">
                {/* Left panel tab bar */}
                <div className="flex border-b border-code-border shrink-0">
                  {([
                    { id: "description" as LeftTab, label: "Description", icon: <Brain className="w-3.5 h-3.5" /> },
                    { id: "examples" as LeftTab, label: "Examples", icon: <Database className="w-3.5 h-3.5" /> },
                    { id: "rubric" as LeftTab, label: "Rubric", icon: <BarChart3 className="w-3.5 h-3.5" /> },
                    { id: "data" as LeftTab, label: "Data", icon: <Download className="w-3.5 h-3.5" /> },
                  ]).map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setLeftTab(tab.id)}
                      className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-mono transition-colors border-b-2 ${
                        leftTab === tab.id
                          ? "text-code-foreground border-primary"
                          : "text-code-muted border-transparent hover:text-code-foreground"
                      }`}
                    >
                      {tab.icon}
                      {tab.label}
                    </button>
                  ))}
                </div>

                {/* Scrollable tab content */}
                <div className="flex-1 overflow-auto">
                  <div className="px-5 py-4">
                    {/* Description tab */}
                    {leftTab === "description" && (
                      <div className="text-sm text-code-foreground/80 leading-relaxed">
                        <MarkdownRenderer content={description} />
                      </div>
                    )}

                    {/* Examples tab */}
                    {leftTab === "examples" && (
                      <>
                        {challenge.datasetSamples &&
                        challenge.datasetSamples.length > 0 ? (
                          <div className="space-y-3">
                            {challenge.datasetSamples.map((sample, i) => (
                              <div
                                key={i}
                                className="rounded-lg bg-code-border/30 p-3.5 space-y-2"
                              >
                                <div>
                                  <span className="text-xs font-mono text-code-blue mb-1 block">
                                    Features:
                                  </span>
                                  <pre className="text-xs font-mono text-code-foreground whitespace-pre-wrap">
                                    {sample.features}
                                  </pre>
                                </div>
                                <div>
                                  <span className="text-xs font-mono text-code-green mb-1 block">
                                    Label:
                                  </span>
                                  <pre className="text-xs font-mono text-code-foreground whitespace-pre-wrap">
                                    {sample.label}
                                  </pre>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="space-y-3">
                            {challenge.testCases.map((tc, i) => (
                              <div
                                key={i}
                                className="rounded-lg bg-code-border/30 p-3.5 space-y-2"
                              >
                                <div>
                                  <span className="text-xs font-mono text-code-blue mb-1 block">
                                    Input:
                                  </span>
                                  <pre className="text-xs font-mono text-code-foreground whitespace-pre-wrap">
                                    {tc.input}
                                  </pre>
                                </div>
                                <div>
                                  <span className="text-xs font-mono text-code-green mb-1 block">
                                    Output:
                                  </span>
                                  <pre className="text-xs font-mono text-code-foreground whitespace-pre-wrap">
                                    {tc.output}
                                  </pre>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    )}

                    {/* Rubric tab */}
                    {leftTab === "rubric" && (
                      <>
                        {challenge.gradingRubric ? (
                          <div className="text-sm text-code-foreground/80 leading-relaxed">
                            <MarkdownRenderer content={challenge.gradingRubric} />
                          </div>
                        ) : scoring ? (
                          <div className="space-y-2 text-sm text-code-foreground/80">
                            <p>
                              Pass threshold: {scoring.pass_threshold ?? "\u2014"}/
                              {scoring.max_score ?? 100}
                            </p>
                            {scoring.categories?.map((cat, i) => (
                              <div key={i} className="text-xs text-code-muted">
                                {cat.name}: {cat.max_score} pts
                                {cat.description ? ` \u2014 ${cat.description}` : ""}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="space-y-2">
                            <p className="text-sm text-code-foreground/80">
                              Target: {challenge.targetMetrics}
                            </p>
                          </div>
                        )}
                      </>
                    )}

                    {/* Data tab — libraries, dataset info, downloads, deliverables */}
                    {leftTab === "data" && (
                      <div className="space-y-6">
                        {/* Allowed Libraries */}
                        {allowedLibs.length > 0 && (
                          <div>
                            <h3 className="text-xs font-mono font-semibold text-code-foreground uppercase tracking-wider mb-3">
                              Allowed Libraries
                            </h3>
                            <div className="flex gap-2 flex-wrap">
                              {allowedLibs.map((lib) => (
                                <Badge
                                  key={lib}
                                  variant="outline"
                                  className="text-xs font-mono"
                                >
                                  {lib}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Dataset Downloads */}
                        <div>
                          <h3 className="text-xs font-mono font-semibold text-code-foreground uppercase tracking-wider mb-3">
                            Dataset
                          </h3>
                          <div className="space-y-3">
                            {backendId && trainFile && (
                              <div>
                                <span className="text-xs font-mono text-code-muted block mb-1">
                                  Download Training Data
                                </span>
                                <a
                                  href={getDatasetDownloadUrl(backendId, trainFile)}
                                  download
                                  className="text-xs font-mono text-code-blue hover:underline break-all"
                                >
                                  {trainFile}
                                </a>
                              </div>
                            )}
                            {dataset?.columns && dataset.columns.length > 0 && (
                              <div>
                                <span className="text-xs font-mono text-code-muted block mb-2">
                                  Columns
                                </span>
                                <div className="space-y-1">
                                  {dataset.columns.map((col, i) => (
                                    <div
                                      key={i}
                                      className="text-xs font-mono text-code-foreground/80"
                                    >
                                      <span className="text-code-blue">{col.name}</span>
                                      {col.type && (
                                        <span className="text-code-muted">
                                          {" "}({col.type})
                                        </span>
                                      )}
                                      {col.description && (
                                        <span className="text-code-muted">
                                          {" "}\u2014 {col.description}
                                        </span>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                            {!trainFile && challenge.dataDownloadUrl && (
                              <div>
                                <span className="text-xs font-mono text-code-muted block mb-1">
                                  Dataset Reference
                                </span>
                                <a
                                  href={challenge.dataDownloadUrl}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-xs font-mono text-code-blue hover:underline break-all"
                                >
                                  {challenge.dataDownloadUrl}
                                </a>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Deliverables */}
                        <div>
                          <h3 className="text-xs font-mono font-semibold text-code-foreground uppercase tracking-wider mb-3">
                            Deliverables
                          </h3>
                          <p className="text-sm text-code-foreground/80 leading-relaxed">
                            {challenge.deliverables ||
                              "A working solution that meets the grading criteria."}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Acceptance Rate — bottom of scrollable area */}
                  <div className="px-5 py-4">
                    <div className="flex items-center justify-between text-xs text-code-muted mb-2">
                      <span>Acceptance Rate</span>
                      <span className="font-mono">{metrics?.acceptance_rate ?? challenge?.acceptance ?? 0}%</span>
                    </div>
                    <div className="w-full h-1.5 rounded-full bg-code-border overflow-hidden">
                      <div
                        className="h-full rounded-full bg-code-green/60"
                        style={{ width: `${metrics?.acceptance_rate ?? challenge?.acceptance ?? 0}%` }}
                      />
                    </div>
                  </div>

                  {/* Prompt Golf card — bottom of scrollable area */}
                  <div className="px-5 py-4 border-t border-code-border">
                    <div className="rounded-lg bg-code-border/20 border border-code-border p-4 space-y-3">
                      <div className="flex items-center gap-2">
                        <Trophy className="w-4 h-4 text-code-yellow" />
                        <span className="text-xs font-mono font-semibold text-code-foreground">
                          Prompt Golf
                        </span>
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-code-muted flex items-center gap-1.5">
                            <Trophy className="w-3 h-3 text-code-yellow" />
                            Shortest passing prompt
                          </span>
                          <span className="font-mono font-semibold text-code-green">
                            {metrics?.shortest_prompt ?? "\u2014"} chars
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-code-muted flex items-center gap-1.5">
                            <MessageSquare className="w-3 h-3 text-code-blue" />
                            Your current prompt
                          </span>
                          <span className={`font-mono font-semibold ${
                            prompt.length === 0
                              ? "text-code-muted"
                              : prompt.length <= (metrics?.shortest_prompt ?? 999)
                              ? "text-code-green"
                              : "text-code-foreground"
                          }`}>
                            {prompt.length} chars
                          </span>
                        </div>
                        {prompt.length > 0 && (
                          <div className="w-full h-1.5 rounded-full bg-code-border overflow-hidden mt-1">
                            <div
                              className={`h-full rounded-full transition-all duration-300 ${
                                prompt.length <= (metrics?.shortest_prompt ?? 999) ? "bg-code-green" : "bg-code-blue"
                              }`}
                              style={{
                                width: `${Math.min(100, ((metrics?.shortest_prompt ?? 80) / Math.max(prompt.length, 1)) * 100)}%`,
                              }}
                            />
                          </div>
                        )}
                        {prompt.length > 0 && prompt.length <= (metrics?.shortest_prompt ?? 0) && (
                          <p className="text-[10px] text-code-green font-mono mt-1">
                            You're beating the record!
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </ResizablePanel>
            <ResizableHandle withHandle />
          </>
        )}

        {/* Right panel */}
        <ResizablePanel defaultSize={sidebarOpen ? 65 : 100} minSize={30}>
          <div className="h-full flex flex-col">
            {/* Right panel tabs + action buttons */}
            <div className="flex items-center border-b border-code-border shrink-0">
              <div className="flex flex-1">
                {rightTabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setRightTab(tab.id)}
                    className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-mono transition-colors border-b-2 ${
                      rightTab === tab.id
                        ? "text-code-foreground border-primary"
                        : "text-code-muted border-transparent hover:text-code-foreground"
                    }`}
                  >
                    {tab.icon}
                    {tab.label}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-1 pr-2">
                <button
                  onClick={handleRun}
                  disabled={isBusy || !backendId}
                  className="flex items-center gap-1.5 px-3 py-2 text-xs font-mono text-code-blue hover:text-code-blue/80 transition-colors disabled:opacity-40"
                  title="Quick test on sample data"
                >
                  <Play className="w-3 h-3" />
                  {isRunning ? "Running..." : "Run"}
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={isBusy || !backendId}
                  className="flex items-center gap-1.5 px-3 py-2 text-xs font-mono text-code-green hover:text-code-green/80 transition-colors disabled:opacity-40"
                  title="Full grading submission"
                >
                  <Send className="w-3 h-3" />
                  {isSubmitting ? "Submitting..." : "Submit"}
                </button>
              </div>
            </div>

            {/* Tab content */}
            <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
              {/* ──────── EDITOR TAB ──────── */}
              {rightTab === "editor" && (
                <div className="flex flex-col h-full overflow-hidden">
                  {/* File bar */}
                  <div className="flex items-center px-5 py-2 border-b border-code-border shrink-0">
                    <div className="flex items-center gap-1.5">
                      <span className="w-3 h-3 rounded-full bg-destructive/60" />
                      <span className="w-3 h-3 rounded-full bg-code-yellow/60" />
                      <span className="w-3 h-3 rounded-full bg-code-green/60" />
                    </div>
                    <span className="text-xs font-mono text-code-muted ml-3">
                      solution.py
                    </span>
                    {backendId && (
                      <button
                        onClick={() => {
                          setCode(starterCode);
                        }}
                        className="ml-auto text-xs font-mono text-code-muted hover:text-code-foreground transition-colors"
                      >
                        Reset to starter
                      </button>
                    )}
                  </div>

                  {/* Vertical split: code editor + agent prompt */}
                  <ResizablePanelGroup direction="vertical" className="flex-1 min-h-0">
                    {/* Code editor panel */}
                    <ResizablePanel defaultSize={75} minSize={30}>
                      <div className="h-full">
                        <CodeEditor
                          value={code}
                          onChange={setCode}
                          language="python"
                          theme={codeTheme === "dark" ? "vs-dark" : "light"}
                        />
                      </div>
                    </ResizablePanel>

                    <ResizableHandle withHandle />

                    {/* Agent prompt panel */}
                    <ResizablePanel defaultSize={25} minSize={10} maxSize={60}>
                      <div className="h-full flex flex-col overflow-hidden">
                        {agentMessage && (
                          <div className="mx-4 mt-3 rounded-lg bg-code-border/30 p-3 text-xs font-mono text-code-foreground/80 leading-relaxed flex gap-2 shrink-0 overflow-auto max-h-24">
                            <Bot className="w-4 h-4 text-code-accent shrink-0 mt-0.5" />
                            <span>{agentMessage}</span>
                          </div>
                        )}
                        <div className="relative flex-1 p-4 min-h-0">
                          <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            onKeyDown={handlePromptKeyDown}
                            placeholder={
                              backendId
                                ? "Describe your approach — the AI agent will write code... (Ctrl+Enter to send)"
                                : "AI agent unavailable — no backend connection"
                            }
                            disabled={!backendId || isAgentWorking}
                            className="w-full h-full bg-code-border/30 text-code-foreground text-sm font-mono rounded-lg px-4 py-3 pr-12 resize-none placeholder:text-code-muted/60 focus:outline-none focus:ring-1 focus:ring-code-accent/50 disabled:opacity-50"
                          />
                          <button
                            onClick={handleAgentSend}
                            disabled={!backendId || isAgentWorking || !prompt.trim()}
                            className="absolute bottom-7 right-7 w-8 h-8 rounded-lg bg-primary flex items-center justify-center hover:opacity-90 transition-opacity disabled:opacity-40"
                          >
                            {isAgentWorking ? (
                              <Loader2 className="w-4 h-4 text-primary-foreground animate-spin" />
                            ) : (
                              <Send className="w-4 h-4 text-primary-foreground" />
                            )}
                          </button>
                        </div>
                      </div>
                    </ResizablePanel>
                  </ResizablePanelGroup>
                </div>
              )}

              {/* ──────── NOTES TAB ──────── */}
              {rightTab === "notes" && (
                <div className="flex-1 p-5 overflow-auto">
                  <textarea
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder="Your notes — planning, observations, agent output, etc..."
                    className="w-full h-full bg-code-border/20 text-code-foreground text-sm font-mono rounded-lg px-4 py-3 resize-none placeholder:text-code-muted/60 focus:outline-none focus:ring-1 focus:ring-code-accent/50 min-h-[400px]"
                  />
                </div>
              )}

              {/* ──────── PAST SUBMISSIONS TAB ──────── */}
              {rightTab === "submissions" && (
                <div className="p-5 overflow-auto">
                  <h3 className="text-xs font-mono text-code-muted uppercase tracking-wider mb-4">
                    Past Submissions
                  </h3>
                  {submissions.length === 0 ? (
                    <div className="text-sm text-code-muted text-center py-12">
                      No submissions yet. Run or submit your solution to see
                      results here.
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {submissions.map((sub, i) => (
                        <div
                          key={i}
                          className="rounded-lg bg-code-border/20 p-3 space-y-1"
                        >
                          <div className="flex items-center gap-2 text-xs font-mono">
                            {sub.result.success ? (
                              <CheckCircle2 className="w-3.5 h-3.5 text-code-green" />
                            ) : (
                              <XCircle className="w-3.5 h-3.5 text-destructive" />
                            )}
                            <span className="text-code-foreground uppercase">
                              {sub.mode}
                            </span>
                            <span className="text-code-muted">
                              {new Date(sub.timestamp).toLocaleTimeString()}
                            </span>
                            {sub.result.grading && (
                              <span
                                className={
                                  sub.result.grading.passed
                                    ? "text-code-green"
                                    : "text-code-yellow"
                                }
                              >
                                {sub.result.grading.total_score}/
                                {sub.result.grading.max_score}
                                {sub.result.grading.passed ? " PASSED" : ""}
                              </span>
                            )}
                            {sub.result.sample_accuracy != null && (
                              <span className="text-code-blue">
                                Sample: {(sub.result.sample_accuracy * 100).toFixed(1)}%
                              </span>
                            )}
                            <span className="text-code-muted ml-auto">
                              {sub.result.execution_time.toFixed(1)}s
                            </span>
                          </div>
                          {sub.result.error && (
                            <p className="text-xs text-destructive font-mono truncate">
                              {sub.result.error}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* ──────── SUBMIT & RESULTS TAB ──────── */}
              {rightTab === "submit" && (
                <div className="p-5 space-y-4 overflow-auto" ref={resultsRef}>
                  <h3 className="text-xs font-mono text-code-muted uppercase tracking-wider">
                    Evaluation Results
                  </h3>

                  {/* Loading spinner */}
                  {isBusy && (
                    <div className="flex items-center gap-3 text-sm text-code-muted py-8 justify-center">
                      <Loader2 className="w-5 h-5 animate-spin" />
                      {isSubmitting
                        ? "Grading your submission (this may take a few minutes)..."
                        : "Running quick test..."}
                    </div>
                  )}

                  {/* Error */}
                  {execError && !isBusy && (
                    <div className="rounded-lg bg-destructive/10 p-4 flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
                      <div className="text-sm font-mono text-destructive">
                        {execError}
                      </div>
                    </div>
                  )}

                  {/* Results display */}
                  {activeResult && !isBusy && (
                    <div className="space-y-4">
                      {/* Timed out warning */}
                      {activeResult.timed_out && (
                        <div className="rounded-lg bg-code-yellow/10 p-3 flex items-center gap-2 text-xs font-mono text-code-yellow">
                          <Clock className="w-4 h-4" />
                          Execution timed out
                        </div>
                      )}

                      {/* Execution error */}
                      {activeResult.error && (
                        <div className="rounded-lg bg-destructive/10 p-3 flex items-start gap-2">
                          <XCircle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
                          <div className="text-xs font-mono text-destructive whitespace-pre-wrap">
                            {activeResult.error}
                          </div>
                        </div>
                      )}

                      {/* Traceback */}
                      {activeResult.traceback && (
                        <details className="group">
                          <summary className="text-xs font-mono text-code-muted cursor-pointer hover:text-code-foreground">
                            Show traceback
                          </summary>
                          <pre className="mt-2 text-xs font-mono text-destructive/80 whitespace-pre-wrap bg-code-border/20 rounded-lg p-3 max-h-48 overflow-auto">
                            {activeResult.traceback}
                          </pre>
                        </details>
                      )}

                      {/* Run mode: sample accuracy */}
                      {activeResult.mode === "run" &&
                        activeResult.sample_accuracy != null && (
                          <div className="rounded-lg bg-code-blue/10 p-4 space-y-2">
                            <div className="flex items-center gap-2 text-sm font-mono">
                              <Play className="w-4 h-4 text-code-blue" />
                              <span className="text-code-foreground font-semibold">
                                Quick Test Result
                              </span>
                            </div>
                            <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                              <div>
                                <span className="text-code-muted block">
                                  Sample Accuracy
                                </span>
                                <span className="text-code-blue text-xl font-bold">
                                  {(activeResult.sample_accuracy * 100).toFixed(1)}%
                                </span>
                              </div>
                              {activeResult.train_time != null && (
                                <div>
                                  <span className="text-code-muted block">
                                    Train Time
                                  </span>
                                  <span className="text-code-foreground text-xl font-bold">
                                    {activeResult.train_time.toFixed(1)}s
                                  </span>
                                </div>
                              )}
                            </div>
                            <p className="text-xs text-code-muted">
                              This is a quick test on a sample. Submit for full
                              grading.
                            </p>
                          </div>
                        )}

                      {/* Submit mode: full grading */}
                      {activeResult.mode === "submit" &&
                        activeResult.grading && (
                          <div className="space-y-4">
                            {/* Overall score banner */}
                            <div
                              className={`rounded-lg p-4 flex items-center gap-3 ${
                                activeResult.grading.passed
                                  ? "bg-code-green/10"
                                  : "bg-code-yellow/10"
                              }`}
                            >
                              {activeResult.grading.passed ? (
                                <Trophy className="w-6 h-6 text-code-green" />
                              ) : (
                                <AlertTriangle className="w-6 h-6 text-code-yellow" />
                              )}
                              <div>
                                <div className="text-xl font-mono font-bold text-code-foreground">
                                  {activeResult.grading.total_score}/
                                  {activeResult.grading.max_score}
                                </div>
                                <div
                                  className={`text-xs font-mono font-semibold ${
                                    activeResult.grading.passed
                                      ? "text-code-green"
                                      : "text-code-yellow"
                                  }`}
                                >
                                  {activeResult.grading.passed
                                    ? "PASSED"
                                    : "NOT YET PASSING"}
                                </div>
                              </div>
                              {/* Overall progress bar */}
                              <div className="flex-1 ml-4">
                                <div className="w-full h-3 rounded-full bg-code-border/40 overflow-hidden">
                                  <div
                                    className={`h-full rounded-full transition-all duration-700 ${
                                      activeResult.grading.passed
                                        ? "bg-code-green"
                                        : "bg-code-yellow"
                                    }`}
                                    style={{
                                      width: `${
                                        (activeResult.grading.total_score /
                                          activeResult.grading.max_score) *
                                        100
                                      }%`,
                                    }}
                                  />
                                </div>
                              </div>
                            </div>

                            {/* Per-category breakdown */}
                            <div className="space-y-3">
                              <h4 className="text-xs font-mono text-code-muted uppercase tracking-wider">
                                Category Breakdown
                              </h4>
                              {activeResult.grading.categories.map(
                                (cat, i) => (
                                  <ScoreBar
                                    key={i}
                                    label={cat.name}
                                    score={cat.score}
                                    maxScore={cat.max_score}
                                    feedback={cat.feedback}
                                  />
                                )
                              )}
                            </div>
                          </div>
                        )}

                      {/* Execution time */}
                      {activeResult.execution_time > 0 && (
                        <div className="flex items-center gap-2 text-xs font-mono text-code-muted">
                          <Clock className="w-3 h-3" />
                          Execution time:{" "}
                          {activeResult.execution_time.toFixed(1)}s
                        </div>
                      )}

                      {/* Stdout / Stderr */}
                      {(activeResult.stdout || activeResult.stderr) && (
                        <details className="group">
                          <summary className="flex items-center gap-1.5 text-xs font-mono text-code-muted cursor-pointer hover:text-code-foreground">
                            <Terminal className="w-3 h-3" />
                            Console Output
                          </summary>
                          <div className="mt-2 space-y-2">
                            {activeResult.stdout && (
                              <pre className="text-xs font-mono text-code-foreground/80 whitespace-pre-wrap bg-code-border/20 rounded-lg p-3 max-h-64 overflow-auto">
                                {activeResult.stdout}
                              </pre>
                            )}
                            {activeResult.stderr && (
                              <pre className="text-xs font-mono text-code-yellow/80 whitespace-pre-wrap bg-code-border/20 rounded-lg p-3 max-h-48 overflow-auto">
                                {activeResult.stderr}
                              </pre>
                            )}
                          </div>
                        </details>
                      )}

                      {/* Clear */}
                      <button
                        onClick={() => {
                          setRunResult(null);
                          setSubmitResult(null);
                          setExecError(null);
                        }}
                        className="text-xs text-code-muted hover:text-code-foreground transition-colors"
                      >
                        Clear results
                      </button>
                    </div>
                  )}

                  {/* Empty state */}
                  {!activeResult && !isBusy && !execError && (
                    <div className="text-sm text-code-muted text-center py-12">
                      Click{" "}
                      <span className="text-code-blue font-mono">Run</span> for
                      a quick test or{" "}
                      <span className="text-code-green font-mono">Submit</span>{" "}
                      for full grading.
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
};

export default AIMLChallenge;
