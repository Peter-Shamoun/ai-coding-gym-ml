import { useState, useCallback } from "react";
import { useParams, Link } from "react-router-dom";
import {
  ArrowLeft,
  Bug,
  Tag,
  FileCode,
  AlertTriangle,
  GitBranch,
  Bot,
  Terminal,
  FolderOpen,
  Puzzle,
  CheckCircle2,
  Copy,
  Check,
  ChevronDown,
  XCircle,
  ClipboardCheck,
  Loader2,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { getProblems } from "@/data/load-problems";
import ThemeToggle from "@/components/ThemeToggle";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import { useAuth } from "@/hooks/use-auth";
import { getCurrentUser, fetchUserIdFromBackend } from "@/lib/auth";

const difficultyColor: Record<string, string> = {
  Easy: "text-code-green",
  Medium: "text-code-yellow",
  Hard: "text-destructive",
};

type LeftTab = "issue" | "context" | "agent";
type RightTab = "guide" | "results";

const CopyButton = ({ text }: { text: string }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button
      onClick={handleCopy}
      className="absolute top-2.5 right-2.5 p-1.5 rounded-md bg-code-border/50 hover:bg-code-border text-code-muted hover:text-code-foreground transition-colors"
    >
      {copied ? <Check className="w-3.5 h-3.5 text-code-green" /> : <Copy className="w-3.5 h-3.5" />}
    </button>
  );
};

const mcpJsonConfig = `{
  "mcpServers": {
    "ai-coding-gym": {
      "type": "stdio",
      "command": "ai-coding-gym-mcp",
      "args": []
    }
  }
}`;

const IDEAccordion = ({
  title,
  subtitle,
  defaultOpen = false,
  children,
}: {
  title: string;
  subtitle?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg bg-code-border/20 border border-code-border overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-code-border/10 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono font-semibold text-code-blue">{title}</span>
          {subtitle && <span className="text-[10px] text-code-muted">{subtitle}</span>}
        </div>
        <ChevronDown
          className={`w-3.5 h-3.5 text-code-muted transition-transform duration-200 ${
            open ? "rotate-0" : "-rotate-90"
          }`}
        />
      </button>
      {open && <div className="px-4 pb-4 pt-1">{children}</div>}
    </div>
  );
};

const StepLine = ({ n, children }: { n: number | string; children: React.ReactNode }) => (
  <div className="flex items-start gap-2">
    <span className="text-[10px] text-code-muted font-mono mt-0.5 shrink-0 w-3 text-right">
      {n}.
    </span>
    <p className="text-xs text-code-foreground/70 leading-relaxed">{children}</p>
  </div>
);

const CommandBlock = ({ text }: { text: string }) => (
  <div className="relative">
    <div className="flex items-center gap-2 bg-code-border/30 rounded-lg px-4 py-3 border border-code-border">
      <Terminal className="w-3.5 h-3.5 text-code-green shrink-0" />
      <code className="text-xs font-mono text-code-foreground">{text}</code>
    </div>
    <CopyButton text={text} />
  </div>
);

const JsonConfigBlock = () => (
  <div className="relative">
    <pre className="text-xs font-mono text-code-foreground/90 whitespace-pre-wrap leading-relaxed bg-code-border/30 rounded-lg px-4 py-3 border border-code-border">
      {mcpJsonConfig}
    </pre>
    <CopyButton text={mcpJsonConfig} />
  </div>
);

const WorkflowStep = ({
  label,
  command,
  hint,
}: {
  label: string;
  command: string;
  hint?: string;
}) => (
  <div>
    <p className="text-xs font-semibold text-code-foreground mb-1.5">{label}</p>
    <CommandBlock text={command} />
    {hint && (
      <p className="text-[10px] text-code-muted leading-relaxed mt-2">{hint}</p>
    )}
  </div>
);

interface Param {
  name: string;
  required: boolean;
  desc: string;
  default?: string;
}

const ParamTable = ({ params }: { params: Param[] }) => (
  <div className="rounded-md border border-code-border overflow-hidden">
    <table className="w-full text-xs font-mono">
      <thead>
        <tr className="bg-code-border/30 text-code-muted">
          <th className="text-left px-3 py-1.5 font-medium">Parameter</th>
          <th className="text-left px-3 py-1.5 font-medium">Description</th>
        </tr>
      </thead>
      <tbody>
        {params.map((p) => (
          <tr key={p.name} className="border-t border-code-border/50">
            <td className="px-3 py-1.5 align-top whitespace-nowrap">
              <code className="text-code-accent">{p.name}</code>
              {p.required ? (
                <span className="ml-1 text-destructive text-[9px]">req</span>
              ) : (
                <span className="ml-1 text-code-muted text-[9px]">opt</span>
              )}
            </td>
            <td className="px-3 py-1.5 text-code-foreground/70">
              {p.desc}
              {p.default && (
                <span className="block text-[10px] text-code-muted mt-0.5">
                  Default: <code className="text-code-blue">{p.default}</code>
                </span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

// API types for submission / GitHub Actions result
interface TestResultFromApi {
  passed: boolean;
  totalTests: number;
  passedTests: number;
  failedTests: number;
  output?: string | null;
  error?: string | null;
  logs?: string | null;
}

interface SubmissionFromApi {
  id: string;
  userId: string;
  problemSlug: string;
  commitHash: string;
  branch: string;
  status: "PENDING" | "COMPLETED" | "FAILED";
  createdAt: string;
  evaluatedAt?: string | null;
  result?: TestResultFromApi | null;
}

const SWEBenchChallenge = () => {
  const { id } = useParams<{ id: string }>();
  const { user, isAuthenticated } = useAuth();
  const { data: challenges = [], isLoading, error } = useQuery({
    queryKey: ["challenges"],
    queryFn: getProblems,
  });

  // URL param is problem slug (e.g. astropy-12907)
  const challenge = challenges.find(
    (c) => c.type === "Human-SWE-Bench" && c.title === id
  );
  const [activeTab, setActiveTab] = useState<LeftTab>("issue");
  const [rightTab, setRightTab] = useState<RightTab>("guide");
  const [codeTheme, setCodeTheme] = useState<"dark" | "light">("light");
  const [submitting, setSubmitting] = useState(false);
  const [latestSubmission, setLatestSubmission] = useState<SubmissionFromApi | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [hasFetched, setHasFetched] = useState(false);

  const backendUrl = import.meta.env.VITE_BACKEND_URL || (import.meta.env.PROD ? '' : 'http://localhost:3001');
  const problemSlug = challenge?.title ?? challenge?.id ?? "";

  const handleCheckResults = useCallback(async () => {
    if (!challenge) return;
    const currentUser = getCurrentUser();
    let userId = currentUser?.userId;
    if (!userId && currentUser?.login) {
      userId = await fetchUserIdFromBackend(currentUser.login) ?? undefined;
    }
    if (!userId) {
      setFetchError("Please sign in to check results.");
      setRightTab("results");
      return;
    }
    setSubmitting(true);
    setFetchError(null);
    setLatestSubmission(null);
    setRightTab("results");
    try {
      const url = `${backendUrl}/api/submissions?userId=${encodeURIComponent(userId)}&problemSlug=${encodeURIComponent(problemSlug)}`;
      const res = await fetch(url);
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `Request failed: ${res.status}`);
      }
      const data = await res.json();
      const submissions: SubmissionFromApi[] = data.submissions ?? [];
      const latest = submissions[0] ?? null;
      setLatestSubmission(latest);
      setHasFetched(true);
    } catch (err) {
      setFetchError(err instanceof Error ? err.message : "Failed to fetch results.");
      setHasFetched(true);
    } finally {
      setSubmitting(false);
    }
  }, [challenge, problemSlug, backendUrl]);

  const result = latestSubmission?.result;
  const passedCount = result?.passedTests ?? 0;
  const totalCount = result?.totalTests ?? 0;
  const allPassed = (result?.passed === true) && totalCount > 0;
  const hasResult = latestSubmission != null;
  const isPending = latestSubmission?.status === "PENDING";
  const hasEvaluatedResult = latestSubmission?.status !== "PENDING" && result != null;

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading challenge...</p>
        </div>
      </div>
    );
  }

  if (error || !challenge) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">
            {error ? "Failed to load challenge" : "Challenge not found"}
          </h1>
          {error && (
            <p className="text-muted-foreground mb-4">
              {error instanceof Error ? error.message : "An unknown error occurred"}
            </p>
          )}
          <Link to="/challenges" className="text-primary hover:underline">
            ← Back to challenges
          </Link>
        </div>
      </div>
    );
  }

  const leftTabs: { id: LeftTab; label: string; icon: React.ReactNode }[] = [
    { id: "issue", label: "Issue", icon: <AlertTriangle className="w-3.5 h-3.5" /> },
    { id: "context", label: "Context", icon: <FileCode className="w-3.5 h-3.5" /> },
    { id: "agent", label: "AI Attempt", icon: <Bot className="w-3.5 h-3.5" /> },
  ];

  return (
    <div className={`h-screen flex flex-col bg-code text-code-foreground overflow-hidden ${codeTheme === "light" ? "code-light" : ""}`}>
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
            <Bug className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium">{challenge.title}</span>
          </div>
          <span
            className={`text-xs font-mono font-semibold ${difficultyColor[challenge.difficulty]}`}
          >
            {challenge.difficulty}
          </span>
        </div>
        <ThemeToggle codeTheme={codeTheme} onToggle={() => setCodeTheme(codeTheme === "dark" ? "light" : "dark")} />
      </div>

      {/* Main content — two panels side by side */}
      <div className="flex flex-1 min-h-0">
        {/* ─── LEFT: Issue & Data ─── */}
        <div className="w-1/2 border-r border-code-border flex flex-col overflow-hidden">
          {/* Tabs */}
          <div className="flex border-b border-code-border shrink-0">
            {leftTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-mono transition-colors border-b-2 ${
                  activeTab === tab.id
                    ? "text-code-foreground border-primary"
                    : "text-code-muted border-transparent hover:text-code-foreground"
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-auto">
            {activeTab === "issue" && (
              <div className="p-5">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-xs font-mono text-primary uppercase tracking-wider">
                    Human-SWE-Bench
                  </span>
                </div>
                <h2 className="text-lg font-semibold mb-3">{challenge.title}</h2>
                {challenge.problemStatement ? (
                  <div className="text-sm text-code-foreground/80 leading-relaxed mb-4 prose prose-invert max-w-none">
                    <MarkdownRenderer content={challenge.problemStatement} />
                  </div>
                ) : (
                  <p className="text-sm text-code-foreground/80 leading-relaxed mb-4">
                    {challenge.description}
                  </p>
                )}

                <div className="flex gap-2 mb-5 flex-wrap">
                  {challenge.tags.map((tag) => (
                    <Badge
                      key={tag}
                      variant="secondary"
                      className="text-[10px] bg-code-border/50 text-code-muted border-none"
                    >
                      <Tag className="w-3 h-3 mr-1" />
                      {tag}
                    </Badge>
                  ))}
                </div>

                {challenge.issueBody && (
                  <div className="rounded-lg bg-code-border/20 border border-code-border p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <GitBranch className="w-4 h-4 text-code-muted" />
                      <span className="text-xs font-mono text-code-muted">Issue #347</span>
                    </div>
                    <pre className="text-xs font-mono text-code-foreground/80 whitespace-pre-wrap leading-relaxed">
                      {challenge.issueBody}
                    </pre>
                  </div>
                )}

                <div className="mt-5">
                  <h3 className="text-xs font-mono text-code-muted uppercase tracking-wider mb-3">
                    Expected Behavior
                  </h3>
                  <div className="space-y-2">
                    {challenge.testCases.map((tc, i) => (
                      <div key={i} className="rounded-lg bg-code-border/30 p-3 space-y-1.5">
                        <div>
                          <span className="text-[10px] font-mono text-code-blue block">Scenario:</span>
                          <span className="text-xs font-mono">{tc.input}</span>
                        </div>
                        <div>
                          <span className="text-[10px] font-mono text-code-green block">Expected:</span>
                          <span className="text-xs font-mono">{tc.output}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === "context" && (
              <div className="p-5">
                <h3 className="text-xs font-mono text-code-muted uppercase tracking-wider mb-4">
                  Repository Structure
                </h3>
                {challenge.repoContext ? (
                  <pre className="text-xs font-mono text-code-foreground/80 whitespace-pre-wrap leading-loose bg-code-border/20 rounded-lg p-4 border border-code-border">
                    {challenge.repoContext}
                  </pre>
                ) : (
                  <p className="text-sm text-code-muted">No repo context available.</p>
                )}

                {challenge.failingTest && (
                  <div className="mt-6">
                    <h3 className="text-xs font-mono text-code-muted uppercase tracking-wider mb-3 flex items-center gap-2">
                      <AlertTriangle className="w-3.5 h-3.5 text-destructive" />
                      Failing Test
                    </h3>
                    <pre className="text-xs font-mono text-code-foreground/90 whitespace-pre-wrap leading-relaxed bg-destructive/5 rounded-lg p-4 border border-destructive/20">
                      {challenge.failingTest}
                    </pre>
                  </div>
                )}

                <div className="mt-6">
                  <div className="flex items-center justify-between text-xs text-code-muted mb-2">
                    <span>Acceptance Rate</span>
                    <span className="font-mono">{challenge.acceptance}%</span>
                  </div>
                  <div className="w-full h-1.5 rounded-full bg-code-border overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary/60"
                      style={{ width: `${challenge.acceptance}%` }}
                    />
                  </div>
                </div>
              </div>
            )}

            {activeTab === "agent" && (
              <div className="p-5">
                <h3 className="text-xs font-mono text-code-muted uppercase tracking-wider mb-3 flex items-center gap-2">
                  <Bot className="w-3.5 h-3.5 text-code-yellow" />
                  What AI Agents Tried
                </h3>
                <p className="text-sm text-code-foreground/70 mb-4 leading-relaxed">
                  Current AI agents attempted to solve this issue but failed.
                  Review their approach to understand the gap, then craft a better solution.
                </p>
                {challenge.agentAttempt ? (
                  <pre className="text-xs font-mono text-code-foreground/80 whitespace-pre-wrap leading-relaxed bg-code-yellow/5 rounded-lg p-4 border border-code-yellow/20">
                    {challenge.agentAttempt}
                  </pre>
                ) : (
                  <p className="text-sm text-code-muted">No agent attempt recorded.</p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ─── RIGHT: Tabbed Panel ─── */}
        <div className="w-1/2 flex flex-col overflow-hidden">
          {/* Right tabs */}
          <div className="flex border-b border-code-border shrink-0">
            <button
              onClick={() => setRightTab("guide")}
              className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-mono transition-colors border-b-2 ${
                rightTab === "guide"
                  ? "text-code-foreground border-primary"
                  : "text-code-muted border-transparent hover:text-code-foreground"
              }`}
            >
              <Puzzle className="w-3.5 h-3.5" />
              Getting Started
            </button>
            <button
              onClick={() => setRightTab("results")}
              className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-mono transition-colors border-b-2 ${
                rightTab === "results"
                  ? "text-code-foreground border-primary"
                  : "text-code-muted border-transparent hover:text-code-foreground"
              }`}
            >
              <ClipboardCheck className="w-3.5 h-3.5" />
              Check Results
              {hasEvaluatedResult && result && (
                <span className={`ml-1 text-[10px] font-semibold ${allPassed ? "text-code-green" : "text-code-yellow"}`}>
                  {passedCount}/{totalCount}
                </span>
              )}
            </button>
          </div>

          {/* Right tab content */}
          <div className="flex-1 overflow-auto">
            {rightTab === "guide" && (
              <div className="p-5 space-y-6">
                {/* Step 1 */}
                <div>
                  <div className="flex items-center gap-2.5 mb-3">
                    <span className="w-6 h-6 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-xs font-bold shrink-0">
                      1
                    </span>
                    <h3 className="text-sm font-semibold text-code-foreground">Installation</h3>
                  </div>
                  <p className="text-xs text-code-foreground/70 leading-relaxed mb-3 ml-8">
                    Install the AI Coding Gym MCP package using pip:
                  </p>
                  <div className="relative ml-8">
                    <div className="flex items-center gap-2 bg-code-border/30 rounded-lg px-4 py-3 border border-code-border">
                      <Terminal className="w-3.5 h-3.5 text-code-green shrink-0" />
                      <code className="text-xs font-mono text-code-foreground">
                        pip install ai-coding-gym-mcp
                      </code>
                    </div>
                    <CopyButton text="pip install ai-coding-gym-mcp" />
                  </div>
                </div>

                {/* Step 2 */}
                <div>
                  <div className="flex items-center gap-2.5 mb-3">
                    <span className="w-6 h-6 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-xs font-bold shrink-0">
                      2
                    </span>
                    <h3 className="text-sm font-semibold text-code-foreground">Prepare Your Workspace</h3>
                  </div>
                  <div className="ml-8 space-y-2">
                    <div className="flex items-start gap-2">
                      <FolderOpen className="w-3.5 h-3.5 text-code-blue mt-0.5 shrink-0" />
                      <p className="text-xs text-code-foreground/70 leading-relaxed">
                        Open your IDE and select (or create) a folder as your current workspace.
                      </p>
                    </div>
                    <div className="flex items-start gap-2">
                      <CheckCircle2 className="w-3.5 h-3.5 text-code-green mt-0.5 shrink-0" />
                      <p className="text-xs text-code-foreground/70 leading-relaxed">
                        This folder will be used to store fetched problems and solutions.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Step 3 */}
                <div>
                  <div className="flex items-center gap-2.5 mb-3">
                    <span className="w-6 h-6 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-xs font-bold shrink-0">
                      3
                    </span>
                    <h3 className="text-sm font-semibold text-code-foreground">
                      Add the Local MCP Server to Your IDE
                    </h3>
                  </div>

                  <div className="ml-8 space-y-2">
                    <IDEAccordion title="VS Code" subtitle="with GitHub Copilot extension" defaultOpen>
                      <div className="space-y-2.5">
                        <StepLine n="1">
                          Press <kbd className="px-1.5 py-0.5 rounded bg-code-border text-[10px] font-mono text-code-foreground">⌘/Ctrl + Shift + P</kbd> to open the Command Palette.
                        </StepLine>
                        <StepLine n="2">
                          Type and select: <code className="px-1.5 py-0.5 rounded bg-code-border text-[10px] font-mono text-code-accent">MCP: Add Server</code>
                        </StepLine>
                        <StepLine n="3">
                          Choose <span className="text-code-foreground font-medium">Command (stdio)</span>.
                        </StepLine>
                        <StepLine n="4">Enter the command:</StepLine>
                        <CommandBlock text="ai-coding-gym-mcp" />
                        <StepLine n="5">
                          Give the server a name (e.g., <code className="px-1.5 py-0.5 rounded bg-code-border text-[10px] font-mono text-code-accent">AICodingGym</code>).
                        </StepLine>
                        <StepLine n="6">
                          Choose whether to install globally or for this workspace.
                          <span className="block mt-1 text-code-green text-[10px]">✓ Workspace installation is recommended.</span>
                        </StepLine>
                      </div>
                    </IDEAccordion>

                    <IDEAccordion title="Cursor" subtitle="">
                      <div className="space-y-2.5">
                        <StepLine n="1">
                          Open <span className="text-code-foreground font-medium">Settings → Cursor Settings → Tools & MCP</span>.
                        </StepLine>
                        <StepLine n="2">
                          Select <span className="text-code-foreground font-medium">Add Custom MCP</span>.
                        </StepLine>
                        <StepLine n="3">Add the following configuration:</StepLine>
                        <JsonConfigBlock />
                      </div>
                    </IDEAccordion>

                    <IDEAccordion title="Claude Code" subtitle="">
                      <div className="space-y-2.5">
                        <StepLine n="1">Open the configuration file:</StepLine>
                        <CommandBlock text="~/.claude.json" />
                        <StepLine n="2">Add the following configuration:</StepLine>
                        <JsonConfigBlock />
                      </div>
                    </IDEAccordion>

                    <div className="flex items-start gap-2 pt-2">
                      <CheckCircle2 className="w-3.5 h-3.5 text-code-green mt-0.5 shrink-0" />
                      <p className="text-xs text-code-foreground/70 leading-relaxed">
                        After setup, confirm that MCP tools are enabled in your IDE or AI environment.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Step 4 – Gym Workflow */}
                <div>
                  <div className="flex items-center gap-2.5 mb-3">
                    <span className="w-6 h-6 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-xs font-bold shrink-0">
                      4
                    </span>
                    <h3 className="text-sm font-semibold text-code-foreground">Gym Workflow</h3>
                  </div>
                  <p className="text-xs text-code-foreground/70 leading-relaxed mb-4 ml-8">
                    All commands below are executed in chat with the AI tools.
                  </p>

                  <div className="ml-8 space-y-4">
                    <WorkflowStep
                      label="Configure your account"
                      command="/configure <user_id>"
                      hint='Get your user_id from the AI Coding Gym website: Sign in with your GitHub account → Dashboard → Account Information → User ID.'
                    />
                    <WorkflowStep
                      label="Fetch a problem"
                      command={`/fetch ${challenge.title}`}
                    />
                    <div>
                      <p className="text-xs font-semibold text-code-foreground mb-1.5">Fix the issue</p>
                      <p className="text-xs text-code-foreground/70 leading-relaxed">
                        Use AI programming tools inside your IDE to fix the bug described in{" "}
                        <code className="px-1.5 py-0.5 rounded bg-code-border text-[10px] font-mono text-code-accent">
                          problem_statement.md
                        </code>.
                      </p>
                    </div>
                    <WorkflowStep
                      label="Submit your solution"
                      command={`/submit ${challenge.title}`}
                    />
                    <div>
                      <p className="text-xs font-semibold text-code-foreground mb-1.5">Check results</p>
                      <p className="text-xs text-code-foreground/70 leading-relaxed">
                        Switch to the{" "}
                        <button
                          onClick={() => setRightTab("results")}
                          className="text-primary hover:underline font-semibold"
                        >
                          Check Results
                        </button>{" "}
                        tab to verify whether your solution passes all tests.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Step 5 – MCP Server Tools Reference */}
                <div>
                  <div className="flex items-center gap-2.5 mb-3">
                    <span className="w-6 h-6 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-xs font-bold shrink-0">
                      5
                    </span>
                    <h3 className="text-sm font-semibold text-code-foreground">MCP Server Tools</h3>
                  </div>
                  <p className="text-xs text-code-foreground/70 leading-relaxed mb-3 ml-8">
                    The AI Coding Gym MCP server provides three tools:
                  </p>

                  <div className="ml-8 space-y-2">
                    <IDEAccordion title="/configure" subtitle="Set up your account">
                      <div className="space-y-2">
                        <p className="text-xs text-code-foreground/70 leading-relaxed">
                          Configure the MCP server with your user ID. This step generates an SSH key pair and registers it with the backend server.
                        </p>
                        <ParamTable params={[
                          { name: "user_id", required: true, desc: "Your user ID for authentication" },
                          { name: "workspace_dir", required: false, desc: "Default workspace directory", default: "./workspace" },
                        ]} />
                      </div>
                    </IDEAccordion>

                    <IDEAccordion title="/fetch" subtitle="Download a problem">
                      <div className="space-y-2">
                        <p className="text-xs text-code-foreground/70 leading-relaxed">
                          Fetches a problem from the backend and clones the repository locally.
                        </p>
                        <ParamTable params={[
                          { name: "problem_id", required: true, desc: `Problem identifier (e.g., ${challenge.title})` },
                          { name: "user_id", required: false, desc: "Uses the configured user ID if not provided" },
                          { name: "workspace_dir", required: false, desc: "Local workspace directory", default: "./workspace" },
                        ]} />
                      </div>
                    </IDEAccordion>

                    <IDEAccordion title="/submit" subtitle="Push your solution">
                      <div className="space-y-2">
                        <p className="text-xs text-code-foreground/70 leading-relaxed">
                          Submit your solution by committing your changes and pushing them to the remote repository.
                        </p>
                        <ParamTable params={[
                          { name: "problem_id", required: true, desc: "Problem identifier" },
                          { name: "user_id", required: false, desc: "Uses the configured user ID if not provided" },
                          { name: "commit_message", required: false, desc: "Custom commit message" },
                        ]} />
                      </div>
                    </IDEAccordion>
                  </div>
                </div>
              </div>
            )}

            {rightTab === "results" && (
              <div className="p-5 space-y-5">
                {/* Check Results button */}
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-semibold text-code-foreground mb-1">Test Results</h3>
                    <p className="text-xs text-code-foreground/70">
                      After submitting via MCP, click to fetch results from GitHub Actions.
                    </p>
                  </div>
                  <button
                    onClick={handleCheckResults}
                    disabled={submitting}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-mono font-semibold hover:opacity-90 transition-opacity disabled:opacity-50 shrink-0"
                  >
                    {submitting ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <ClipboardCheck className="w-3.5 h-3.5" />
                    )}
                    {submitting ? "Checking…" : "Check Results"}
                  </button>
                </div>

                {/* Loading state */}
                {submitting && (
                  <div className="flex flex-col items-center justify-center py-12 text-code-muted">
                    <Loader2 className="w-6 h-6 animate-spin mb-3" />
                    <p className="text-xs font-mono">Fetching results…</p>
                  </div>
                )}

                {/* Fetch error */}
                {!submitting && fetchError && (
                  <div className="rounded-lg px-4 py-3 border bg-destructive/10 border-destructive/30">
                    <p className="text-xs font-mono text-destructive">{fetchError}</p>
                  </div>
                )}

                {/* No submissions yet (fetched, empty list) */}
                {!submitting && !fetchError && hasFetched && !latestSubmission && (
                  <div className="flex flex-col items-center justify-center py-16 text-code-muted">
                    <ClipboardCheck className="w-8 h-8 mb-3 opacity-30" />
                    <p className="text-xs font-mono mb-1">No submissions yet</p>
                    <p className="text-[10px] text-code-muted/60">Submit via MCP first, then click "Check Results".</p>
                  </div>
                )}

                {/* Pending: GitHub Actions running */}
                {!submitting && !fetchError && hasResult && isPending && (
                  <div className="rounded-lg px-4 py-4 border border-code-yellow/30 bg-code-yellow/10">
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 text-code-yellow animate-spin shrink-0" />
                      <p className="text-xs font-mono text-code-foreground">Tests are running (GitHub Actions). Check back in a moment.</p>
                    </div>
                    <p className="text-[10px] text-code-muted mt-2">Commit: {latestSubmission?.commitHash?.slice(0, 7)} · Branch: {latestSubmission?.branch}</p>
                  </div>
                )}

                {/* Evaluated result from webhook (GitHub Actions POST) */}
                {!submitting && !fetchError && hasEvaluatedResult && result && (
                  <div className="space-y-3">
                    <div className={`rounded-lg px-4 py-3 border ${
                      allPassed
                        ? "bg-code-green/10 border-code-green/30"
                        : "bg-destructive/10 border-destructive/30"
                    }`}>
                      <div className="flex items-center gap-2">
                        {allPassed ? (
                          <CheckCircle2 className="w-4 h-4 text-code-green shrink-0" />
                        ) : (
                          <XCircle className="w-4 h-4 text-destructive shrink-0" />
                        )}
                        <span className={`text-xs font-mono font-semibold ${
                          allPassed ? "text-code-green" : "text-destructive"
                        }`}>
                          {allPassed
                            ? "All tests passed! Challenge solved."
                            : `${result.failedTests ?? totalCount - passedCount} test(s) failed. Review and try again.`}
                        </span>
                        <span className={`text-xs font-mono font-semibold ml-auto ${
                          allPassed ? "text-code-green" : "text-code-yellow"
                        }`}>
                          {passedCount}/{totalCount}
                        </span>
                      </div>
                      {latestSubmission?.commitHash && (
                        <p className="text-[10px] text-code-muted mt-1.5">Commit {latestSubmission.commitHash.slice(0, 7)} · {latestSubmission.branch}</p>
                      )}
                    </div>

                    {/* Output / Logs from GitHub Actions */}
                    {(result.output || result.error || result.logs) && (
                      <IDEAccordion title="GitHub Actions output" subtitle="Logs" defaultOpen>
                        <div className="space-y-2">
                          {result.error && (
                            <div>
                              <p className="text-[10px] font-mono font-semibold text-destructive mb-1">Error</p>
                              <pre className="text-[10px] font-mono text-destructive/90 whitespace-pre-wrap break-words rounded bg-destructive/10 p-2 max-h-40 overflow-auto">
                                {result.error}
                              </pre>
                            </div>
                          )}
                          {result.output && (
                            <div>
                              <p className="text-[10px] font-mono font-semibold text-code-muted mb-1">Output</p>
                              <pre className="text-[10px] font-mono text-code-foreground/90 whitespace-pre-wrap break-words rounded bg-code-border/20 p-2 max-h-40 overflow-auto">
                                {result.output}
                              </pre>
                            </div>
                          )}
                          {result.logs && (
                            <div>
                              <p className="text-[10px] font-mono font-semibold text-code-muted mb-1">Logs</p>
                              <pre className="text-[10px] font-mono text-code-foreground/90 whitespace-pre-wrap break-words rounded bg-code-border/20 p-2 max-h-48 overflow-auto">
                                {result.logs}
                              </pre>
                            </div>
                          )}
                        </div>
                      </IDEAccordion>
                    )}
                  </div>
                )}

                {/* Has submission but no result yet (e.g. old submission without webhook) */}
                {!submitting && !fetchError && hasResult && latestSubmission && !isPending && !result && (
                  <div className="rounded-lg px-4 py-3 border border-code-border bg-code-border/10">
                    <p className="text-xs font-mono text-code-muted">No test result for this submission yet. Ensure GitHub Actions webhook is configured.</p>
                    <p className="text-[10px] text-code-muted mt-1">Commit {latestSubmission.commitHash?.slice(0, 7)} · {latestSubmission.branch}</p>
                  </div>
                )}

                {/* Idle: no fetch done yet */}
                {!submitting && !fetchError && !hasFetched && (
                  <div className="flex flex-col items-center justify-center py-16 text-code-muted">
                    <ClipboardCheck className="w-8 h-8 mb-3 opacity-30" />
                    <p className="text-xs font-mono mb-1">No results yet</p>
                    <p className="text-[10px] text-code-muted/60">After submitting via MCP, click "Check Results" to fetch from GitHub Actions.</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SWEBenchChallenge;
