import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import {
  PanelLeft,
  ArrowLeft,
  Code2,
  Tag,
  ChevronRight,
  Trophy,
  MessageSquare,
  Loader2,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { getProblems } from "@/data/load-problems";
import TaskPanelRight from "@/components/TaskPanelRight";
import ThemeToggle from "@/components/ThemeToggle";
import MarkdownRenderer from "@/components/MarkdownRenderer";

const difficultyColor: Record<string, string> = {
  Easy: "text-code-green",
  Medium: "text-code-yellow",
  Hard: "text-destructive",
};

const LeetCodeChallenge = () => {
  const { id } = useParams<{ id: string }>();
  const { data: challenges = [], isLoading, error } = useQuery({
    queryKey: ["challenges"],
    queryFn: getProblems,
  });

  // URL param is problem slug (e.g. two-sum)
  const challenge = challenges.find(
    (c) => c.type === "LeetCodePrompt" && c.title === id
  );
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [promptLength, setPromptLength] = useState(0);
  const [codeTheme, setCodeTheme] = useState<"dark" | "light">("light");
  const shortestPrompt = 80; // mock: shortest successful prompt

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
            <Code2 className="w-4 h-4 text-code-blue" />
            <span className="text-sm font-medium">{challenge.title}</span>
          </div>
          <span
            className={`text-xs font-mono font-semibold ${difficultyColor[challenge.difficulty]}`}
          >
            {challenge.difficulty}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <ThemeToggle codeTheme={codeTheme} onToggle={() => setCodeTheme(codeTheme === "dark" ? "light" : "dark")} />
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="flex items-center gap-1.5 text-xs text-code-muted hover:text-code-foreground transition-colors"
          >
            <PanelLeft className="w-4 h-4" />
            {sidebarOpen ? "Hide" : "Show"} Description
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-1 min-h-0">
        {/* Collapsible sidebar */}
        <div
          className={`border-r border-code-border bg-code transition-all duration-300 overflow-hidden shrink-0 ${
            sidebarOpen ? "w-[380px]" : "w-0"
          }`}
        >
          <div className="w-[380px] h-full overflow-auto">
            {/* Description */}
            <div className="px-5 py-4 border-b border-code-border">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-mono text-code-accent uppercase tracking-wider">
                  LeetCode Prompt
                </span>
              </div>
              <h2 className="text-lg font-semibold mb-3">{challenge.title}</h2>
              {challenge.problemStatement ? (
                <div className="text-sm text-code-foreground/80 leading-relaxed">
                  <MarkdownRenderer content={challenge.problemStatement} />
                </div>
              ) : (
                <p className="text-sm text-code-foreground/80 leading-relaxed">
                  {challenge.description}
                </p>
              )}
              <div className="flex gap-2 mt-3 flex-wrap">
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
            </div>

            {/* Test cases */}
            <div className="px-5 py-4 border-b border-code-border">
              <h3 className="text-xs font-mono text-code-muted uppercase tracking-wider mb-4">
                Examples
              </h3>
              <div className="space-y-3">
                {challenge.testCases.map((tc, i) => (
                  <div key={i} className="rounded-lg bg-code-border/30 p-3.5 space-y-2">
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
            </div>

            {/* Acceptance */}
            <div className="px-5 py-4">
              <div className="flex items-center justify-between text-xs text-code-muted mb-2">
                <span>Acceptance Rate</span>
                <span className="font-mono">{challenge.acceptance}%</span>
              </div>
              <div className="w-full h-1.5 rounded-full bg-code-border overflow-hidden">
                <div
                  className="h-full rounded-full bg-code-green/60"
                  style={{ width: `${challenge.acceptance}%` }}
                />
              </div>
            </div>

            {/* Prompt length social feature */}
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
                      {shortestPrompt} chars
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-code-muted flex items-center gap-1.5">
                      <MessageSquare className="w-3 h-3 text-code-blue" />
                      Your current prompt
                    </span>
                    <span className={`font-mono font-semibold ${
                      promptLength === 0
                        ? "text-code-muted"
                        : promptLength <= shortestPrompt
                        ? "text-code-green"
                        : "text-code-foreground"
                    }`}>
                      {promptLength} chars
                    </span>
                  </div>
                  {promptLength > 0 && (
                    <div className="w-full h-1.5 rounded-full bg-code-border overflow-hidden mt-1">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${
                          promptLength <= shortestPrompt ? "bg-code-green" : "bg-code-blue"
                        }`}
                        style={{
                          width: `${Math.min(100, (shortestPrompt / Math.max(promptLength, 1)) * 100)}%`,
                        }}
                      />
                    </div>
                  )}
                  {promptLength > 0 && promptLength <= shortestPrompt && (
                    <p className="text-[10px] text-code-green font-mono mt-1">
                      🎉 You're beating the record!
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Collapsed sidebar toggle */}
        {!sidebarOpen && (
          <button
            onClick={() => setSidebarOpen(true)}
            className="shrink-0 w-8 flex items-center justify-center border-r border-code-border hover:bg-code-border/30 transition-colors"
          >
            <ChevronRight className="w-4 h-4 text-code-muted" />
          </button>
        )}

        {/* Code editor panel - takes remaining space */}
        <div className="flex-1 min-w-0">
          <TaskPanelRight
            testCases={challenge.testCases}
            codeStub={challenge.codeStub}
            onPromptChange={(p) => setPromptLength(p.length)}
          />
        </div>
      </div>
    </div>
  );
};

export default LeetCodeChallenge;
