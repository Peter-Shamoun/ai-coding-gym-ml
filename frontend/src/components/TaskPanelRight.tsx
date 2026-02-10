import { useState } from "react";
import { Send, Play, CheckCircle2, XCircle, X } from "lucide-react";

interface TestCase {
  input: string;
  output: string;
}

interface TestResult {
  input: string;
  output: string;
  expected: string;
  passed: boolean;
}

interface TaskPanelRightProps {
  testCases?: TestCase[];
  codeStub?: string;
  onPromptChange?: (prompt: string) => void;
}

const TaskPanelRight = ({ testCases = [], codeStub, onPromptChange }: TaskPanelRightProps) => {
  const [prompt, setPrompt] = useState("");
  const [testResults, setTestResults] = useState<TestResult[] | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const handleRun = () => {
    if (testCases.length === 0) return;
    setIsRunning(true);
    setTimeout(() => {
      const results: TestResult[] = testCases.map((tc) => {
        const passed = Math.random() > 0.4;
        return {
          input: tc.input,
          expected: tc.output,
          output: passed ? tc.output : '"error"',
          passed,
        };
      });
      setTestResults(results);
      setIsRunning(false);
    }, 800);
  };

  const passedCount = testResults?.filter((r) => r.passed).length ?? 0;
  const totalCount = testResults?.length ?? 0;

  const placeholderCode = codeStub || `# AI-generated code will appear here
# after you submit your prompt...

def solve():
    """
    Write a prompt that instructs the AI 
    to solve this challenge correctly.
    """
    pass`;

  return (
    <div className="flex flex-col h-full">
      {/* Editor header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-code-border">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-destructive/60" />
            <span className="w-3 h-3 rounded-full bg-code-yellow/60" />
            <span className="w-3 h-3 rounded-full bg-code-green/60" />
          </div>
          <span className="text-xs font-mono text-code-muted">solution.py</span>
        </div>
        <button
          onClick={handleRun}
          disabled={isRunning}
          className="flex items-center gap-1.5 text-xs font-mono text-code-green hover:text-code-green/80 transition-colors disabled:opacity-50"
        >
          <Play className="w-3 h-3" />
          {isRunning ? "Running…" : "Run"}
        </button>
      </div>

      {/* Code area */}
      <div className="flex-1 overflow-auto p-5">
        <pre className="text-sm font-mono leading-relaxed">
          {placeholderCode.split("\n").map((line, i) => (
            <div key={i} className="flex">
              <span className="w-8 text-right pr-4 text-code-muted/50 select-none text-xs leading-relaxed">
                {i + 1}
              </span>
              <span className={
                line.startsWith("#") || line.includes('"""')
                  ? "text-code-muted"
                  : line.includes("def ")
                  ? "text-code-foreground"
                  : line.includes("pass")
                  ? "text-code-accent"
                  : "text-code-foreground"
              }>
                {line.includes("def ") ? (
                  <>
                    <span className="text-code-blue">def </span>
                    <span className="text-code-yellow">{line.replace("def ", "").split("(")[0]}</span>
                    <span className="text-code-foreground">({line.split("(")[1]?.split(")")[0]})</span>
                    <span className="text-code-foreground"> {"->"} </span>
                    <span className="text-code-blue">str</span>
                    <span className="text-code-foreground">:</span>
                  </>
                ) : (
                  line
                )}
              </span>
            </div>
          ))}
        </pre>
      </div>

      {/* Test results panel */}
      {testResults && (
        <div className="border-t border-code-border">
          <div className="flex items-center justify-between px-5 py-2.5">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-code-muted uppercase tracking-wider">Tests</span>
              <span className={`text-xs font-mono font-semibold ${passedCount === totalCount ? "text-code-green" : "text-code-yellow"}`}>
                {passedCount}/{totalCount} passed
              </span>
            </div>
            <button onClick={() => setTestResults(null)} className="text-code-muted hover:text-code-foreground transition-colors">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="px-5 pb-3 space-y-1.5 max-h-[140px] overflow-auto">
            {testResults.map((r, i) => (
              <div key={i} className={`flex items-start gap-2 text-xs font-mono rounded-md px-3 py-2 ${r.passed ? "bg-code-green/10" : "bg-destructive/10"}`}>
                {r.passed ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-code-green mt-0.5 shrink-0" />
                ) : (
                  <XCircle className="w-3.5 h-3.5 text-destructive mt-0.5 shrink-0" />
                )}
                <div className="min-w-0">
                  <span className="text-code-foreground">{r.input}</span>
                  {!r.passed && (
                    <div className="mt-1 text-code-muted">
                      Expected <span className="text-code-green">{r.expected}</span>, got <span className="text-destructive">{r.output}</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Prompt input */}
      <div className="border-t border-code-border p-4">
        <div className="relative">
          <textarea
            value={prompt}
            onChange={(e) => {
              setPrompt(e.target.value);
              onPromptChange?.(e.target.value);
            }}
            placeholder="Describe how the AI should solve this challenge..."
            className="w-full bg-code-border/30 text-code-foreground text-sm font-mono rounded-lg px-4 py-3 pr-12 resize-none placeholder:text-code-muted/60 focus:outline-none focus:ring-1 focus:ring-code-accent/50 min-h-[80px]"
            rows={3}
          />
          <button
            className="absolute bottom-3 right-3 w-8 h-8 rounded-lg bg-primary flex items-center justify-center hover:opacity-90 transition-opacity"
          >
            <Send className="w-4 h-4 text-primary-foreground" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default TaskPanelRight;
