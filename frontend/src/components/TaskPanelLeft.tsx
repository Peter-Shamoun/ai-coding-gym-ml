interface TestCase {
  input: string;
  output: string;
}

interface TaskPanelLeftProps {
  title: string;
  description: string;
  testCases: TestCase[];
  category?: string;
}

const TaskPanelLeft = ({ title, description, testCases, category }: TaskPanelLeftProps) => {
  return (
    <div className="flex flex-col h-full overflow-auto">
      {/* Task header */}
      <div className="px-5 py-4 border-b border-code-border">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-mono text-code-accent uppercase tracking-wider">Challenge</span>
          {category && (
            <>
              <span className="text-xs font-mono text-code-muted">•</span>
              <span className="text-xs font-mono text-code-blue">{category}</span>
            </>
          )}
        </div>
        <h3 className="text-lg font-semibold text-code-foreground">{title}</h3>
      </div>

      {/* Description */}
      <div className="px-5 py-4 border-b border-code-border">
        <p className="text-sm text-code-foreground/80 leading-relaxed">{description}</p>
      </div>

      {/* Test cases */}
      <div className="px-5 py-4 flex-1">
        <h4 className="text-xs font-mono text-code-muted uppercase tracking-wider mb-4">
          Examples
        </h4>
        <div className="space-y-4">
          {testCases.map((tc, i) => (
            <div key={i} className="rounded-lg bg-code-border/30 p-4 space-y-2">
              <div>
                <span className="text-xs font-mono text-code-blue mb-1 block">Input:</span>
                <pre className="text-sm font-mono text-code-foreground whitespace-pre-wrap">{tc.input}</pre>
              </div>
              <div>
                <span className="text-xs font-mono text-code-green mb-1 block">Output:</span>
                <pre className="text-sm font-mono text-code-foreground whitespace-pre-wrap">{tc.output}</pre>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TaskPanelLeft;
