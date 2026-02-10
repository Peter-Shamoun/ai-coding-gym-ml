import Editor, { type OnMount } from "@monaco-editor/react";
import { Loader2 } from "lucide-react";
import { useCallback, useRef } from "react";

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  language?: string;
  theme?: "vs-dark" | "light";
  readOnly?: boolean;
  height?: string;
}

const CodeEditor = ({
  value,
  onChange,
  language = "python",
  theme = "vs-dark",
  readOnly = false,
  height = "100%",
}: CodeEditorProps) => {
  const editorRef = useRef<Parameters<OnMount>[0] | null>(null);

  const handleMount: OnMount = useCallback((editor) => {
    editorRef.current = editor;
    editor.focus();
  }, []);

  const handleChange = useCallback(
    (val: string | undefined) => {
      onChange(val ?? "");
    },
    [onChange]
  );

  return (
    <Editor
      height={height}
      language={language}
      theme={theme}
      value={value}
      onChange={handleChange}
      onMount={handleMount}
      loading={
        <div className="flex items-center justify-center h-full gap-2 text-code-muted">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span className="text-xs font-mono">Loading editor…</span>
        </div>
      }
      options={{
        readOnly,
        fontSize: 15,
        fontFamily: "'Fira Code', 'Cascadia Code', 'JetBrains Mono', Menlo, monospace",
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        lineNumbers: "on",
        renderLineHighlight: "line",
        tabSize: 4,
        insertSpaces: true,
        wordWrap: "on",
        padding: { top: 12, bottom: 12 },
        automaticLayout: true,
        suggestOnTriggerCharacters: true,
        quickSuggestions: true,
      }}
    />
  );
};

export default CodeEditor;
