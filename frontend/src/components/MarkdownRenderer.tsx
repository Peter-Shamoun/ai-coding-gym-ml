import React from "react";

interface MarkdownRendererProps {
  content: string;
}

const MarkdownRenderer = ({ content }: MarkdownRendererProps) => {
  const lines = content.trim().split("\n");
  const elements: React.ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Code block
    if (line.startsWith("```")) {
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      elements.push(
        <pre
          key={elements.length}
          className="bg-code-border/30 text-code-foreground rounded-lg p-4 text-sm font-mono overflow-x-auto my-4 border border-code-border"
        >
          <code>{codeLines.join("\n")}</code>
        </pre>
      );
      continue;
    }

    // Horizontal rule
    if (line.trim() === "---") {
      elements.push(
        <hr key={elements.length} className="border-code-border my-6" />
      );
      i++;
      continue;
    }

    // Empty line
    if (line.trim() === "") {
      i++;
      continue;
    }

    // Blockquote
    if (line.trimStart().startsWith("> ")) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].trimStart().startsWith("> ")) {
        quoteLines.push(lines[i].trimStart().slice(2));
        i++;
      }
      elements.push(
        <blockquote
          key={elements.length}
          className="border-l-2 border-primary/50 pl-4 my-4 text-code-foreground/70 italic text-sm leading-relaxed"
        >
          {quoteLines.map((ql, qi) => (
            <p key={qi}>{renderInline(ql)}</p>
          ))}
        </blockquote>
      );
      continue;
    }

    // Headings
    if (line.startsWith("# ")) {
      elements.push(
        <h1
          key={elements.length}
          className="text-xl font-bold mt-6 mb-3 text-code-foreground"
        >
          {renderInline(line.slice(2))}
        </h1>
      );
      i++;
      continue;
    }

    if (line.startsWith("## ")) {
      elements.push(
        <h2
          key={elements.length}
          className="text-lg font-bold mt-6 mb-2 text-code-foreground"
        >
          {renderInline(line.slice(3))}
        </h2>
      );
      i++;
      continue;
    }

    if (line.startsWith("### ")) {
      elements.push(
        <h3
          key={elements.length}
          className="text-sm font-semibold uppercase tracking-wider mt-5 mb-2 text-code-accent"
        >
          {renderInline(line.slice(4))}
        </h3>
      );
      i++;
      continue;
    }

    if (line.startsWith("#### ")) {
      elements.push(
        <h4
          key={elements.length}
          className="text-sm font-semibold mt-4 mb-1.5 text-code-foreground"
        >
          {renderInline(line.slice(5))}
        </h4>
      );
      i++;
      continue;
    }

    // Checklist
    if (line.trimStart().startsWith("- [ ] ") || line.trimStart().startsWith("- [x] ")) {
      const listItems: React.ReactNode[] = [];
      while (
        i < lines.length &&
        (lines[i].trimStart().startsWith("- [ ] ") ||
          lines[i].trimStart().startsWith("- [x] "))
      ) {
        const checked = lines[i].trimStart().startsWith("- [x] ");
        const text = lines[i].trimStart().slice(6);
        listItems.push(
          <li key={listItems.length} className="flex items-start gap-2">
            <span className={`mt-1 w-4 h-4 rounded border flex items-center justify-center shrink-0 ${checked ? "bg-primary border-primary" : "border-muted-foreground/40"}`}>
              {checked && <span className="text-primary-foreground text-xs">✓</span>}
            </span>
            <span>{renderInline(text)}</span>
          </li>
        );
        i++;
      }
      elements.push(
        <ul key={elements.length} className="space-y-2 my-3">
          {listItems}
        </ul>
      );
      continue;
    }

    // Ordered list
    if (/^\d+\.\s/.test(line.trimStart())) {
      const listItems: React.ReactNode[] = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i].trimStart())) {
        const text = lines[i].trimStart().replace(/^\d+\.\s/, "");
        listItems.push(
          <li key={listItems.length} className="pl-1">{renderInline(text)}</li>
        );
        i++;
      }
      elements.push(
        <ol
          key={elements.length}
          className="list-decimal list-outside pl-5 space-y-1.5 my-3 text-code-foreground/80"
        >
          {listItems}
        </ol>
      );
      continue;
    }

    // Unordered list
    if (line.trimStart().startsWith("- ")) {
      const listItems: React.ReactNode[] = [];
      while (i < lines.length && lines[i].trimStart().startsWith("- ")) {
        const text = lines[i].trimStart().slice(2);
        listItems.push(
          <li key={listItems.length} className="pl-1">{renderInline(text)}</li>
        );
        i++;
      }
      elements.push(
        <ul
          key={elements.length}
          className="list-disc list-outside pl-5 space-y-1.5 my-3 text-code-foreground/80"
        >
          {listItems}
        </ul>
      );
      continue;
    }

    // Paragraph
    elements.push(
      <p
        key={elements.length}
        className="text-code-foreground/80 leading-relaxed my-2"
      >
        {renderInline(line)}
      </p>
    );
    i++;
  }

  return <>{elements}</>;
};

function renderInline(text: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  // Match: **bold**, *italic*, `code`, [link](url)
  const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|\[(.+?)\]\((.+?)\))/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    if (match[2]) {
      parts.push(
        <strong key={parts.length} className="font-semibold text-code-foreground">
          {match[2]}
        </strong>
      );
    } else if (match[3]) {
      parts.push(
        <em key={parts.length} className="italic">
          {match[3]}
        </em>
      );
    } else if (match[4]) {
      parts.push(
        <code
          key={parts.length}
          className="bg-code-border/30 text-code-blue px-1.5 py-0.5 rounded text-xs font-mono"
        >
          {match[4]}
        </code>
      );
    } else if (match[5] && match[6]) {
      parts.push(
        <a
          key={parts.length}
          href={match[6]}
          target="_blank"
          rel="noopener noreferrer"
          className="text-code-blue hover:underline"
        >
          {match[5]}
        </a>
      );
    }
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length === 1 ? parts[0] : parts;
}

export default MarkdownRenderer;
