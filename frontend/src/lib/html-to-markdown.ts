/**
 * Convert HTML to Markdown format
 * This is a simple implementation that handles common HTML elements
 */
export function htmlToMarkdown(html: string): string {
  if (!html) return "";

  let text = html;

  // If it's already plain text (no HTML tags), process it directly
  if (!html.includes("<")) {
    text = html;
  } else {
    // Create a temporary DOM element to parse HTML
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, "text/html");
    const body = doc.body;

    let markdown = "";

    // Process each child node
    for (let i = 0; i < body.childNodes.length; i++) {
      const node = body.childNodes[i];
      markdown += processNode(node);
    }

    text = markdown;
  }

  // Clean up: remove duplicate content and "Description" label
  const lines = text.split("\n");
  
  // Remove "Description" label if it appears on its own line (near the beginning)
  const filteredLines: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    // Skip "Description" label if it appears near the beginning (within first 5 lines)
    if (line.toLowerCase() === "description" && i < 5) {
      continue;
    }
    filteredLines.push(lines[i]); // Keep original line (with spacing) for better formatting
  }
  
  // Detect and remove duplicate content
  // If the content appears to be duplicated (same title appears twice)
  if (filteredLines.length > 5) {
    const firstLine = filteredLines[0].trim();
    // Find where the duplicate starts (look for the same title again)
    let duplicateStartIndex = -1;
    for (let i = 1; i < filteredLines.length; i++) {
      if (filteredLines[i].trim() === firstLine && i > filteredLines.length / 2) {
        duplicateStartIndex = i;
        break;
      }
    }
    
    // If we found a duplicate, remove everything from that point onwards
    if (duplicateStartIndex > 0) {
      filteredLines.splice(duplicateStartIndex);
    }
  }
  
  text = filteredLines.join("\n");

  // Clean up extra newlines
  text = text
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  return text;
}

function processNode(node: Node): string {
  if (node.nodeType === Node.TEXT_NODE) {
    return node.textContent || "";
  }

  if (node.nodeType !== Node.ELEMENT_NODE) {
    return "";
  }

  const element = node as HTMLElement;
  const tagName = element.tagName.toLowerCase();
  const children = Array.from(element.childNodes)
    .map(processNode)
    .join("")
    .trim();

  switch (tagName) {
    case "p":
      return children ? `${children}\n\n` : "";

    case "h1":
      return children ? `# ${children}\n\n` : "";

    case "h2":
      return children ? `## ${children}\n\n` : "";

    case "h3":
      return children ? `### ${children}\n\n` : "";

    case "h4":
      return children ? `#### ${children}\n\n` : "";

    case "h5":
      return children ? `##### ${children}\n\n` : "";

    case "h6":
      return children ? `###### ${children}\n\n` : "";

    case "strong":
    case "b":
      return children ? `**${children}**` : "";

    case "em":
    case "i":
      return children ? `*${children}*` : "";

    case "code":
      // Check if it's inside a pre tag (code block)
      if (element.parentElement?.tagName.toLowerCase() === "pre") {
        return children;
      }
      return children ? `\`${children}\`` : "";

    case "pre":
      const codeContent = element.querySelector("code")
        ? element.querySelector("code")!.textContent || ""
        : element.textContent || "";
      return codeContent ? `\`\`\`\n${codeContent}\n\`\`\`\n\n` : "";

    case "ul":
      const ulItems = Array.from(element.querySelectorAll("li"))
        .map((li) => {
          const content = processNode(li).trim();
          return content ? `- ${content}` : "";
        })
        .filter(Boolean);
      return ulItems.length > 0 ? `${ulItems.join("\n")}\n\n` : "";

    case "ol":
      const olItems = Array.from(element.querySelectorAll("li"))
        .map((li, index) => {
          const content = processNode(li).trim();
          return content ? `${index + 1}. ${content}` : "";
        })
        .filter(Boolean);
      return olItems.length > 0 ? `${olItems.join("\n")}\n\n` : "";

    case "li":
      // Process children but don't add list markers here (handled by parent)
      return children;

    case "blockquote":
      return children
        ? children
            .split("\n")
            .map((line) => (line.trim() ? `> ${line.trim()}` : ""))
            .filter(Boolean)
            .join("\n") + "\n\n"
        : "";

    case "hr":
      return "---\n\n";

    case "br":
      return "\n";

    case "a":
      const href = element.getAttribute("href") || "";
      const linkText = children || href;
      return href ? `[${linkText}](${href})` : linkText;

    case "img":
      const src = element.getAttribute("src") || "";
      const alt = element.getAttribute("alt") || "";
      return src ? `![${alt}](${src})` : "";

    case "table":
      return convertTable(element);

    case "div":
    case "span":
      // Just return children for div/span
      return children;

    default:
      // For unknown tags, just return the text content
      return children;
  }
}

function convertTable(table: HTMLElement): string {
  const rows = Array.from(table.querySelectorAll("tr"));
  if (rows.length === 0) return "";

  let markdown = "";

  // Process header row
  const headerRow = rows[0];
  const headers = Array.from(headerRow.querySelectorAll("th, td")).map(
    (cell) => processNode(cell).trim()
  );
  if (headers.length > 0) {
    markdown += `| ${headers.join(" | ")} |\n`;
    markdown += `| ${headers.map(() => "---").join(" | ")} |\n`;
  }

  // Process data rows
  for (let i = 1; i < rows.length; i++) {
    const cells = Array.from(rows[i].querySelectorAll("td")).map(
      (cell) => processNode(cell).trim()
    );
    if (cells.length > 0) {
      markdown += `| ${cells.join(" | ")} |\n`;
    }
  }

  return markdown + "\n";
}
