import type { Challenge, ChallengeType, Difficulty, ChallengeStatus } from "./challenges-data";
import { aimlChallenges } from "./aiml-challenges";
import { htmlToMarkdown } from "@/lib/html-to-markdown";

/**
 * Raw problem data structure from problems.json
 */
export interface RawProblem {
  index: number;
  slug: string;
  difficulty: "Easy" | "Medium" | "Hard";
  problem_statement: string;
  code_skeleton: string;
  type: "communication" | "bugfix" | "bug_fix";
}

/**
 * Convert slug (kebab-case) to title (Title Case)
 * Example: "two-sum" -> "Two Sum"
 * For bugfix type: "scikit-learn-10297" -> "Scikit-Learn 10297"
 */
function slugToTitle(slug: string, isBugfix: boolean = false): string {
  if (isBugfix) {
    // For bugfix: format is "project-name-issue-number"
    // Extract project name and issue number
    const parts = slug.split("-");
    if (parts.length >= 3) {
      // Handle cases like "scikit-learn-10297"
      // Find where the issue number starts (last part is always the number)
      const issueNumber = parts[parts.length - 1];
      const projectParts = parts.slice(0, -1);
      
      // Capitalize each part of project name and join with hyphens
      const projectName = projectParts
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join("-");
      
      return `${projectName} ${issueNumber}`;
    } else if (parts.length === 2) {
      // Simple case: "django-11141"
      const projectName = parts[0].charAt(0).toUpperCase() + parts[0].slice(1);
      return `${projectName} ${parts[1]}`;
    }
  }
  
  // For communication type: "two-sum" -> "Two Sum"
  return slug
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * Generate challenge ID based on type and index
 */
function generateId(type: ChallengeType, index: number): string {
  const prefix = type === "LeetCodePrompt" ? "lcp" : "hsb";
  return `${prefix}-${String(index).padStart(3, "0")}`;
}

/**
 * Map problem type from JSON to Challenge type
 */
function mapType(type: "communication" | "bugfix" | "bug_fix"): ChallengeType {
  return type === "communication" ? "LeetCodePrompt" : "Human-SWE-Bench";
}

/**
 * Extract plain text from HTML (simple implementation)
 * For more complex HTML parsing, consider using a library like DOMParser
 */
function htmlToText(html: string): string {
  // Simple regex-based extraction (for basic HTML)
  // Remove HTML tags and decode common entities
  return html
    .replace(/<[^>]*>/g, " ") // Remove HTML tags
    .replace(/&nbsp;/g, " ")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ") // Collapse whitespace
    .trim();
}

/**
 * Extract a short description from problem statement
 * Takes first paragraph or first 200 characters
 */
function extractDescription(problemStatement: string): string {
  const text = htmlToText(problemStatement);
  // Try to get first sentence or first 200 chars
  const firstSentence = text.split(/[.!?]/)[0];
  if (firstSentence.length > 0 && firstSentence.length < 200) {
    return firstSentence.trim();
  }
  return text.substring(0, 200).trim() + (text.length > 200 ? "..." : "");
}

/**
 * Extract tags from problem statement or slug
 * This is a simple implementation - you might want to enhance it
 */
function extractTags(slug: string, problemStatement: string): string[] {
  // Common algorithm/data structure tags based on slug patterns
  const tagMap: Record<string, string[]> = {
    "two-sum": ["Array", "Hash Map"],
    "reverse-integer": ["Math"],
    "container-with-most-water": ["Array", "Two Pointers"],
    "integer-to-roman": ["Math", "String"],
    "roman-to-integer": ["Math", "String"],
    "longest-common-prefix": ["String"],
    "3sum": ["Array", "Two Pointers"],
    "add-two-numbers": ["Linked List", "Math"],
    "regular-expression-matching": ["String", "Dynamic Programming"],
    "zigzag-conversion": ["String"],
    "palindrome-number": ["Math"],
    "median-of-two-sorted-arrays": ["Array", "Binary Search"],
    "longest-palindromic-substring": ["String", "Dynamic Programming"],
    "longest-substring-without-repeating-characters": ["String", "Sliding Window"],
    "merge-two-sorted-lists": ["Linked List"],
  };

  // Check if we have predefined tags for this slug
  if (tagMap[slug]) {
    return tagMap[slug];
  }

  // Fallback: extract from problem statement keywords
  const keywords = ["Array", "String", "Linked List", "Tree", "Graph", "Dynamic Programming", "Backtracking", "Hash Map", "Stack", "Queue"];
  const foundTags: string[] = [];
  const lowerStatement = problemStatement.toLowerCase();

  keywords.forEach((keyword) => {
    if (lowerStatement.includes(keyword.toLowerCase())) {
      foundTags.push(keyword);
    }
  });

  return foundTags.length > 0 ? foundTags : ["Algorithm"];
}

/**
 * Generate default acceptance rate based on difficulty
 */
function getDefaultAcceptance(difficulty: string | undefined | null): number {
  const ranges: Record<string, { min: number; max: number }> = {
    Easy: { min: 70, max: 90 },
    Medium: { min: 40, max: 70 },
    Hard: { min: 20, max: 50 },
  };

  // Handle undefined/null or normalize difficulty to match expected values
  if (!difficulty || typeof difficulty !== 'string') {
    const range = ranges["Medium"];
    return Math.floor((range.min + range.max) / 2);
  }

  const normalizedDifficulty = difficulty.charAt(0).toUpperCase() + difficulty.slice(1).toLowerCase();
  const range = ranges[normalizedDifficulty] || ranges["Medium"]; // Default to Medium if not found
  
  // Return a random value in the range (or you could use a fixed value)
  return Math.floor((range.min + range.max) / 2);
}

/**
 * Convert raw problem data to Challenge format
 */
function convertToChallenge(raw: RawProblem): Challenge {
  const type = mapType(raw.type);
  const id = generateId(type, raw.index);
  // Use slug directly as title
  const title = raw.slug;
  const description = extractDescription(raw.problem_statement);
  const tags = extractTags(raw.slug, raw.problem_statement);
  
  // Ensure difficulty is valid, default to "Medium" if not
  const validDifficulties: Difficulty[] = ["Easy", "Medium", "Hard"];
  const normalizedDifficulty = validDifficulties.includes(raw.difficulty as Difficulty) 
    ? (raw.difficulty as Difficulty)
    : "Medium";
  
  const acceptance = getDefaultAcceptance(normalizedDifficulty);

  // Convert HTML to Markdown for display
  const problemStatementMarkdown = htmlToMarkdown(raw.problem_statement);

  const challenge: Challenge = {
    id,
    title,
    type,
    difficulty: normalizedDifficulty,
    status: "new" as ChallengeStatus, // Default status
    tags,
    acceptance,
    description, // Short description for list view
    problemStatement: problemStatementMarkdown, // Full Markdown problem statement for detail view
    testCases: [], // Empty by default - can be extracted from problem_statement if needed
    codeStub: raw.code_skeleton,
  };

  // For Human-SWE-Bench type, we might need additional fields
  // These would need to be in the JSON if available
  if (type === "Human-SWE-Bench") {
    // You can add repoContext, issueBody, etc. if they exist in the JSON
    // For now, we'll leave them undefined
  }

  return challenge;
}

/**
 * Load problems from JSON file and convert to Challenge format
 */
export async function loadProblems(): Promise<Challenge[]> {
  let jsonProblems: Challenge[] = [];

  try {
    const response = await fetch("/data/problems.json");
    if (response.ok) {
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        const rawProblems: RawProblem[] = await response.json();
        jsonProblems = rawProblems.map(convertToChallenge);
      }
    }
  } catch (error) {
    console.warn("Could not load problems.json (non-fatal):", error);
  }

  // Always include AI/ML challenges even if problems.json is unavailable
  return [...jsonProblems, ...aimlChallenges];
}

/**
 * Load problems with caching
 */
let cachedProblems: Challenge[] | null = null;

export async function getProblems(): Promise<Challenge[]> {
  if (cachedProblems) {
    return cachedProblems;
  }

  cachedProblems = await loadProblems();
  return cachedProblems;
}

/**
 * Clear the cache (useful for development or when data updates)
 */
export function clearProblemsCache(): void {
  cachedProblems = null;
}
