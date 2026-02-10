export interface BlogPost {
  slug: string;
  title: string;
  date: string;
  author: string;
  excerpt: string;
  tags: string[];
  content: string;
}

export const blogPosts: BlogPost[] = [
  {
    slug: "prompting-ai-for-better-code",
    title: "5 Prompting Techniques for Better AI-Generated Code",
    date: "2026-02-05",
    author: "AICodingGym Team",
    excerpt:
      "Learn how to craft precise prompts that produce cleaner, more reliable code from AI assistants.",
    tags: ["Prompting", "Best Practices"],
    content: `
## Why Prompting Matters

The quality of AI-generated code is directly proportional to the quality of your prompt. A vague instruction leads to generic, often buggy output. A well-structured prompt produces production-ready code.

### 1. Be Specific About Edge Cases

Instead of saying *"write a sort function"*, try:

\`\`\`
Write a function that sorts an array of integers in ascending order.
It should handle empty arrays, arrays with one element, and arrays
with duplicate values. Use an efficient algorithm (O(n log n)).
\`\`\`

### 2. Specify the Return Type

Always tell the AI what shape your output should take. This reduces ambiguity and helps the AI avoid unnecessary assumptions.

### 3. Provide Example Inputs and Outputs

Nothing clarifies intent like concrete examples:

\`\`\`
Input: [3, 1, 4, 1, 5]
Output: [1, 1, 3, 4, 5]
\`\`\`

### 4. Constrain the Solution

If you need a specific algorithm or approach, say so. Otherwise the AI may pick something suboptimal for your use case.

### 5. Ask for Tests

End your prompt with: *"Also write unit tests covering edge cases."* This forces the AI to think about correctness from multiple angles.

---

Mastering these techniques is exactly what AICodingGym is designed to help you practice. Try a challenge today!
`,
  },
  {
    slug: "debugging-with-ai",
    title: "How to Debug Code Effectively with AI Assistants",
    date: "2026-01-28",
    author: "AICodingGym Team",
    excerpt:
      "AI can be a powerful debugging partner — if you know how to guide it. Here's a practical workflow.",
    tags: ["Debugging", "AI Workflow"],
    content: `
## AI as a Debugging Partner

Debugging is one of the most underrated use cases for AI coding assistants. Instead of just generating new code, you can use AI to *understand* and *fix* existing code.

### Step 1: Describe the Symptom

Start by telling the AI exactly what's going wrong:

\`\`\`
This function is supposed to return the sum of numbers from 1 to n,
but it returns one less than expected for all inputs.
\`\`\`

### Step 2: Provide the Code

Paste the buggy code and let the AI analyze it line by line.

### Step 3: Ask for Explanation Before Fix

A good debugging prompt:

\`\`\`
Explain why this code produces incorrect output, then suggest
a minimal fix. Don't rewrite the entire function.
\`\`\`

### Step 4: Verify with Tests

Always ask the AI to produce test cases that confirm the fix works — including the original failing case.

---

Practice this workflow in AICodingGym's **Debug with AI** challenges!
`,
  },
  {
    slug: "verifying-ai-code",
    title: "Don't Trust, Verify: A Guide to Reviewing AI Code",
    date: "2026-01-20",
    author: "AICodingGym Team",
    excerpt:
      "AI-generated code can look correct at first glance but hide subtle bugs. Learn a systematic review process.",
    tags: ["Code Review", "Verification"],
    content: `
## The Verification Mindset

AI-generated code often *looks* right. It follows conventions, uses good variable names, and even includes comments. But looks can deceive.

### Common AI Code Pitfalls

1. **Off-by-one errors** — Loops that miss the last element or include one too many
2. **Missing null checks** — AI often assumes inputs are well-formed
3. **Incorrect edge cases** — Empty inputs, negative numbers, Unicode strings
4. **Subtle logic errors** — Code that works for the happy path but fails on boundaries

### A Verification Checklist

- [ ] Read the code line by line — don't skim
- [ ] Trace through with a simple example manually
- [ ] Trace through with an edge case (empty, null, max value)
- [ ] Check that all branches are reachable
- [ ] Run the code with adversarial inputs
- [ ] Compare against a reference implementation if available

### Using AI to Verify AI

You can even ask a *second* AI prompt to review the first output:

\`\`\`
Review this function for correctness. List any edge cases
it might fail on and suggest fixes.
\`\`\`

---

Build your verification skills in AICodingGym's **Verify AI Code** challenges!
`,
  },
];
