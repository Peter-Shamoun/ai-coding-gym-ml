export type ChallengeType = "LeetCodePrompt" | "Human-SWE-Bench" | "AI-ML";
export type Difficulty = "Easy" | "Medium" | "Hard";
export type ChallengeStatus = "new" | "attempted" | "solved";

export interface TestCase {
  input: string;
  output: string;
}

export interface Challenge {
  id: string;
  title: string;
  type: ChallengeType;
  difficulty: Difficulty;
  status: ChallengeStatus;
  tags: string[];
  acceptance: number;
  description: string; // Short description for list view
  problemStatement?: string; // Full HTML problem statement for detail view
  testCases: TestCase[];
  codeStub?: string;
  // Human-SWE-Bench specific
  repoContext?: string;
  issueBody?: string;
  failingTest?: string;
  agentAttempt?: string;
  // AI-ML specific
  backendId?: string; // Flask backend challenge ID (e.g. "email-spam-detection")
  dataset?: string;
  taskType?: string;
  targetMetrics?: string;
  dataDownloadUrl?: string;
  validationScript?: string;
  datasetSamples?: { features: string; label: string }[];
  gradingRubric?: string;
  deliverables?: string;
  allowedLibraries?: string[];
}

export const challenges: Challenge[] = [
  // ─── LeetCodePrompt ───
  {
    id: "lcp-001",
    title: "Two Sum",
    type: "LeetCodePrompt",
    difficulty: "Easy",
    status: "new",
    tags: ["Array", "Hash Map"],
    acceptance: 82,
    description:
      "Write a prompt that instructs AI to find two numbers in an array that add up to a target, returning their indices.",
    testCases: [
      { input: "two_sum([2,7,11,15], 9)", output: "[0, 1]" },
      { input: "two_sum([3,2,4], 6)", output: "[1, 2]" },
      { input: "two_sum([3,3], 6)", output: "[0, 1]" },
    ],
    codeStub: `def two_sum(nums: list[int], target: int) -> list[int]:
    """
    Given an array of integers nums and an integer target,
    return indices of the two numbers that add up to target.
    """
    pass`,
  },
  {
    id: "lcp-002",
    title: "Reverse Linked List",
    type: "LeetCodePrompt",
    difficulty: "Easy",
    status: "solved",
    tags: ["Linked List", "Recursion"],
    acceptance: 78,
    description:
      "Craft a prompt to get AI to reverse a singly linked list iteratively and recursively.",
    testCases: [
      { input: "reverse([1,2,3,4,5])", output: "[5,4,3,2,1]" },
      { input: "reverse([1,2])", output: "[2,1]" },
      { input: "reverse([])", output: "[]" },
    ],
    codeStub: `class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head: ListNode) -> ListNode:
    """Reverse a singly linked list."""
    pass`,
  },
  {
    id: "lcp-003",
    title: "Valid Parentheses",
    type: "LeetCodePrompt",
    difficulty: "Easy",
    status: "attempted",
    tags: ["Stack", "String"],
    acceptance: 75,
    description:
      "Prompt the AI to check if a string of brackets is valid, handling all three bracket types.",
    testCases: [
      { input: 'is_valid("()")', output: "true" },
      { input: 'is_valid("()[]{}")', output: "true" },
      { input: 'is_valid("(]")', output: "false" },
    ],
    codeStub: `def is_valid(s: str) -> bool:
    """
    Given a string containing '(', ')', '{', '}', '[', ']',
    determine if the input string is valid.
    """
    pass`,
  },
  {
    id: "lcp-004",
    title: "Merge Intervals",
    type: "LeetCodePrompt",
    difficulty: "Medium",
    status: "new",
    tags: ["Array", "Sorting"],
    acceptance: 61,
    description:
      "Write a prompt to merge overlapping intervals in a list, preserving non-overlapping ones.",
    testCases: [
      { input: "merge([[1,3],[2,6],[8,10],[15,18]])", output: "[[1,6],[8,10],[15,18]]" },
      { input: "merge([[1,4],[4,5]])", output: "[[1,5]]" },
    ],
    codeStub: `def merge(intervals: list[list[int]]) -> list[list[int]]:
    """Merge all overlapping intervals."""
    pass`,
  },
  {
    id: "lcp-005",
    title: "LRU Cache",
    type: "LeetCodePrompt",
    difficulty: "Medium",
    status: "new",
    tags: ["Hash Map", "Linked List", "Design"],
    acceptance: 54,
    description:
      "Design a prompt that guides AI to implement an LRU cache with O(1) get and put operations.",
    testCases: [
      { input: "cache = LRUCache(2); cache.put(1,1); cache.get(1)", output: "1" },
      { input: "cache.put(2,2); cache.put(3,3); cache.get(1)", output: "-1" },
    ],
    codeStub: `class LRUCache:
    def __init__(self, capacity: int):
        pass
    def get(self, key: int) -> int:
        pass
    def put(self, key: int, value: int) -> None:
        pass`,
  },
  {
    id: "lcp-006",
    title: "Word Search",
    type: "LeetCodePrompt",
    difficulty: "Medium",
    status: "attempted",
    tags: ["Backtracking", "Matrix"],
    acceptance: 48,
    description:
      "Prompt AI to find if a word exists in a 2D grid of characters via adjacent cell traversal.",
    testCases: [
      { input: 'exist([["A","B"],["C","D"]], "ABDC")', output: "true" },
      { input: 'exist([["A","B"],["C","D"]], "ABCD")', output: "false" },
    ],
    codeStub: `def exist(board: list[list[str]], word: str) -> bool:
    """Return true if word exists in the grid."""
    pass`,
  },
  {
    id: "lcp-007",
    title: "Serialize & Deserialize Binary Tree",
    type: "LeetCodePrompt",
    difficulty: "Hard",
    status: "new",
    tags: ["Tree", "BFS", "Design"],
    acceptance: 38,
    description:
      "Craft a prompt for AI to serialize a binary tree to a string and back without losing structure.",
    testCases: [
      { input: "serialize(deserialize('[1,2,3,null,null,4,5]'))", output: "'[1,2,3,null,null,4,5]'" },
      { input: "serialize(deserialize('[]'))", output: "'[]'" },
    ],
    codeStub: `class Codec:
    def serialize(self, root) -> str:
        pass
    def deserialize(self, data: str):
        pass`,
  },
  {
    id: "lcp-008",
    title: "Median of Two Sorted Arrays",
    type: "LeetCodePrompt",
    difficulty: "Hard",
    status: "new",
    tags: ["Binary Search", "Divide & Conquer"],
    acceptance: 32,
    description:
      "Write a prompt to find the median of two sorted arrays in O(log(m+n)) time.",
    testCases: [
      { input: "median([1,3], [2])", output: "2.0" },
      { input: "median([1,2], [3,4])", output: "2.5" },
    ],
    codeStub: `def find_median(nums1: list[int], nums2: list[int]) -> float:
    """Find median of two sorted arrays in O(log(m+n))."""
    pass`,
  },
  {
    id: "lcp-009",
    title: "Trapping Rain Water",
    type: "LeetCodePrompt",
    difficulty: "Hard",
    status: "new",
    tags: ["Two Pointers", "Stack", "DP"],
    acceptance: 41,
    description:
      "Prompt AI to compute how much rainwater is trapped between elevation bars.",
    testCases: [
      { input: "trap([0,1,0,2,1,0,1,3,2,1,2,1])", output: "6" },
      { input: "trap([4,2,0,3,2,5])", output: "9" },
    ],
    codeStub: `def trap(height: list[int]) -> int:
    """Compute trapped rainwater given elevation map."""
    pass`,
  },
  {
    id: "lcp-010",
    title: "Longest Substring Without Repeating",
    type: "LeetCodePrompt",
    difficulty: "Medium",
    status: "solved",
    tags: ["Sliding Window", "Hash Map"],
    acceptance: 58,
    description:
      "Guide AI to find the length of the longest substring without repeating characters.",
    testCases: [
      { input: 'length("abcabcbb")', output: "3" },
      { input: 'length("bbbbb")', output: "1" },
      { input: 'length("pwwkew")', output: "3" },
    ],
    codeStub: `def length_of_longest(s: str) -> int:
    """Length of the longest substring without repeating chars."""
    pass`,
  },

  // ─── Human-SWE-Bench ───
  {
    id: "hsb-001",
    title: "Fix Race Condition in Async Queue",
    type: "Human-SWE-Bench",
    difficulty: "Hard",
    status: "new",
    tags: ["Concurrency", "Debugging"],
    acceptance: 18,
    description:
      "An async task queue processes jobs out of order under load. AI agents fail to identify the root cause. Find and fix the race condition.",
    testCases: [
      { input: "queue.enqueue(A, B, C) under load", output: "Processed in order: A, B, C" },
      { input: "queue.enqueue(X) during processing", output: "X queued, not dropped" },
    ],
    repoContext: "async-queue/src/queue.ts — 120 lines\nasync-queue/src/worker.ts — 85 lines\nasync-queue/tests/queue.test.ts — 60 lines",
    issueBody: `## Bug Report

**Describe the bug:** Under concurrent load (~50 rps), the async queue processes tasks out of order and occasionally drops tasks entirely.

**Steps to reproduce:**
1. Start the queue with concurrency = 1
2. Rapidly enqueue 50 tasks
3. Observe output order differs from input order

**Expected behavior:** Tasks processed in FIFO order, none dropped.

**Environment:** Node.js 20, TypeScript 5.3`,
    failingTest: `test('processes tasks in FIFO order', async () => {
  const results: number[] = [];
  const queue = new AsyncQueue({ concurrency: 1 });
  for (let i = 0; i < 50; i++) {
    queue.enqueue(async () => { results.push(i); });
  }
  await queue.drain();
  expect(results).toEqual(Array.from({length: 50}, (_, i) => i));
  // FAILS: results are out of order
});`,
    agentAttempt: `// AI Agent attempted: Added a mutex lock around enqueue()
// Result: Deadlock under load — the lock is never released
//         when a task throws an error during processing.`,
  },
  {
    id: "hsb-002",
    title: "Resolve Circular Dependency in Module Loader",
    type: "Human-SWE-Bench",
    difficulty: "Hard",
    status: "new",
    tags: ["Architecture", "Debugging"],
    acceptance: 15,
    description:
      "A custom module loader crashes on circular imports. Current AI agents suggest breaking changes. Find a backward-compatible fix.",
    testCases: [
      { input: "load('moduleA') where A→B→A", output: "Both modules loaded, no crash" },
      { input: "load('moduleC') no cycles", output: "Module C loaded normally" },
    ],
    repoContext: "loader/src/resolver.ts — 200 lines\nloader/src/cache.ts — 45 lines",
    issueBody: `## Issue: Circular import causes stack overflow

When module A imports module B which imports module A, the loader enters infinite recursion.

Current AI agents suggest rewriting the entire module API, but we need backward compatibility with 200+ existing plugins.`,
    failingTest: `test('handles circular dependency', () => {
  const loader = new ModuleLoader();
  // A depends on B, B depends on A
  expect(() => loader.load('A')).not.toThrow();
});`,
    agentAttempt: `// AI suggested: Lazy-load all imports via dynamic import()
// Problem: Breaks synchronous require() calls used by plugins`,
  },
  {
    id: "hsb-003",
    title: "Fix Timezone-Aware Date Comparison",
    type: "Human-SWE-Bench",
    difficulty: "Medium",
    status: "attempted",
    tags: ["Date/Time", "Edge Cases"],
    acceptance: 29,
    description:
      "A scheduling app fails across DST boundaries. AI produces code that works in tests but fails in production. Fix the underlying logic.",
    testCases: [
      { input: 'isOverlapping("2024-03-10T01:00", "2024-03-10T03:00", "US/Eastern")', output: "true (DST skip)" },
      { input: 'isOverlapping("2024-11-03T01:00", "2024-11-03T01:30", "US/Eastern")', output: "Handles ambiguous time" },
    ],
    repoContext: "scheduler/src/date-utils.ts — 90 lines\nscheduler/src/booking.ts — 150 lines",
    issueBody: `## Bug: Bookings vanish during DST transition

Events scheduled between 2:00-3:00 AM on March 10 (spring forward) disappear. Events on Nov 3 (fall back) double-book the 1:00-2:00 AM slot.`,
    failingTest: `test('spring forward: 2:30 AM does not exist', () => {
  const result = createBooking('2024-03-10T02:30', 'US/Eastern');
  expect(result.error).toBe('INVALID_TIME');
  // FAILS: silently creates booking at wrong time
});`,
    agentAttempt: `// AI used: moment.tz() with .isValid() check
// Problem: moment considers 2:30 AM "valid" by shifting it`,
  },
  {
    id: "hsb-004",
    title: "Memory Leak in Event Listener Cleanup",
    type: "Human-SWE-Bench",
    difficulty: "Medium",
    status: "new",
    tags: ["Memory", "DOM", "Debugging"],
    acceptance: 34,
    description:
      "A React component leaks memory on unmount. AI suggests useEffect cleanup but misses the root cause in a third-party library integration.",
    testCases: [
      { input: "Mount/unmount component 100 times", output: "Heap size stable (±5%)" },
      { input: "Check detached DOM nodes after unmount", output: "0 detached nodes" },
    ],
    repoContext: "dashboard/src/components/Chart.tsx — 80 lines\ndashboard/src/hooks/useChartLib.ts — 40 lines",
    issueBody: `## Memory leak: Dashboard crashes after ~2 hours

The Chart component leaks ~2MB per mount/unmount cycle. Chrome DevTools shows detached DOM nodes held by the charting library's internal event bus.`,
    failingTest: `test('no detached nodes after unmount', () => {
  const { unmount } = render(<Chart data={mockData} />);
  unmount();
  expect(getDetachedNodes()).toHaveLength(0);
  // FAILS: 3 detached canvas elements remain
});`,
    agentAttempt: `// AI added: useEffect cleanup calling chart.destroy()
// Problem: chart.destroy() doesn't unsubscribe from the
//          library's global event bus (resize observer)`,
  },
  {
    id: "hsb-005",
    title: "Unicode Normalization in Search Index",
    type: "Human-SWE-Bench",
    difficulty: "Hard",
    status: "new",
    tags: ["Unicode", "Search", "Edge Cases"],
    acceptance: 12,
    description:
      "Full-text search misses results with accented characters. AI agents normalize inconsistently. Implement correct NFC/NFD handling.",
    testCases: [
      { input: 'search("café")', output: "Matches 'café' (both NFC and NFD)" },
      { input: 'search("naive")', output: "Matches 'naïve'" },
    ],
    repoContext: "search/src/indexer.ts — 180 lines\nsearch/src/tokenizer.ts — 95 lines",
    issueBody: `## Search misses accented characters

Searching for "café" doesn't find entries stored as "café" (different Unicode normalization form). Affects ~15% of non-English content.`,
    failingTest: `test('finds NFC and NFD variants', () => {
  index.add('café');  // NFC: é = U+00E9
  const results = index.search('cafe\\u0301'); // NFD: e + combining accent
  expect(results).toHaveLength(1);
  // FAILS: returns empty array
});`,
    agentAttempt: `// AI used: str.normalize('NFC') on search query only
// Problem: Index was built with mixed NFC/NFD — both sides need normalizing`,
  },
  {
    id: "hsb-006",
    title: "Flaky Integration Test Due to Port Collision",
    type: "Human-SWE-Bench",
    difficulty: "Easy",
    status: "new",
    tags: ["Testing", "DevOps"],
    acceptance: 52,
    description:
      "CI tests randomly fail because multiple test suites bind the same port. AI suggests hardcoding — find a robust dynamic solution.",
    testCases: [
      { input: "Run 5 test suites in parallel", output: "All pass, no EADDRINUSE" },
      { input: "Run on CI with limited ports", output: "Dynamic port allocation works" },
    ],
    repoContext: "api/tests/setup.ts — 30 lines\napi/tests/auth.test.ts — 90 lines",
    issueBody: `## Flaky CI: EADDRINUSE on port 3001

~20% of CI runs fail with EADDRINUSE because parallel test workers all try to bind port 3001. Need dynamic port allocation.`,
    failingTest: `// Fails intermittently in CI
beforeAll(async () => {
  server = app.listen(3001); // EADDRINUSE when parallel
});`,
    agentAttempt: `// AI suggested: Use port 0 for random assignment
// Problem: Tests that make HTTP requests need to know the port`,
  },
  {
    id: "hsb-007",
    title: "Fix Float Precision in Financial Calculator",
    type: "Human-SWE-Bench",
    difficulty: "Medium",
    status: "new",
    tags: ["Math", "Precision", "Finance"],
    acceptance: 38,
    description:
      "A financial calculator shows rounding errors on compound interest. AI uses toFixed() which compounds the problem. Fix it properly.",
    testCases: [
      { input: "compound(1000, 0.05, 30)", output: "4321.94 (not 4321.95)" },
      { input: "compound(0.1 + 0.2, 0.1, 1)", output: "Correct to 2 decimal places" },
    ],
    repoContext: "finance/src/calculator.ts — 60 lines\nfinance/src/decimal.ts — 25 lines",
    issueBody: `## Rounding errors in compound interest

Users report penny discrepancies on long-term calculations. The error compounds: 30-year calculations can be off by $1-2.`,
    failingTest: `test('30-year compound interest', () => {
  const result = compound(1000, 0.05, 30);
  expect(result).toBe(4321.94);
  // FAILS: returns 4321.950000000001
});`,
    agentAttempt: `// AI used: Math.round(result * 100) / 100
// Problem: Rounding at the end doesn't fix intermediate precision loss`,
  },
  {
    id: "hsb-008",
    title: "Deadlock in Database Transaction Retry",
    type: "Human-SWE-Bench",
    difficulty: "Hard",
    status: "new",
    tags: ["Database", "Concurrency", "Debugging"],
    acceptance: 11,
    description:
      "Retry logic for failed DB transactions causes deadlocks under concurrent writes. AI agents can't reproduce it. Diagnose and solve.",
    testCases: [
      { input: "10 concurrent writes to same row", output: "All succeed (with retries)" },
      { input: "Retry after serialization failure", output: "No deadlock, completes in <5s" },
    ],
    repoContext: "api/src/db/transaction.ts — 70 lines\napi/src/db/retry.ts — 35 lines",
    issueBody: `## Deadlock under concurrent writes

When 10+ users update the same resource simultaneously, the retry logic causes a deadlock chain. The app hangs until connection timeout (30s).`,
    failingTest: `test('concurrent writes resolve', async () => {
  const promises = Array.from({length: 10}, (_, i) =>
    updateResource(id, { value: i })
  );
  await expect(Promise.all(promises)).resolves.toBeDefined();
  // FAILS: Hangs, then timeout after 30s
}, 10000);`,
    agentAttempt: `// AI added: Exponential backoff on retry
// Problem: All transactions back off identically (same seed),
//          causing synchronized retries → repeated deadlock`,
  },
];

/**
 * Placeholder challenges data (for development/fallback)
 * 
 * To use real data from problems.json, import and use:
 * 
 * ```ts
 * import { getProblems } from './load-problems';
 * 
 * // In a component or hook:
 * const challenges = await getProblems();
 * ```
 * 
 * Or use React Query:
 * ```ts
 * import { useQuery } from '@tanstack/react-query';
 * import { getProblems } from '@/data/load-problems';
 * 
 * const { data: challenges = [] } = useQuery({
 *   queryKey: ['challenges'],
 *   queryFn: getProblems,
 * });
 * ```
 */
