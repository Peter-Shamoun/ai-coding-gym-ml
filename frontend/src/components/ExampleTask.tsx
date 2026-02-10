import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Shuffle } from "lucide-react";
import TaskPanelLeft from "./TaskPanelLeft";
import TaskPanelRight from "./TaskPanelRight";

const challenges = [
  {
    title: "Reverse a String",
    category: "Frame Tasks for AI",
    description:
      "Write a prompt that instructs the AI to create a function reversing a string. It should handle edge cases like empty strings and single characters.",
    testCases: [
      { input: 'reverse_string("hello")', output: '"olleh"' },
      { input: 'reverse_string("AICodingGym")', output: '"myGgnidoCIA"' },
      { input: 'reverse_string("")', output: '""' },
    ],
  },
  {
    title: "Fix the Off-by-One Bug",
    category: "Debug with AI",
    description:
      "The following function has an off-by-one error in its loop. Write a prompt that helps the AI identify and fix the bug without changing the algorithm.",
    testCases: [
      { input: "sum_range(1, 5)", output: "15" },
      { input: "sum_range(0, 0)", output: "0" },
      { input: "sum_range(3, 3)", output: "3" },
    ],
  },
  {
    title: "Validate a Palindrome",
    category: "Verify AI Code",
    description:
      "The AI generated a palindrome checker, but it may have issues. Write a prompt to verify correctness, suggest edge-case tests, and fix any problems.",
    testCases: [
      { input: 'is_palindrome("racecar")', output: "true" },
      { input: 'is_palindrome("hello")', output: "false" },
      { input: 'is_palindrome("A man a plan a canal Panama")', output: "true" },
    ],
  },
  {
    title: "Parse Nested JSON Safely",
    category: "Solve What AI Can't",
    description:
      "Write a prompt for a function that safely extracts deeply nested values from JSON without crashing. AI often misses null-safety — guide it carefully.",
    testCases: [
      { input: 'safe_get({"a":{"b":1}}, "a.b")', output: "1" },
      { input: 'safe_get({"a":{}}, "a.b.c")', output: "None" },
      { input: 'safe_get(None, "x")', output: "None" },
    ],
  },
];

const ExampleTask = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const task = challenges[currentIndex];

  const handleRandom = useCallback(() => {
    let next: number;
    do {
      next = Math.floor(Math.random() * challenges.length);
    } while (next === currentIndex && challenges.length > 1);
    setCurrentIndex(next);
  }, [currentIndex]);

  return (
    <section id="example" className="py-16 px-6">
      <div className="container mx-auto max-w-6xl">
        <motion.div
          className="flex items-center justify-between mb-6"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold text-foreground">
              Try a Challenge
            </h2>
          </div>
          <button
            onClick={handleRandom}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-border bg-card text-sm font-medium text-foreground hover:bg-secondary transition-colors"
          >
            <Shuffle className="w-4 h-4 text-primary" />
            Random
          </button>
        </motion.div>

        <motion.div
          className="rounded-xl border border-border overflow-hidden shadow-xl shadow-foreground/5"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.15 }}
        >
          <div className="grid grid-cols-2 min-h-[520px]">
            {/* Left panel - Task & Examples */}
            <div className="bg-code border-r border-code-border">
              <TaskPanelLeft
                title={task.title}
                description={task.description}
                testCases={task.testCases}
                category={task.category}
              />
            </div>

            {/* Right panel - Code & Prompt */}
            <div className="bg-code">
              <TaskPanelRight testCases={task.testCases} />
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default ExampleTask;
