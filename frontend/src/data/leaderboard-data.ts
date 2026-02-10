export interface LeaderboardEntry {
  rank: number;
  name: string;
  avatar?: string;
  score: number;
  solved: number;
  streak: number;
}

export interface ToolEntry {
  rank: number;
  name: string;
  icon: string;
  score: number;
  challengesCompleted: number;
  avgAccuracy: number;
}

export interface ModelEntry {
  rank: number;
  name: string;
  provider: string;
  score: number;
  passRate: number;
  avgTokens: number;
}

export const userLeaderboard: LeaderboardEntry[] = [
  { rank: 1, name: "sarah_codes", score: 4820, solved: 142, streak: 23 },
  { rank: 2, name: "devMaster99", score: 4510, solved: 134, streak: 18 },
  { rank: 3, name: "ai_whisperer", score: 4340, solved: 128, streak: 31 },
  { rank: 4, name: "promptNinja", score: 4120, solved: 119, streak: 12 },
  { rank: 5, name: "codeZen", score: 3980, solved: 115, streak: 9 },
  { rank: 6, name: "bug_hunter_42", score: 3750, solved: 108, streak: 15 },
  { rank: 7, name: "algoAlice", score: 3620, solved: 101, streak: 7 },
  { rank: 8, name: "rustacean_dev", score: 3480, solved: 96, streak: 11 },
  { rank: 9, name: "ml_engineer", score: 3310, solved: 89, streak: 5 },
  { rank: 10, name: "fullstack_fox", score: 3150, solved: 82, streak: 14 },
];

export const toolLeaderboard: ToolEntry[] = [
  { rank: 1, name: "Cursor", icon: "⚡", score: 9450, challengesCompleted: 312, avgAccuracy: 94.2 },
  { rank: 2, name: "Claude Code", icon: "🧠", score: 9180, challengesCompleted: 298, avgAccuracy: 93.5 },
  { rank: 3, name: "GitHub Copilot", icon: "🤖", score: 8720, challengesCompleted: 285, avgAccuracy: 91.1 },
  { rank: 4, name: "Windsurf", icon: "🏄", score: 8340, challengesCompleted: 267, avgAccuracy: 89.8 },
  { rank: 5, name: "VS Code + AI", icon: "💻", score: 7890, challengesCompleted: 251, avgAccuracy: 87.3 },
  { rank: 6, name: "Aider", icon: "🛠", score: 7540, challengesCompleted: 238, avgAccuracy: 86.1 },
  { rank: 7, name: "Cline", icon: "📟", score: 7120, challengesCompleted: 219, avgAccuracy: 84.5 },
  { rank: 8, name: "JetBrains AI", icon: "🔧", score: 6850, challengesCompleted: 204, avgAccuracy: 83.2 },
];

export const modelLeaderboard: ModelEntry[] = [
  { rank: 1, name: "Claude 4 Opus", provider: "Anthropic", score: 9720, passRate: 96.8, avgTokens: 1240 },
  { rank: 2, name: "GPT-5", provider: "OpenAI", score: 9540, passRate: 95.1, avgTokens: 1380 },
  { rank: 3, name: "Claude 4 Sonnet", provider: "Anthropic", score: 9280, passRate: 93.4, avgTokens: 980 },
  { rank: 4, name: "Gemini 2.5 Pro", provider: "Google", score: 9050, passRate: 91.8, avgTokens: 1150 },
  { rank: 5, name: "GPT-4.5", provider: "OpenAI", score: 8710, passRate: 89.5, avgTokens: 1420 },
  { rank: 6, name: "DeepSeek R2", provider: "DeepSeek", score: 8480, passRate: 87.9, avgTokens: 1050 },
  { rank: 7, name: "Llama 4 405B", provider: "Meta", score: 8120, passRate: 85.3, avgTokens: 1310 },
  { rank: 8, name: "Mistral Large 3", provider: "Mistral", score: 7850, passRate: 83.1, avgTokens: 1180 },
  { rank: 9, name: "Qwen 3 Max", provider: "Alibaba", score: 7540, passRate: 80.6, avgTokens: 1090 },
  { rank: 10, name: "Command R++", provider: "Cohere", score: 7210, passRate: 78.2, avgTokens: 1260 },
];
