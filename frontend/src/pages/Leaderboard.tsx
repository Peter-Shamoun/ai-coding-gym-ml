import { useState } from "react";
import { motion } from "framer-motion";
import { Trophy, Users, Wrench, Cpu, Crown, Flame, Target, Zap, Hash } from "lucide-react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import {
  userLeaderboard,
  toolLeaderboard,
  modelLeaderboard,
} from "@/data/leaderboard-data";

type Tab = "users" | "tools" | "models";

const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: "users", label: "Users", icon: <Users className="w-4 h-4" /> },
  { id: "tools", label: "Tools", icon: <Wrench className="w-4 h-4" /> },
  { id: "models", label: "Models", icon: <Cpu className="w-4 h-4" /> },
];

const rankBadge = (rank: number) => {
  if (rank === 1) return <Crown className="w-4 h-4 text-code-yellow" />;
  if (rank === 2) return <Crown className="w-4 h-4 text-muted-foreground" />;
  if (rank === 3) return <Crown className="w-4 h-4 text-primary" />;
  return <span className="text-sm font-mono text-muted-foreground w-4 text-center">{rank}</span>;
};

const Leaderboard = () => {
  const [activeTab, setActiveTab] = useState<Tab>("users");

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header />
      <main className="pt-28 pb-20 px-6">
        <div className="container mx-auto max-w-4xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mb-10"
          >
            <div className="flex items-center gap-3 mb-2">
              <Trophy className="w-8 h-8 text-primary" />
              <h1 className="text-3xl sm:text-4xl font-bold">Leaderboard</h1>
            </div>
            <p className="text-muted-foreground">
              See who's leading in AI-assisted coding challenges.
            </p>
          </motion.div>

          {/* Tabs */}
          <motion.div
            className="flex gap-1 p-1 rounded-xl bg-secondary/60 w-fit mb-8"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
          >
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  activeTab === tab.id
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </motion.div>

          {/* Table */}
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="rounded-xl border border-border overflow-hidden"
          >
            {activeTab === "users" && <UsersTable />}
            {activeTab === "tools" && <ToolsTable />}
            {activeTab === "models" && <ModelsTable />}
          </motion.div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

const UsersTable = () => (
  <table className="w-full">
    <thead>
      <tr className="bg-secondary/40 text-xs font-mono uppercase tracking-wider text-muted-foreground">
        <th className="text-left px-5 py-3 w-16"><Hash className="w-3.5 h-3.5" /></th>
        <th className="text-left px-5 py-3">User</th>
        <th className="text-right px-5 py-3">Score</th>
        <th className="text-right px-5 py-3 hidden sm:table-cell">Solved</th>
        <th className="text-right px-5 py-3 hidden sm:table-cell">Streak</th>
      </tr>
    </thead>
    <tbody>
      {userLeaderboard.map((u, i) => (
        <tr
          key={u.name}
          className={`border-t border-border/60 transition-colors hover:bg-secondary/20 ${
            i < 3 ? "bg-primary/[0.03]" : ""
          }`}
        >
          <td className="px-5 py-3.5">{rankBadge(u.rank)}</td>
          <td className="px-5 py-3.5">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center text-sm font-semibold text-foreground">
                {u.name[0].toUpperCase()}
              </div>
              <span className="font-medium text-sm">{u.name}</span>
            </div>
          </td>
          <td className="px-5 py-3.5 text-right">
            <span className="font-mono font-semibold text-sm text-primary">{u.score.toLocaleString()}</span>
          </td>
          <td className="px-5 py-3.5 text-right hidden sm:table-cell">
            <span className="flex items-center justify-end gap-1.5 text-sm text-muted-foreground">
              <Target className="w-3.5 h-3.5" />
              {u.solved}
            </span>
          </td>
          <td className="px-5 py-3.5 text-right hidden sm:table-cell">
            <span className="flex items-center justify-end gap-1.5 text-sm text-muted-foreground">
              <Flame className="w-3.5 h-3.5 text-primary" />
              {u.streak}d
            </span>
          </td>
        </tr>
      ))}
    </tbody>
  </table>
);

const ToolsTable = () => (
  <table className="w-full">
    <thead>
      <tr className="bg-secondary/40 text-xs font-mono uppercase tracking-wider text-muted-foreground">
        <th className="text-left px-5 py-3 w-16"><Hash className="w-3.5 h-3.5" /></th>
        <th className="text-left px-5 py-3">Tool</th>
        <th className="text-right px-5 py-3">Score</th>
        <th className="text-right px-5 py-3 hidden sm:table-cell">Completed</th>
        <th className="text-right px-5 py-3 hidden sm:table-cell">Accuracy</th>
      </tr>
    </thead>
    <tbody>
      {toolLeaderboard.map((t, i) => (
        <tr
          key={t.name}
          className={`border-t border-border/60 transition-colors hover:bg-secondary/20 ${
            i < 3 ? "bg-primary/[0.03]" : ""
          }`}
        >
          <td className="px-5 py-3.5">{rankBadge(t.rank)}</td>
          <td className="px-5 py-3.5">
            <div className="flex items-center gap-3">
              <span className="text-lg">{t.icon}</span>
              <span className="font-medium text-sm">{t.name}</span>
            </div>
          </td>
          <td className="px-5 py-3.5 text-right">
            <span className="font-mono font-semibold text-sm text-primary">{t.score.toLocaleString()}</span>
          </td>
          <td className="px-5 py-3.5 text-right hidden sm:table-cell">
            <span className="flex items-center justify-end gap-1.5 text-sm text-muted-foreground">
              <Target className="w-3.5 h-3.5" />
              {t.challengesCompleted}
            </span>
          </td>
          <td className="px-5 py-3.5 text-right hidden sm:table-cell">
            <span className="flex items-center justify-end gap-1.5 text-sm text-muted-foreground">
              <Zap className="w-3.5 h-3.5 text-code-green" />
              {t.avgAccuracy}%
            </span>
          </td>
        </tr>
      ))}
    </tbody>
  </table>
);

const ModelsTable = () => (
  <table className="w-full">
    <thead>
      <tr className="bg-secondary/40 text-xs font-mono uppercase tracking-wider text-muted-foreground">
        <th className="text-left px-5 py-3 w-16"><Hash className="w-3.5 h-3.5" /></th>
        <th className="text-left px-5 py-3">Model</th>
        <th className="text-right px-5 py-3">Score</th>
        <th className="text-right px-5 py-3 hidden sm:table-cell">Pass Rate</th>
        <th className="text-right px-5 py-3 hidden sm:table-cell">Avg Tokens</th>
      </tr>
    </thead>
    <tbody>
      {modelLeaderboard.map((m, i) => (
        <tr
          key={m.name}
          className={`border-t border-border/60 transition-colors hover:bg-secondary/20 ${
            i < 3 ? "bg-primary/[0.03]" : ""
          }`}
        >
          <td className="px-5 py-3.5">{rankBadge(m.rank)}</td>
          <td className="px-5 py-3.5">
            <div>
              <span className="font-medium text-sm block">{m.name}</span>
              <span className="text-xs text-muted-foreground">{m.provider}</span>
            </div>
          </td>
          <td className="px-5 py-3.5 text-right">
            <span className="font-mono font-semibold text-sm text-primary">{m.score.toLocaleString()}</span>
          </td>
          <td className="px-5 py-3.5 text-right hidden sm:table-cell">
            <span className="flex items-center justify-end gap-1.5 text-sm text-muted-foreground">
              <Zap className="w-3.5 h-3.5 text-code-green" />
              {m.passRate}%
            </span>
          </td>
          <td className="px-5 py-3.5 text-right hidden sm:table-cell">
            <span className="font-mono text-sm text-muted-foreground">{m.avgTokens.toLocaleString()}</span>
          </td>
        </tr>
      ))}
    </tbody>
  </table>
);

export default Leaderboard;
