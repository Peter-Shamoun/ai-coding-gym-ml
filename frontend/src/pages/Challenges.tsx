import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { Search, Filter, Code2, Bug, Brain, CheckCircle2, Circle, Clock, ChevronRight, Loader2 } from "lucide-react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { Badge } from "@/components/ui/badge";
import { getProblems } from "@/data/load-problems";
import type { Challenge, ChallengeType, Difficulty, ChallengeStatus } from "@/data/challenges-data";
import { useUserProgress } from "@/hooks/use-user-progress";

const typeConfig: Record<ChallengeType, { label: string; icon: React.ReactNode; color: string }> = {
  LeetCodePrompt: {
    label: "LeetCode Prompt",
    icon: <Code2 className="w-3.5 h-3.5" />,
    color: "text-code-blue",
  },
  "Human-SWE-Bench": {
    label: "Human-SWE-Bench",
    icon: <Bug className="w-3.5 h-3.5" />,
    color: "text-primary",
  },
  "AI-ML": {
    label: "AI / ML",
    icon: <Brain className="w-3.5 h-3.5" />,
    color: "text-code-accent",
  },
};

const difficultyColor: Record<Difficulty, string> = {
  Easy: "text-code-green",
  Medium: "text-code-yellow",
  Hard: "text-destructive",
};

const statusConfig: Record<ChallengeStatus, { icon: React.ReactNode; label: string }> = {
  new: { icon: <Circle className="w-3.5 h-3.5 text-muted-foreground" />, label: "New" },
  attempted: { icon: <Clock className="w-3.5 h-3.5 text-code-yellow" />, label: "Attempted" },
  solved: { icon: <CheckCircle2 className="w-3.5 h-3.5 text-code-green" />, label: "Solved" },
};

const Challenges = () => {
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<ChallengeType | "all">("all");
  const [difficultyFilter, setDifficultyFilter] = useState<Difficulty | "all">("all");
  const [statusFilter, setStatusFilter] = useState<ChallengeStatus | "all">("all");

  const { data: challenges = [], isLoading, error } = useQuery({
    queryKey: ["challenges"],
    queryFn: getProblems,
  });

  // Get user progress
  const { getChallengeStatus, markAsAttempted } = useUserProgress();

  // Merge challenges with user progress
  const challengesWithStatus = useMemo(() => {
    return challenges.map((challenge) => ({
      ...challenge,
      status: getChallengeStatus(challenge.id) as ChallengeStatus,
    }));
  }, [challenges, getChallengeStatus]);

  const filtered = useMemo(() => {
    return challengesWithStatus.filter((c) => {
      if (typeFilter !== "all" && c.type !== typeFilter) return false;
      if (difficultyFilter !== "all" && c.difficulty !== difficultyFilter) return false;
      if (statusFilter !== "all" && c.status !== statusFilter) return false;
      if (search) {
        const q = search.toLowerCase();
        return (
          c.title.toLowerCase().includes(q) ||
          c.tags.some((t) => t.toLowerCase().includes(q)) ||
          c.description.toLowerCase().includes(q)
        );
      }
      return true;
    });
  }, [challengesWithStatus, search, typeFilter, difficultyFilter, statusFilter]);

  const counts = useMemo(() => ({
    all: challengesWithStatus.length,
    LeetCodePrompt: challengesWithStatus.filter((c) => c.type === "LeetCodePrompt").length,
    "Human-SWE-Bench": challengesWithStatus.filter((c) => c.type === "Human-SWE-Bench").length,
    "AI-ML": challengesWithStatus.filter((c) => c.type === "AI-ML").length,
  }), [challengesWithStatus]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <main className="pt-28 pb-20 px-6">
          <div className="container mx-auto max-w-4xl flex items-center justify-center min-h-[400px]">
            <div className="flex flex-col items-center gap-4">
              <Loader2 className="w-8 h-8 animate-spin text-primary" />
              <p className="text-muted-foreground">Loading challenges...</p>
            </div>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <main className="pt-28 pb-20 px-6">
          <div className="container mx-auto max-w-4xl flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <h2 className="text-xl font-bold mb-2 text-destructive">Failed to load challenges</h2>
              <p className="text-muted-foreground mb-4">
                {error instanceof Error ? error.message : "An unknown error occurred"}
              </p>
              <button
                onClick={() => window.location.reload()}
                className="text-primary hover:underline"
              >
                Try again
              </button>
            </div>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header />
      <main className="pt-28 pb-20 px-6">
        <div className="container mx-auto max-w-4xl">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mb-8"
          >
            <h1 className="text-3xl sm:text-4xl font-bold mb-2">Challenges</h1>
            <p className="text-muted-foreground">
              {challenges.length} challenges across {Object.keys(typeConfig).length} categories. Filter, search, and start solving.
            </p>
          </motion.div>

          {/* Search & Filters */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className="space-y-4 mb-8"
          >
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search challenges by name, tag, or description..."
                className="w-full bg-card border border-border rounded-xl pl-11 pr-4 py-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 transition-shadow"
              />
            </div>

            {/* Filter row */}
            <div className="flex flex-wrap gap-3">
              {/* Type filter */}
              <div className="flex items-center gap-1.5">
                <Filter className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="text-xs text-muted-foreground font-medium mr-1">Type</span>
                {(["all", "LeetCodePrompt", "Human-SWE-Bench", "AI-ML"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setTypeFilter(t)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                      typeFilter === t
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary/60 text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {t === "all" ? `All (${counts.all})` : `${typeConfig[t].label} (${counts[t]})`}
                  </button>
                ))}
              </div>

              {/* Difficulty filter */}
              <div className="flex items-center gap-1.5 ml-auto">
                {(["all", "Easy", "Medium", "Hard"] as const).map((d) => (
                  <button
                    key={d}
                    onClick={() => setDifficultyFilter(d)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                      difficultyFilter === d
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary/60 text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {d === "all" ? "All" : d}
                  </button>
                ))}
              </div>

              {/* Status filter */}
              <div className="flex items-center gap-1.5">
                {(["all", "new", "attempted", "solved"] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setStatusFilter(s)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                      statusFilter === s
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary/60 text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {s === "all" ? "All Status" : statusConfig[s].label}
                  </button>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Challenge list */}
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            className="rounded-xl border border-border overflow-hidden"
          >
            {filtered.length === 0 ? (
              <div className="py-16 text-center text-muted-foreground">
                <p className="text-sm">No challenges match your filters.</p>
                <button
                  onClick={() => {
                    setSearch("");
                    setTypeFilter("all");
                    setDifficultyFilter("all");
                    setStatusFilter("all");
                  }}
                  className="text-primary text-sm mt-2 hover:underline"
                >
                  Clear all filters
                </button>
              </div>
            ) : (
              filtered.map((challenge, i) => (
                <ChallengeRow 
                  key={challenge.id} 
                  challenge={challenge} 
                  isFirst={i === 0}
                  markAsAttempted={markAsAttempted}
                />
              ))
            )}
          </motion.div>

          {/* Result count */}
          <p className="text-xs text-muted-foreground mt-4">
            Showing {filtered.length} of {challengesWithStatus.length} challenges
          </p>
        </div>
      </main>
      <Footer />
    </div>
  );
};

/** URL uses problem slug (e.g. astropy-12907, two-sum, Email Spam Detection) */
const getChallengeUrl = (challenge: Challenge) => {
  const prefix = challenge.type === "LeetCodePrompt" ? "lcp" : challenge.type === "AI-ML" ? "aiml" : "hsb";
  return `/challenges/${prefix}/${encodeURIComponent(challenge.title)}`;
};

const ChallengeRow = ({
  challenge,
  isFirst,
  markAsAttempted,
}: {
  challenge: Challenge;
  isFirst: boolean;
  markAsAttempted: (problemSlug: string) => void;
}) => {
  const type = typeConfig[challenge.type];
  const status = statusConfig[challenge.status];

  const handleClick = () => {
    // Mark as attempted when user clicks on challenge
    if (challenge.status === "new") {
      markAsAttempted(challenge.id);
    }
  };

  return (
    <Link
      to={getChallengeUrl(challenge)}
      onClick={handleClick}
      className={`group flex items-center gap-4 px-5 py-4 hover:bg-secondary/20 transition-colors cursor-pointer ${
        !isFirst ? "border-t border-border/60" : ""
      }`}
    >
      {/* Status icon */}
      <div className="shrink-0">{status.icon}</div>

      {/* Main content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <h3 className="text-sm font-medium truncate">{challenge.title}</h3>
          <span className={`text-xs font-mono font-semibold ${difficultyColor[challenge.difficulty]}`}>
            {challenge.difficulty}
          </span>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`inline-flex items-center gap-1 text-xs ${type.color}`}>
            {type.icon}
            {type.label}
          </span>
          {challenge.tags.slice(0, 3).map((tag) => (
            <Badge
              key={tag}
              variant="secondary"
              className="text-[10px] px-1.5 py-0 font-normal"
            >
              {tag}
            </Badge>
          ))}
        </div>
      </div>

      {/* Acceptance rate */}
      <div className="hidden sm:flex flex-col items-end shrink-0">
        <span className="text-xs text-muted-foreground font-mono">{challenge.acceptance}%</span>
        <div className="w-16 h-1.5 rounded-full bg-secondary mt-1.5 overflow-hidden">
          <div
            className="h-full rounded-full bg-primary/60"
            style={{ width: `${challenge.acceptance}%` }}
          />
        </div>
      </div>

      {/* Arrow */}
      <ChevronRight className="w-4 h-4 text-muted-foreground/40 group-hover:text-foreground/60 transition-colors shrink-0" />
    </Link>
  );
};

export default Challenges;
