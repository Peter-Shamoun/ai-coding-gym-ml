import { motion } from "framer-motion";
import { MessageSquare, Bug, ShieldCheck, Brain, BarChart3 } from "lucide-react";

const skillPills = [
  { label: "Frame Tasks for AI", icon: MessageSquare },
  { label: "Solve What AI Can't", icon: Brain },
  { label: "Debug with AI", icon: Bug },
  { label: "Verify AI Code", icon: ShieldCheck },
  { label: "AI / ML Challenges", icon: BarChart3 },
];

const HeroSection = () => {
  return (
    <section className="pt-32 pb-12 px-6">
      <div className="container mx-auto max-w-3xl text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-secondary text-secondary-foreground text-sm font-medium mb-6">
            <span className="w-2 h-2 rounded-full bg-primary" />
            Now in Beta
          </div>
        </motion.div>

        <motion.h1
          className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight text-foreground leading-[1.1] mb-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
        >
          Practice Your{" "}
          <span className="text-primary">AI Coding</span>
          {" "}Skills
        </motion.h1>

        <motion.p
          className="text-lg sm:text-xl text-muted-foreground max-w-xl mx-auto leading-relaxed mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
        >
          Task-Based Challenges to sharpen how you work with AI.
        </motion.p>

        <motion.div
          className="flex flex-wrap items-center justify-start gap-3 mb-10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.25, ease: "easeOut" }}
        >
          {skillPills.map((pill) => (
            <div
              key={pill.label}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-border bg-card text-sm text-foreground font-medium"
            >
              <pill.icon className="w-4 h-4 text-primary" />
              {pill.label}
            </div>
          ))}
          <span className="inline-flex items-center text-sm text-muted-foreground">
            …and more is coming
          </span>
        </motion.div>

        <motion.div
          className="flex items-center justify-center gap-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.35, ease: "easeOut" }}
        >
          <a
            href="#example"
            className="inline-flex items-center justify-center rounded-lg bg-primary text-primary-foreground px-6 py-3 text-sm font-medium hover:opacity-90 transition-opacity"
          >
            Try a Challenge
          </a>
          <a
            href="#docs"
            className="inline-flex items-center justify-center rounded-lg border border-border bg-background text-foreground px-6 py-3 text-sm font-medium hover:bg-secondary transition-colors"
          >
            Learn More
          </a>
        </motion.div>
      </div>
    </section>
  );
};

export default HeroSection;
