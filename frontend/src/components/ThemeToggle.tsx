import { Moon, Sun } from "lucide-react";

interface ThemeToggleProps {
  codeTheme: "dark" | "light";
  onToggle: () => void;
}

const ThemeToggle = ({ codeTheme, onToggle }: ThemeToggleProps) => {
  return (
    <button
      onClick={onToggle}
      className="p-1.5 rounded-md bg-code-border/50 hover:bg-code-border text-code-muted hover:text-code-foreground transition-colors"
      aria-label="Toggle theme"
    >
      {codeTheme === "dark" ? (
        <Sun className="w-4 h-4" />
      ) : (
        <Moon className="w-4 h-4" />
      )}
    </button>
  );
};

export default ThemeToggle;
