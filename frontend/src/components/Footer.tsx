const Footer = () => {
  return (
    <footer className="border-t border-border py-10 px-6">
      <div className="container mx-auto max-w-6xl flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <div className="w-5 h-5 rounded bg-primary flex items-center justify-center">
            <span className="text-primary-foreground font-mono text-[10px] font-bold">&gt;_</span>
          </div>
          <span>AICodingGym © 2026</span>
        </div>
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <a href="#" className="hover:text-foreground transition-colors">Twitter</a>
          <a href="#" className="hover:text-foreground transition-colors">GitHub</a>
          <a href="#" className="hover:text-foreground transition-colors">Discord</a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
