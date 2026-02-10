import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import Blog from "./pages/Blog";
import BlogPost from "./pages/BlogPost";
import Challenges from "./pages/Challenges";
import LeetCodeChallenge from "./pages/LeetCodeChallenge";
import SWEBenchChallenge from "./pages/SWEBenchChallenge";
import AIMLChallenge from "./pages/AIMLChallenge";
import Leaderboard from "./pages/Leaderboard";
import AuthCallback from "./pages/AuthCallback";
import Profile from "./pages/Profile";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/blog" element={<Blog />} />
          <Route path="/challenges" element={<Challenges />} />
          <Route path="/challenges/lcp/:id" element={<LeetCodeChallenge />} />
          <Route path="/challenges/hsb/:id" element={<SWEBenchChallenge />} />
          <Route path="/challenges/aiml/:id" element={<AIMLChallenge />} />
          <Route path="/blog/:slug" element={<BlogPost />} />
          <Route path="/leaderboard" element={<Leaderboard />} />
          <Route path="/auth/callback" element={<AuthCallback />} />
          <Route path="/profile" element={<Profile />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
