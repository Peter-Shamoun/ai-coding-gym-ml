import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { handleOAuthCallback } from "@/lib/auth";
import { Loader2 } from "lucide-react";

const AuthCallback = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const code = searchParams.get("code");
    const state = searchParams.get("state");
    const errorParam = searchParams.get("error");

    if (errorParam) {
      setError(errorParam);
      setTimeout(() => navigate("/"), 3000);
      return;
    }

    if (!code || !state) {
      setError("Missing authorization code");
      setTimeout(() => navigate("/"), 3000);
      return;
    }

    handleOAuthCallback(code, state)
      .then((result) => {
        if (result.success) {
          navigate("/");
        } else {
          setError(result.error || "Authentication failed");
          setTimeout(() => navigate("/"), 3000);
        }
      })
      .catch((err) => {
        setError(err.message || "Authentication failed");
        setTimeout(() => navigate("/"), 3000);
      });
  }, [searchParams, navigate]);

  if (error) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4 text-destructive">Authentication Error</h1>
          <p className="text-muted-foreground mb-4">{error}</p>
          <p className="text-sm text-muted-foreground">Redirecting to home...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-4" />
        <p className="text-muted-foreground">Completing authentication...</p>
      </div>
    </div>
  );
};

export default AuthCallback;
