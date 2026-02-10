import { useAuth } from "@/hooks/use-auth";
import { useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { LogOut, Github, Mail, User as UserIcon, Copy, Check, Loader2 } from "lucide-react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { fetchUserIdFromBackend, getCurrentUser } from "@/lib/auth";

const Profile = () => {
  const { user, isAuthenticated, logout, setUser } = useAuth();
  const navigate = useNavigate();
  const [copied, setCopied] = useState(false);
  const [userId, setUserId] = useState<string | null>((user as any)?.userId || null);
  const [loadingUserId, setLoadingUserId] = useState(false);

  useEffect(() => {
    if (!isAuthenticated) {
      navigate("/");
    }
  }, [isAuthenticated, navigate]);

  useEffect(() => {
    // If userId is not available, try to fetch it from backend
    if (isAuthenticated && user && !userId) {
      setLoadingUserId(true);
      fetchUserIdFromBackend(user.login)
        .then((fetchedUserId) => {
          if (fetchedUserId) {
            setUserId(fetchedUserId);
            // Update localStorage with userId
            const currentUser = getCurrentUser();
            if (currentUser) {
              currentUser.userId = fetchedUserId;
              localStorage.setItem("github_user", JSON.stringify(currentUser));
              setUser(currentUser);
            }
          }
        })
        .catch((error) => {
          console.error("Failed to fetch user ID:", error);
        })
        .finally(() => {
          setLoadingUserId(false);
        });
    }
  }, [isAuthenticated, user, userId, setUser]);

  if (!isAuthenticated || !user) {
    return null;
  }

  const displayUserId = userId || (loadingUserId ? "Loading..." : "Not available");

  const handleCopyUserId = () => {
    if (userId && userId !== "Not available" && !loadingUserId) {
      navigator.clipboard.writeText(userId);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Header />
      <main className="flex-1 pt-16">
        <div className="container mx-auto px-6 py-12 max-w-4xl">
          <h1 className="text-3xl font-bold mb-8">Profile</h1>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-4">
                <Avatar className="h-20 w-20">
                  <AvatarImage src={user.avatar_url} alt={user.login} />
                  <AvatarFallback className="text-2xl">
                    {user.login.charAt(0).toUpperCase()}
                  </AvatarFallback>
                </Avatar>
                <div>
                  <CardTitle className="text-2xl">{user.name || user.login}</CardTitle>
                  <CardDescription className="text-base">@{user.login}</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* User ID Section */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">User ID</label>
                <div className="flex items-center gap-2">
                  <div className="flex-1 px-3 py-2 bg-muted rounded-md font-mono text-sm flex items-center gap-2">
                    {loadingUserId ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Loading...
                      </>
                    ) : (
                      displayUserId
                    )}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleCopyUserId}
                    disabled={!userId || userId === "Not available" || loadingUserId}
                  >
                    {copied ? (
                      <>
                        <Check className="w-4 h-4 mr-2" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="w-4 h-4 mr-2" />
                        Copy
                      </>
                    )}
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  Your unique user ID for AI Coding Gym. Use this ID when configuring your account.
                </p>
              </div>

              <div className="border-t pt-6 space-y-4">
                {/* GitHub Username */}
                <div className="flex items-center gap-3">
                  <Github className="w-5 h-5 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium">GitHub Username</p>
                    <p className="text-sm text-muted-foreground">{user.login}</p>
                  </div>
                </div>

                {/* Email */}
                {user.email && (
                  <div className="flex items-center gap-3">
                    <Mail className="w-5 h-5 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">Email</p>
                      <p className="text-sm text-muted-foreground">{user.email}</p>
                    </div>
                  </div>
                )}

                {/* Display Name */}
                {user.name && (
                  <div className="flex items-center gap-3">
                    <UserIcon className="w-5 h-5 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">Display Name</p>
                      <p className="text-sm text-muted-foreground">{user.name}</p>
                    </div>
                  </div>
                )}
              </div>

              <div className="border-t pt-6">
                <Button variant="destructive" onClick={logout} className="w-full">
                  <LogOut className="w-4 h-4 mr-2" />
                  Log out
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Profile;
