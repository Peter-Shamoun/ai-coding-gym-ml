import { useState, useEffect } from "react";
import { getCurrentUser, isLoggedIn, logout as authLogout } from "@/lib/auth";

interface User {
  login: string;
  avatar_url: string;
  name?: string;
  email?: string;
}

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const currentUser = getCurrentUser();
    setUser(currentUser);
    setLoading(false);
  }, []);

  const logout = () => {
    authLogout();
    setUser(null);
  };

  return {
    user,
    isAuthenticated: isLoggedIn(),
    loading,
    logout,
    setUser,
  };
}
