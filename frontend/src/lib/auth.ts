/**
 * GitHub OAuth configuration and utilities
 */

const GITHUB_CLIENT_ID = "Ov23lijijJtsBuUbS9kd";
const GITHUB_REDIRECT_URI = `${window.location.origin}/auth/callback`;
const GITHUB_AUTH_URL = `https://github.com/login/oauth/authorize`;
const GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token";

/**
 * Generate a random state string for OAuth security
 */
function generateState(): string {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}

/**
 * Initiate GitHub OAuth login
 */
export function loginWithGitHub(): void {
  const state = generateState();
  // Store state in sessionStorage for verification
  sessionStorage.setItem("github_oauth_state", state);

  const params = new URLSearchParams({
    client_id: GITHUB_CLIENT_ID,
    redirect_uri: GITHUB_REDIRECT_URI,
    scope: "read:user user:email",
    state: state,
  });

  window.location.href = `${GITHUB_AUTH_URL}?${params.toString()}`;
}

/**
 * Handle OAuth callback and exchange code for token
 * Note: This requires a backend endpoint to exchange the code for an access token
 * using the client secret. See backend-api-example.js for implementation.
 */
export async function handleOAuthCallback(code: string, state: string): Promise<{ success: boolean; error?: string }> {
  // Verify state
  const storedState = sessionStorage.getItem("github_oauth_state");
  if (!storedState || storedState !== state) {
    return { success: false, error: "Invalid state parameter" };
  }

  sessionStorage.removeItem("github_oauth_state");

  try {
    // Call backend API to exchange code for token
    // The backend should handle the token exchange using the client secret
    // Use relative path in production (nginx proxies /api to backend), or use env variable
    const backendUrl = import.meta.env.VITE_BACKEND_URL || (import.meta.env.PROD ? '' : 'http://localhost:3001');
    const response = await fetch(`${backendUrl}/api/auth/github/callback`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ code }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: "Failed to authenticate" }));
      throw new Error(errorData.error || "Failed to authenticate");
    }

    const data = await response.json();
    
    // Store user info and token
    if (data.access_token) {
      localStorage.setItem("github_access_token", data.access_token);
      
      // If backend returns user info with userId, use it directly
      if (data.user && data.user.userId) {
        localStorage.setItem("github_user", JSON.stringify(data.user));
      } else {
        // Otherwise fetch user info from GitHub and merge with userId if available
        const userInfo = await fetchUserInfo(data.access_token);
        if (data.user?.userId) {
          userInfo.userId = data.user.userId;
        }
        localStorage.setItem("github_user", JSON.stringify(userInfo));
      }
    } else if (data.user) {
      // If backend returns user info directly
      localStorage.setItem("github_user", JSON.stringify(data.user));
      if (data.access_token) {
        localStorage.setItem("github_access_token", data.access_token);
      }
    } else {
      throw new Error("No access token or user info received");
    }

    return { success: true };
  } catch (error) {
    console.error("OAuth callback error:", error);
    return { success: false, error: error instanceof Error ? error.message : "Authentication failed" };
  }
}

/**
 * Get current user from localStorage
 */
export function getCurrentUser(): { login: string; avatar_url: string; name?: string; email?: string; userId?: string } | null {
  const userStr = localStorage.getItem("github_user");
  if (!userStr) return null;
  
  try {
    return JSON.parse(userStr);
  } catch {
    return null;
  }
}

/**
 * Fetch user ID from backend if not available in localStorage
 */
export async function fetchUserIdFromBackend(githubId: string): Promise<string | null> {
  try {
    const backendUrl = import.meta.env.VITE_BACKEND_URL || (import.meta.env.PROD ? '' : 'http://localhost:3001');
    const url = `${backendUrl}/api/users?githubId=${encodeURIComponent(githubId)}`;
    console.log("Fetching user ID from:", url);
    
    const response = await fetch(url);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Failed to fetch user ID: ${response.status} ${response.statusText}`, errorText);
      throw new Error(`Failed to fetch user ID: ${response.status}`);
    }
    
    const data = await response.json();
    console.log("User ID response:", data);
    return data.userId || null;
  } catch (error) {
    console.error("Error fetching user ID:", error);
    return null;
  }
}

/**
 * Get GitHub access token
 */
export function getAccessToken(): string | null {
  return localStorage.getItem("github_access_token");
}

/**
 * Check if user is logged in
 */
export function isLoggedIn(): boolean {
  return getCurrentUser() !== null;
}

/**
 * Logout user
 */
export function logout(): void {
  localStorage.removeItem("github_access_token");
  localStorage.removeItem("github_user");
}

/**
 * Get user info from GitHub API
 */
export async function fetchUserInfo(token: string): Promise<any> {
  const response = await fetch("https://api.github.com/user", {
    headers: {
      Authorization: `token ${token}`,
      Accept: "application/vnd.github.v3+json",
    },
  });

  if (!response.ok) {
    throw new Error("Failed to fetch user info");
  }

  return response.json();
}
