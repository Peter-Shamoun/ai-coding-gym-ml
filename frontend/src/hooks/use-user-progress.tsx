import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "./use-auth";
import { getCurrentUser } from "@/lib/auth";

const backendUrl = import.meta.env.VITE_BACKEND_URL || (import.meta.env.PROD ? '' : 'http://localhost:3001');

/**
 * Fetch user progress from backend
 */
async function fetchUserProgress(userId: string) {
  const response = await fetch(`${backendUrl}/api/users/${userId}/progress`);
  if (!response.ok) {
    throw new Error("Failed to fetch user progress");
  }
  return response.json();
}

/**
 * Update user progress
 */
async function updateUserProgress(userId: string, problemSlug: string, status: "attempted" | "solved") {
  const response = await fetch(`${backendUrl}/api/users/${userId}/progress`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ problemSlug, status }),
  });
  if (!response.ok) {
    throw new Error("Failed to update user progress");
  }
  return response.json();
}

/**
 * Hook to get and update user progress
 */
export function useUserProgress() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const userId = (user as any)?.userId;

  // Fetch user progress
  const { data: progressData, isLoading } = useQuery({
    queryKey: ["userProgress", userId],
    queryFn: () => fetchUserProgress(userId!),
    enabled: !!userId,
    staleTime: 30000, // Cache for 30 seconds
  });

  // Mutation to update progress
  const updateProgress = useMutation({
    mutationFn: ({ problemSlug, status }: { problemSlug: string; status: "attempted" | "solved" }) =>
      updateUserProgress(userId!, problemSlug, status),
    onSuccess: () => {
      // Invalidate and refetch progress
      queryClient.invalidateQueries({ queryKey: ["userProgress", userId] });
    },
  });

  // Helper functions
  const getChallengeStatus = (challengeId: string): "new" | "attempted" | "solved" => {
    if (!progressData?.progress || !progressData.progress[challengeId]) {
      return "new";
    }
    const status = progressData.progress[challengeId].status;
    return status === "solved" ? "solved" : "attempted";
  };

  const markAsAttempted = (problemSlug: string) => {
    if (userId) {
      updateProgress.mutate({ problemSlug, status: "attempted" });
    }
  };

  const markAsSolved = (problemSlug: string) => {
    if (userId) {
      updateProgress.mutate({ problemSlug, status: "solved" });
    }
  };

  return {
    progress: progressData?.progress || {},
    solvedProblems: new Set(progressData?.solvedProblems || []),
    attemptedProblems: new Set(progressData?.attemptedProblems || []),
    isLoading,
    getChallengeStatus,
    markAsAttempted,
    markAsSolved,
    updateProgress,
  };
}
