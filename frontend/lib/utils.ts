// utils.ts
// Utility functions for the frontend

/**
 * cn - Combines class names conditionally
 * Usage: cn('a', condition && 'b', ...)
 */
export function cn(...inputs: Array<string | undefined | false | null>): string {
  return inputs.filter(Boolean).join(' ')
}
