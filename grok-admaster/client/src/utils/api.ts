/**
 * Shared API utility helpers.
 */

/**
 * Build query params that include an optional account_id filter.
 * Eliminates the repeated `accountId ? { account_id: accountId } : {}` pattern
 * that appears across every API endpoint file.
 */
export function buildAccountParams(
  accountId?: number,
  extra: Record<string, string | number> = {}
): Record<string, string | number> {
  return accountId != null ? { account_id: accountId, ...extra } : { ...extra }
}
