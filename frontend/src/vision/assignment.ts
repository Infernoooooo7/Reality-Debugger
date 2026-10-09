/**
 * Optimal bipartite matching with a cost limit, equivalent to the
 * `lap.lapjv(cost, extend_cost=True, cost_limit=thresh)` call ByteTrack uses:
 * the cost matrix is extended with "leave unmatched" entries of cost
 * thresh/2 on both sides, so a pair is only matched when that is cheaper than
 * leaving both members unmatched (cost < thresh), and the total cost of the
 * matching is minimal. Solved with the Hungarian algorithm (O(n^3)); our
 * matrices are at most a few dozen rows.
 */

export interface Assignment {
  matches: [number, number][]
  unmatchedRows: number[]
  unmatchedCols: number[]
}

const INF = 1e9

/** Hungarian algorithm on a square matrix (row-major); returns col assigned to each row. */
function hungarian(cost: Float64Array, n: number): Int32Array {
  // e-maxx formulation with 1-based potentials.
  const u = new Float64Array(n + 1)
  const v = new Float64Array(n + 1)
  const p = new Int32Array(n + 1)
  const way = new Int32Array(n + 1)
  for (let i = 1; i <= n; i++) {
    p[0] = i
    let j0 = 0
    const minv = new Float64Array(n + 1).fill(Infinity)
    const used = new Uint8Array(n + 1)
    do {
      used[j0] = 1
      const i0 = p[j0]!
      let delta = Infinity
      let j1 = 0
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue
        const cur = cost[(i0 - 1) * n + (j - 1)]! - u[i0]! - v[j]!
        if (cur < minv[j]!) {
          minv[j] = cur
          way[j] = j0
        }
        if (minv[j]! < delta) {
          delta = minv[j]!
          j1 = j
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]!] = u[p[j]!]! + delta
          v[j] = v[j]! - delta
        } else {
          minv[j] = minv[j]! - delta
        }
      }
      j0 = j1
    } while (p[j0] !== 0)
    do {
      const j1 = way[j0]!
      p[j0] = p[j1]!
      j0 = j1
    } while (j0)
  }
  const rowToCol = new Int32Array(n).fill(-1)
  for (let j = 1; j <= n; j++) if (p[j]! > 0) rowToCol[p[j]! - 1] = j - 1
  return rowToCol
}

/**
 * Match rows to columns of `cost` (rows x cols) where only pairs with
 * cost <= thresh may be matched. `cols` must be given explicitly because an
 * empty cost matrix (no rows) cannot tell how many columns there are.
 */
export function linearAssignment(cost: number[][], thresh: number, cols: number): Assignment {
  const rows = cost.length
  if (!rows || !cols) {
    return { matches: [], unmatchedRows: [...Array(rows).keys()], unmatchedCols: [...Array(cols).keys()] }
  }
  const n = rows + cols
  const big = new Float64Array(n * n).fill(INF)
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const c = cost[i]![j]!
      big[i * n + j] = c > thresh ? INF : c
    }
    big[i * n + cols + i] = thresh / 2 // row i stays unmatched
  }
  for (let j = 0; j < cols; j++) {
    big[(rows + j) * n + j] = thresh / 2 // column j stays unmatched
    for (let k = 0; k < rows; k++) big[(rows + j) * n + cols + k] = 0 // dummy-dummy
  }
  const rowToCol = hungarian(big, n)
  const matches: [number, number][] = []
  const matchedRows = new Set<number>()
  const matchedCols = new Set<number>()
  for (let i = 0; i < rows; i++) {
    const j = rowToCol[i]!
    if (j >= 0 && j < cols && cost[i]![j]! <= thresh) {
      matches.push([i, j])
      matchedRows.add(i)
      matchedCols.add(j)
    }
  }
  return {
    matches,
    unmatchedRows: [...Array(rows).keys()].filter((i) => !matchedRows.has(i)),
    unmatchedCols: [...Array(cols).keys()].filter((j) => !matchedCols.has(j)),
  }
}
