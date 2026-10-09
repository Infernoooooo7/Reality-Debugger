import { describe, expect, it } from 'vitest'
import { linearAssignment } from '../assignment'

describe('linearAssignment', () => {
  it('leaves every column unmatched when there are no rows', () => {
    // Regression: an empty cost matrix used to report no unmatched columns, so
    // the tracker never started a track on the first frame.
    expect(linearAssignment([], 0.8, 3)).toEqual({ matches: [], unmatchedRows: [], unmatchedCols: [0, 1, 2] })
  })

  it('leaves every row unmatched when there are no columns', () => {
    expect(linearAssignment([[], []], 0.8, 0)).toEqual({ matches: [], unmatchedRows: [0, 1], unmatchedCols: [] })
  })

  it('finds the optimal matching where a greedy one fails', () => {
    // Greedy takes (0,0)=0.1 and is left with (1,1)=0.9 > thresh; the optimum is 0.2 + 0.15.
    const result = linearAssignment(
      [
        [0.1, 0.2],
        [0.15, 0.9],
      ],
      0.5,
      2,
    )
    expect(result.matches.sort()).toEqual([
      [0, 1],
      [1, 0],
    ])
    expect(result.unmatchedRows).toEqual([])
    expect(result.unmatchedCols).toEqual([])
  })

  it('never matches a pair above the threshold', () => {
    const result = linearAssignment(
      [
        [0.6, 0.95],
        [0.7, 0.2],
      ],
      0.5,
      2,
    )
    expect(result.matches).toEqual([[1, 1]])
    expect(result.unmatchedRows).toEqual([0])
    expect(result.unmatchedCols).toEqual([0])
  })

  it('handles rectangular matrices', () => {
    const result = linearAssignment([[0.9, 0.1, 0.3]], 0.5, 3)
    expect(result.matches).toEqual([[0, 1]])
    expect(result.unmatchedCols).toEqual([0, 2])
  })
})
