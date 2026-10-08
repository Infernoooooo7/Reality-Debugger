import { useEffect, useRef, useState } from 'react'

const GLYPHS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%&/<>'

/**
 * Text that "decodes" into its new value whenever it changes, like a status
 * readout re-latching. Purely cosmetic; settles in ~350 ms.
 */
export function DecodeText({ text, className }: { text: string; className?: string }) {
  const [shown, setShown] = useState(text)
  const frame = useRef<number | null>(null)

  useEffect(() => {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      setShown(text)
      return
    }
    const start = performance.now()
    const duration = 360
    const step = (now: number) => {
      const progress = Math.min(1, (now - start) / duration)
      const fixed = Math.floor(progress * text.length)
      let out = text.slice(0, fixed)
      for (let i = fixed; i < text.length; i++) {
        const ch = text[i]!
        out += ch === ' ' ? ' ' : GLYPHS[(Math.random() * GLYPHS.length) | 0]
      }
      setShown(out)
      if (progress < 1) frame.current = requestAnimationFrame(step)
    }
    frame.current = requestAnimationFrame(step)
    return () => {
      if (frame.current) cancelAnimationFrame(frame.current)
    }
  }, [text])

  return (
    <span className={className} aria-label={text}>
      <span aria-hidden="true">{shown}</span>
    </span>
  )
}
