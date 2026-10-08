import { useEffect, useRef, useState } from 'react'

const GLYPHS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%&/<>'

const reducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false

/**
 * Text that "decodes" into its new value whenever it changes, like a status
 * readout re-latching. Purely cosmetic; settles in ~350 ms.
 */
export function DecodeText({ text, className }: { text: string; className?: string }) {
  const [frame, setFrame] = useState<{ source: string; shown: string }>({ source: text, shown: text })
  const raf = useRef<number | null>(null)

  useEffect(() => {
    if (reducedMotion()) return
    const start = performance.now()
    const duration = 360
    const step = (now: number) => {
      const progress = Math.min(1, (now - start) / duration)
      const fixed = Math.floor(progress * text.length)
      let out = text.slice(0, fixed)
      for (let i = fixed; i < text.length; i++) {
        out += text[i] === ' ' ? ' ' : GLYPHS[(Math.random() * GLYPHS.length) | 0]
      }
      setFrame({ source: text, shown: out })
      if (progress < 1) raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current)
    }
  }, [text])

  // Until the first animation frame for a new text arrives, show it as-is.
  const shown = frame.source === text ? frame.shown : text
  return (
    <span className={className} aria-label={text}>
      <span aria-hidden="true">{shown}</span>
    </span>
  )
}
