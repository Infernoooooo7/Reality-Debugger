/** Minimal line icons drawn on a 24px grid (square caps: instrument style). */
const PATHS: Record<string, string> = {
  arrow: 'M4 12h15M13 6l6 6-6 6',
  back: 'M20 12H5M11 6l-6 6 6 6',
  pause: 'M8 5v14M16 5v14',
  play: 'M7 4.5v15l12-7.5z',
  flip: 'M4 9h13l-3-3M20 15H7l3 3',
  close: 'M5 5l14 14M19 5L5 19',
  check: 'M4 12.5l5 5L20 6',
  copy: 'M8 8h11v11H8zM5 16V5h11',
  download: 'M12 4v11M7 10l5 5 5-5M5 20h14',
  share: 'M12 3v12M7 8l5-5 5 5M5 13v7h14v-7',
  scan: 'M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4M4 12h16',
  image: 'M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4',
  video: 'M3 6h13v12H3zM16 10l5-3v10l-5-3',
  trash: 'M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13',
  retry: 'M20 11a8 8 0 1 0-2.3 5.7M20 5v6h-6',
  eye: 'M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 9v6M9 12h6',
  upload: 'M12 20V9M7 14l5-5 5 5M5 4h14',
}

export function Icon({ name, size = 18, strokeWidth = 2 }: { name: keyof typeof PATHS | string; size?: number; strokeWidth?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="square"
      strokeLinejoin="miter"
      aria-hidden="true"
      focusable="false"
    >
      <path d={PATHS[name] ?? ''} />
    </svg>
  )
}
