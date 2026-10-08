import { useEffect, useState } from 'react'

export type Route = 'home' | 'live' | 'image' | 'video'

const ROUTES: Record<string, Route> = {
  '': 'home',
  '/': 'home',
  '/live': 'live',
  '/image': 'image',
  '/video': 'video',
}

function parse(hash: string): Route {
  const path = hash.replace(/^#/, '').split('?')[0] ?? ''
  return ROUTES[path] ?? 'home'
}

export function navigate(route: Route): void {
  const target = route === 'home' ? '#/' : `#/${route}`
  if (window.location.hash !== target) window.location.hash = target
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash))
  useEffect(() => {
    const onChange = () => setRoute(parse(window.location.hash))
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}
