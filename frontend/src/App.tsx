import { lazy, Suspense, useEffect } from 'react'
import { Toasts } from './components/Toasts'
import { useRoute } from './lib/router'
import { Home } from './screens/Home'
import { useSystem } from './state/system'

const LiveScan = lazy(() => import('./screens/LiveScan'))
const ImageDebug = lazy(() => import('./screens/ImageDebug'))
const VideoDebug = lazy(() => import('./screens/VideoDebug'))

function Loading() {
  return (
    <div className="screen-loading">
      <span className="t-label">Loading module…</span>
      <div className="spinner-bar" />
    </div>
  )
}

export function App() {
  const route = useRoute()

  useEffect(() => {
    void useSystem.getState().refreshHealth()
  }, [])

  useEffect(() => {
    window.scrollTo(0, 0)
  }, [route])

  return (
    <>
      <Suspense fallback={<Loading />}>
        {route === 'home' ? <Home /> : null}
        {route === 'live' ? <LiveScan /> : null}
        {route === 'image' ? <ImageDebug /> : null}
        {route === 'video' ? <VideoDebug /> : null}
      </Suspense>
      <Toasts />
    </>
  )
}
