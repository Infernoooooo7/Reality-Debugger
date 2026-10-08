import '@fontsource-variable/big-shoulders-display'
import '@fontsource-variable/instrument-sans'
import '@fontsource-variable/martian-mono/standard.css'
import './styles/tokens.css'
import './styles/base.css'
import './styles/components.css'
import './styles/report.css'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
