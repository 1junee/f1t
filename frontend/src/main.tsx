import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// index.html의 root에 검색 화면(App)을 연결하는 브라우저 진입점이다.
// 개발 모드의 StrictMode는 effect 실행·정리를 추가로 점검한다. 정리 함수가 필요한 이유다.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
