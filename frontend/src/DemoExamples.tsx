import { useEffect, useState } from 'react'

const examples = [
  {
    id: 'side-stripe',
    title: '01 · 옆선 더하기',
    query: '이 사진의 트레이닝 팬츠와 비슷한데, 옆에 라인이 있는 바지 찾아줘.',
    alt: '베이지 트레이닝 팬츠를 입은 사람',
  },
  {
    id: 'short-sleeve-shirt',
    title: '02 · 반팔 셔츠로 바꾸기',
    query: '이 사진의 셔츠와 비슷한데, 소매가 짧은 반팔 셔츠로 찾아줘.',
    alt: '세로 줄무늬 긴소매 셔츠',
  },
]

type Props = {
  disabled: boolean
  onSelect: (query: string, file: File) => void
}

export function DemoExamples({ disabled, onSelect }: Props) {
  const [files, setFiles] = useState<Record<string, File>>({})
  const [failed, setFailed] = useState<Record<string, boolean>>({})
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    for (const example of examples) {
      void (async () => {
        try {
          const response = await fetch(`/demo/${example.id}.jpg`, { signal: controller.signal })
          if (!response.ok) throw new Error('사진을 불러오지 못했습니다.')
          const blob = await response.blob()
          if (!blob.type.startsWith('image/')) throw new Error('이미지가 아닙니다.')
          if (controller.signal.aborted) return
          const file = new File([blob], `${example.id}.jpg`, { type: blob.type })
          setFiles((current) => ({ ...current, [example.id]: file }))
          setFailed((current) => ({ ...current, [example.id]: false }))
        } catch {
          if (!controller.signal.aborted) setFailed((current) => ({ ...current, [example.id]: true }))
        }
      })()
    }
    return () => controller.abort()
  }, [attempt])

  return (
    <section className="f1t-demo-examples" aria-label="사진으로 검색해 보기">
      <p>사진으로 바로 시작해 보세요 <span>선택하면 사진과 요청이 함께 입력돼요.</span></p>
      <div className="f1t-demo-grid">
        {examples.map((example) => (
          <button key={example.id} type="button" className="f1t-demo-card"
            disabled={disabled || (!files[example.id] && !failed[example.id])}
            onClick={() => {
              if (files[example.id]) onSelect(example.query, files[example.id])
              else {
                setFailed({})
                setAttempt((current) => current + 1)
              }
            }}>
            <img src={`/demo/${example.id}.jpg`} alt={example.alt} />
            <span className="f1t-demo-copy">
              <strong>{example.title}</strong>
              <span>{example.query}</span>
              <small>{files[example.id] ? '사진·요청 불러오기 ↗' : failed[example.id] ? '사진 로드 실패 · 다시 시도' : '사진 불러오는 중…'}</small>
            </span>
          </button>
        ))}
      </div>
    </section>
  )
}
