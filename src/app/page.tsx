'use client'

import dynamic from 'next/dynamic'

// SSR無効化でブラウザAPI使用コンポーネントを保護
const StressEstimationApp = dynamic(() => import('@/components/StressEstimationApp'), {
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-800 mb-4">次世代ストレス推定システム</h1>
        <p className="text-gray-600">読み込み中...</p>
      </div>
    </div>
  )
})

export default function Home() {
  return <StressEstimationApp />
}