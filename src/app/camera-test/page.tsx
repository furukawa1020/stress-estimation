'use client'

import dynamic from 'next/dynamic'

// SSR無効化でクライアント専用コンポーネントとして読み込み
const SimpleCameraTest = dynamic(() => import('@/components/SimpleCameraTest'), {
  ssr: false,
  loading: () => <div className="p-8 text-center">カメラテストを読み込み中...</div>
})

export default function CameraTestPage() {
  return <SimpleCameraTest />
}