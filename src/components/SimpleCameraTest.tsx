'use client'

import React, { useState, useRef, useEffect } from 'react'

export default function SimpleCameraTest() {
  const [isStreamActive, setIsStreamActive] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  // デバイス列挙
  useEffect(() => {
    const getDevices = async () => {
      try {
        const deviceList = await navigator.mediaDevices.enumerateDevices()
        const videoDevices = deviceList.filter(device => device.kind === 'videoinput')
        setDevices(videoDevices)
        console.log('利用可能なカメラデバイス:', videoDevices)
      } catch (err) {
        console.error('デバイス列挙エラー:', err)
      }
    }
    getDevices()
  }, [])

  // シンプルなカメラ起動
  const startCamera = async () => {
    try {
      setError(null)
      console.log('カメラアクセス開始...')

      // 最もシンプルな設定
      const constraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      }

      console.log('制約条件:', constraints)

      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      
      console.log('ストリーム取得成功:', stream)
      console.log('ビデオトラック:', stream.getVideoTracks())

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setIsStreamActive(true)
        console.log('ビデオ要素にストリーム設定完了')
      }

    } catch (err) {
      console.error('カメラアクセスエラー:', err)
      
      if (err instanceof Error) {
        let errorMessage = `カメラアクセスエラー: ${err.name}`
        
        switch (err.name) {
          case 'NotAllowedError':
            errorMessage = 'カメラアクセスが拒否されました。ブラウザでカメラアクセスを許可してください。'
            break
          case 'NotFoundError':
            errorMessage = 'カメラデバイスが見つかりません。'
            break
          case 'NotReadableError':
            errorMessage = 'カメラが他のアプリケーションで使用中です。'
            break
          case 'OverconstrainedError':
            errorMessage = 'カメラの設定要求が厳しすぎます。'
            break
          case 'SecurityError':
            errorMessage = 'セキュリティエラー: HTTPSが必要な場合があります。'
            break
          default:
            errorMessage = `不明なエラー: ${err.message}`
        }
        
        setError(errorMessage)
      }
    }
  }

  // カメラ停止
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
      setIsStreamActive(false)
      console.log('カメラストップ')
    }
  }

  // ブラウザサポート確認
  const checkSupport = () => {
    const support = {
      getUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
      enumerateDevices: !!(navigator.mediaDevices && navigator.mediaDevices.enumerateDevices),
      isSecureContext: window.isSecureContext,
      protocol: window.location.protocol,
      host: window.location.host
    }
    console.log('ブラウザサポート状況:', support)
    return support
  }

  const support = checkSupport()

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8">
          📹 シンプルカメラテスト
        </h1>

        {/* サポート状況 */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-bold mb-4">🔍 ブラウザサポート状況</h2>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>getUserMedia: {support.getUserMedia ? '✅ 対応' : '❌ 非対応'}</div>
            <div>enumerateDevices: {support.enumerateDevices ? '✅ 対応' : '❌ 非対応'}</div>
            <div>Secure Context: {support.isSecureContext ? '✅ 安全' : '⚠️ 非安全'}</div>
            <div>Protocol: {support.protocol}</div>
            <div>Host: {support.host}</div>
            <div>利用可能カメラ: {devices.length}台</div>
          </div>
        </div>

        {/* エラー表示 */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
            <h3 className="font-bold">❌ エラー</h3>
            <p>{error}</p>
          </div>
        )}

        {/* カメラデバイス一覧 */}
        {devices.length > 0 && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-xl font-bold mb-4">📱 利用可能カメラデバイス</h2>
            {devices.map((device, index) => (
              <div key={device.deviceId} className="text-sm mb-2">
                {index + 1}. {device.label || `カメラ ${index + 1}`}
              </div>
            ))}
          </div>
        )}

        {/* カメラコントロール */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-bold mb-4">🎮 カメラコントロール</h2>
          <div className="flex space-x-4">
            {!isStreamActive ? (
              <button
                onClick={startCamera}
                className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-6 rounded"
              >
                ▶️ カメラ開始
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-6 rounded"
              >
                ⏹️ カメラ停止
              </button>
            )}
          </div>
        </div>

        {/* ビデオ表示 */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold mb-4">📺 カメラ映像</h2>
          <div className="flex justify-center">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="border border-gray-300 rounded"
              width="640"
              height="480"
            />
          </div>
          {isStreamActive && (
            <p className="text-center text-green-600 mt-4">
              ✅ カメラストリーム起動中
            </p>
          )}
        </div>
      </div>
    </div>
  )
}