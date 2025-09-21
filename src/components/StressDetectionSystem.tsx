'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Camera, CameraOff, Play, Pause, Activity } from 'lucide-react'
import { WebRTCManager } from '@/lib/webrtc-manager'
import { StressAnalyzer } from '@/lib/stress-analyzer'
import { DataCollector } from '@/lib/data-collector'

interface StressData {
  heartRate: number
  stressLevel: number
  emotionalState: 'calm' | 'neutral' | 'stressed' | 'anxious'
  confidence: number
  timestamp: number
  // 学術的指標
  rppgSignal: number[]
  facialLandmarks: number[][]
  pupilDiameter: number
  microExpressions: string[]
  headPose: { yaw: number; pitch: number; roll: number }
  autonomicNervousSystem: {
    sympathetic: number
    parasympathetic: number
    balance: number
  }
}

interface Props {
  isActive: boolean
  onToggle: (active: boolean) => void
  onStressDataUpdate: (data: StressData) => void
}

export function StressDetectionSystem({ isActive, onToggle, onStressDataUpdate }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fps, setFps] = useState(0)
  const [systemStatus, setSystemStatus] = useState<'idle' | 'initializing' | 'active' | 'error'>('idle')
  
  // 学術研究用データ収集
  const [sessionData, setSessionData] = useState<{
    startTime: number
    samples: StressData[]
    metadata: {
      userId: string
      environment: string
      calibrationData: any
    }
  } | null>(null)

  const webrtcManager = useRef<WebRTCManager | null>(null)
  const stressAnalyzer = useRef<StressAnalyzer | null>(null)
  const dataCollector = useRef<DataCollector | null>(null)
  const animationFrameId = useRef<number | null>(null)

  // システム初期化
  const initializeSystem = useCallback(async () => {
    try {
      setIsLoading(true)
      setSystemStatus('initializing')
      setError(null)

      // WebRTCマネージャー初期化
      webrtcManager.current = new WebRTCManager()
      await webrtcManager.current.initialize()

      // ストレス分析エンジン初期化
      stressAnalyzer.current = new StressAnalyzer()
      await stressAnalyzer.current.loadModels()

      // データ収集システム初期化
      dataCollector.current = new DataCollector()
      
      // カメラストリーム開始
      const stream = await webrtcManager.current.startCamera()
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      // セッション開始
      const newSession = {
        startTime: Date.now(),
        samples: [],
        metadata: {
          userId: `user_${Date.now()}`,
          environment: navigator.userAgent,
          calibrationData: await stressAnalyzer.current.calibrate(videoRef.current!)
        }
      }
      setSessionData(newSession)

      setSystemStatus('active')
      startAnalysisLoop()

    } catch (err) {
      console.error('System initialization failed:', err)
      setError(err instanceof Error ? err.message : 'システム初期化に失敗しました')
      setSystemStatus('error')
    } finally {
      setIsLoading(false)
    }
  }, [])

  // 分析ループ
  const startAnalysisLoop = useCallback(() => {
    let lastTime = performance.now()
    let frameCount = 0

    const analyzeFrame = async (timestamp: number) => {
      if (!isActive || !videoRef.current || !canvasRef.current || !stressAnalyzer.current) {
        return
      }

      try {
        // FPS計算
        frameCount++
        if (timestamp - lastTime >= 1000) {
          setFps(Math.round(frameCount * 1000 / (timestamp - lastTime)))
          frameCount = 0
          lastTime = timestamp
        }

        // フレーム解析
        const analysisResult = await stressAnalyzer.current.analyzeFrame(
          videoRef.current,
          canvasRef.current
        )

        if (analysisResult) {
          const stressData: StressData = {
            heartRate: analysisResult.heartRate,
            stressLevel: analysisResult.stressLevel,
            emotionalState: analysisResult.emotionalState,
            confidence: analysisResult.confidence,
            timestamp: Date.now(),
            rppgSignal: analysisResult.rppgSignal,
            facialLandmarks: analysisResult.facialLandmarks,
            pupilDiameter: analysisResult.pupilDiameter,
            microExpressions: analysisResult.microExpressions,
            headPose: analysisResult.headPose,
            autonomicNervousSystem: analysisResult.autonomicNervousSystem
          }

          // データ更新
          onStressDataUpdate(stressData)

          // 学術データ収集
          if (sessionData && dataCollector.current) {
            dataCollector.current.addSample(stressData)
            setSessionData(prev => prev ? {
              ...prev,
              samples: [...prev.samples, stressData]
            } : null)
          }
        }

      } catch (err) {
        console.error('Frame analysis error:', err)
      }

      if (isActive) {
        animationFrameId.current = requestAnimationFrame(analyzeFrame)
      }
    }

    animationFrameId.current = requestAnimationFrame(analyzeFrame)
  }, [isActive, onStressDataUpdate, sessionData])

  // システム停止
  const stopSystem = useCallback(async () => {
    try {
      setSystemStatus('idle')
      
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current)
        animationFrameId.current = null
      }

      if (webrtcManager.current) {
        await webrtcManager.current.stopCamera()
      }

      // 学術データのエクスポート
      if (sessionData && dataCollector.current) {
        await dataCollector.current.exportSession(sessionData)
      }

      setSessionData(null)
    } catch (err) {
      console.error('System stop error:', err)
      setError('システム停止中にエラーが発生しました')
    }
  }, [sessionData])

  // アクティブ状態変更ハンドラー
  useEffect(() => {
    if (isActive) {
      initializeSystem()
    } else {
      stopSystem()
    }

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current)
      }
    }
  }, [isActive, initializeSystem, stopSystem])

  return (
    <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 border border-gray-200 shadow-lg">
      {/* ヘッダー */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Activity className="w-6 h-6 text-purple-600" />
          <h2 className="text-xl font-semibold text-gray-800">ストレス検出システム</h2>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-3 h-3 rounded-full ${
            systemStatus === 'active' ? 'bg-green-500' :
            systemStatus === 'initializing' ? 'bg-yellow-500' :
            systemStatus === 'error' ? 'bg-red-500' : 'bg-gray-300'
          }`} />
          <span className="text-sm text-gray-600">{fps} FPS</span>
        </div>
      </div>

      {/* 動画表示エリア */}
      <div className="relative mb-6">
        <motion.div
          className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video"
          whileHover={{ scale: 1.02 }}
          transition={{ duration: 0.2 }}
        >
          <video
            ref={videoRef}
            className="w-full h-full object-cover"
            playsInline
            muted
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full"
            style={{ mixBlendMode: 'overlay' }}
          />
          
          {/* オーバーレイ情報 */}
          <div className="absolute top-4 left-4 bg-black/70 text-white px-3 py-1 rounded text-sm">
            {systemStatus === 'active' ? '分析中...' :
             systemStatus === 'initializing' ? '初期化中...' :
             systemStatus === 'error' ? 'エラー' : '待機中'}
          </div>

          {sessionData && (
            <div className="absolute top-4 right-4 bg-black/70 text-white px-3 py-1 rounded text-sm">
              サンプル数: {sessionData.samples.length}
            </div>
          )}
        </motion.div>
      </div>

      {/* コントロールパネル */}
      <div className="flex items-center justify-between">
        <motion.button
          onClick={() => onToggle(!isActive)}
          disabled={isLoading}
          className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors ${
            isActive
              ? 'bg-red-600 hover:bg-red-700 text-white'
              : 'bg-purple-600 hover:bg-purple-700 text-white'
          } disabled:opacity-50 disabled:cursor-not-allowed`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {isLoading ? (
            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : isActive ? (
            <>
              <Pause className="w-5 h-5" />
              停止
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              開始
            </>
          )}
        </motion.button>

        <div className="flex items-center gap-4 text-sm text-gray-600">
          <div className="flex items-center gap-2">
            <Camera className="w-4 h-4" />
            <span>WebRTC</span>
          </div>
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4" />
            <span>AI分析</span>
          </div>
        </div>
      </div>

      {/* エラー表示 */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg"
        >
          <p className="text-red-600 text-sm">{error}</p>
        </motion.div>
      )}

      {/* 学術的注記 */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h4 className="font-semibold text-blue-800 mb-2">📊 データ収集について</h4>
        <p className="text-blue-700 text-sm">
          このシステムは学術研究レベルの精度でストレス指標を測定し、
          国際学会発表可能な形式でデータを収集・保存します。
          すべてのデータはブラウザ内で処理され、プライバシーが保護されます。
        </p>
      </div>
    </div>
  )
}