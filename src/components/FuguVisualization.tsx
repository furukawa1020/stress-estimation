'use client'

import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Eye, Heart, Brain } from '@/lib/icons'

interface StressData {
  heartRate: number
  stressLevel: number
  emotionalState: 'calm' | 'neutral' | 'stressed' | 'anxious'
  confidence: number
  timestamp: number
}

interface Props {
  stressData: StressData
}

export function FuguVisualization({ stressData }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | null>(null)
  const [fuguSize, setFuguSize] = useState(1.0)
  const [fuguColor, setFuguColor] = useState('#4FC3F7')
  const [spikes, setSpikes] = useState(0)

  // ストレスデータに基づくフグの状態計算
  useEffect(() => {
    const { stressLevel, emotionalState, heartRate } = stressData

    // サイズ変化（ストレスレベルに応じて膨らむ）
    const newSize = 1.0 + (stressLevel * 1.5) // 最大2.5倍まで
    setFuguSize(newSize)

    // 色変化（感情状態に応じて）
    const colors = {
      calm: '#4FC3F7',      // 青（冷静）
      neutral: '#66BB6A',   // 緑（中立）
      stressed: '#FFA726',  // オレンジ（ストレス）
      anxious: '#EF5350'    // 赤（不安）
    }
    setFuguColor(colors[emotionalState])

    // トゲの数（心拍数の変動に応じて）
    const heartRateStress = Math.abs(heartRate - 75) / 75
    setSpikes(Math.floor(heartRateStress * 20))

  }, [stressData])

  // Three.jsアニメーション
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let time = 0

    const animate = () => {
      time += 0.02

      // キャンバスクリア
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const centerX = canvas.width / 2
      const centerY = canvas.height / 2
      const baseRadius = 80

      // フグの本体（呼吸するようなアニメーション）
      const breathingEffect = Math.sin(time * 2) * 0.1 + 1
      const currentRadius = baseRadius * fuguSize * breathingEffect

      // グラデーション作成
      const gradient = ctx.createRadialGradient(
        centerX, centerY, 0,
        centerX, centerY, currentRadius
      )
      gradient.addColorStop(0, fuguColor)
      gradient.addColorStop(0.7, fuguColor + '80')
      gradient.addColorStop(1, fuguColor + '20')

      // フグの体
      ctx.fillStyle = gradient
      ctx.beginPath()
      ctx.arc(centerX, centerY, currentRadius, 0, Math.PI * 2)
      ctx.fill()

      // ストレストゲ（ストレスレベルに応じて）
      if (spikes > 0) {
        ctx.strokeStyle = fuguColor
        ctx.lineWidth = 3
        
        for (let i = 0; i < spikes; i++) {
          const angle = (Math.PI * 2 * i) / spikes + time
          const spikeLength = 20 + Math.sin(time * 3 + i) * 5
          
          const startX = centerX + Math.cos(angle) * currentRadius
          const startY = centerY + Math.sin(angle) * currentRadius
          const endX = centerX + Math.cos(angle) * (currentRadius + spikeLength)
          const endY = centerY + Math.sin(angle) * (currentRadius + spikeLength)
          
          ctx.beginPath()
          ctx.moveTo(startX, startY)
          ctx.lineTo(endX, endY)
          ctx.stroke()
        }
      }

      // 目（感情に応じて変化）
      const eyeSize = 8 + stressData.stressLevel * 5
      const eyeY = centerY - 20
      
      ctx.fillStyle = '#FFFFFF'
      // 左目
      ctx.beginPath()
      ctx.arc(centerX - 25, eyeY, eyeSize, 0, Math.PI * 2)
      ctx.fill()
      // 右目
      ctx.beginPath()
      ctx.arc(centerX + 25, eyeY, eyeSize, 0, Math.PI * 2)
      ctx.fill()

      // 瞳孔
      ctx.fillStyle = '#000000'
      const pupilOffset = Math.sin(time) * 2
      // 左瞳孔
      ctx.beginPath()
      ctx.arc(centerX - 25 + pupilOffset, eyeY, eyeSize * 0.4, 0, Math.PI * 2)
      ctx.fill()
      // 右瞳孔
      ctx.beginPath()
      ctx.arc(centerX + 25 + pupilOffset, eyeY, eyeSize * 0.4, 0, Math.PI * 2)
      ctx.fill()

      // 口（感情表現）
      ctx.strokeStyle = '#000000'
      ctx.lineWidth = 4
      ctx.lineCap = 'round'
      
      const mouthY = centerY + 15
      const mouthCurve = stressData.emotionalState === 'calm' ? -10 :
                        stressData.emotionalState === 'stressed' ? 10 :
                        stressData.emotionalState === 'anxious' ? 15 : 0

      ctx.beginPath()
      ctx.moveTo(centerX - 15, mouthY)
      ctx.quadraticCurveTo(centerX, mouthY + mouthCurve, centerX + 15, mouthY)
      ctx.stroke()

      // 頬の赤み（ストレス時）
      if (stressData.stressLevel > 0.5) {
        const blushAlpha = stressData.stressLevel * 0.3
        ctx.fillStyle = `rgba(255, 100, 100, ${blushAlpha})`
        
        // 左頬
        ctx.beginPath()
        ctx.arc(centerX - 40, centerY + 10, 15, 0, Math.PI * 2)
        ctx.fill()
        // 右頬
        ctx.beginPath()
        ctx.arc(centerX + 40, centerY + 10, 15, 0, Math.PI * 2)
        ctx.fill()
      }

      // パーティクル効果（心拍に合わせて）
      if (stressData.heartRate > 80) {
        const particleCount = Math.floor((stressData.heartRate - 80) / 10)
        ctx.fillStyle = fuguColor + '60'
        
        for (let i = 0; i < particleCount; i++) {
          const angle = (Math.PI * 2 * i) / particleCount + time * 2
          const distance = currentRadius + 30 + Math.sin(time * 5 + i) * 10
          const particleX = centerX + Math.cos(angle) * distance
          const particleY = centerY + Math.sin(angle) * distance
          const particleSize = 3 + Math.sin(time * 3 + i) * 2
          
          ctx.beginPath()
          ctx.arc(particleX, particleY, particleSize, 0, Math.PI * 2)
          ctx.fill()
        }
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [fuguSize, fuguColor, spikes, stressData])

  // ストレス状態メッセージ
  const getStressMessage = () => {
    const { stressLevel, emotionalState } = stressData
    
    if (stressLevel < 0.3) {
      return { 
        text: "とてもリラックスしています 😌", 
        color: "text-green-600",
        bgColor: "bg-green-50"
      }
    } else if (stressLevel < 0.6) {
      return { 
        text: "少し緊張していますね 😐", 
        color: "text-yellow-600",
        bgColor: "bg-yellow-50"
      }
    } else if (stressLevel < 0.8) {
      return { 
        text: "ストレスを感じています 😰", 
        color: "text-orange-600",
        bgColor: "bg-orange-50"
      }
    } else {
      return { 
        text: "高いストレス状態です！ 😨", 
        color: "text-red-600",
        bgColor: "bg-red-50"
      }
    }
  }

  const stressMessage = getStressMessage()

  return (
    <div className="bg-white/90 backdrop-blur-sm rounded-xl p-6 border border-gray-200 shadow-lg">
      {/* ヘッダー */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Brain className="w-6 h-6 text-blue-600" />
          <h2 className="text-xl font-semibold text-gray-800">ストレス可視化</h2>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <div className={`w-3 h-3 rounded-full ${
            stressData.confidence > 0.8 ? 'bg-green-500' :
            stressData.confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
          }`} />
          <span>信頼度: {Math.round(stressData.confidence * 100)}%</span>
        </div>
      </div>

      {/* フグ可視化エリア */}
      <div className="relative mb-6">
        <motion.div 
          className="relative bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-8 border border-blue-100"
          animate={{ 
            backgroundColor: stressData.stressLevel > 0.7 ? '#FEF2F2' : '#F0F9FF' 
          }}
          transition={{ duration: 1 }}
        >
          <canvas
            ref={canvasRef}
            width={300}
            height={300}
            className="w-full h-auto max-w-sm mx-auto"
          />
          
          {/* オーバーレイ情報 */}
          <div className="absolute top-4 left-4 bg-white/80 backdrop-blur-sm rounded-lg p-2 text-xs">
            <div className="flex items-center gap-1 mb-1">
              <Heart className="w-3 h-3 text-red-500" />
              <span>{stressData.heartRate} BPM</span>
            </div>
            <div className="flex items-center gap-1">
              <Eye className="w-3 h-3 text-blue-500" />
              <span>{stressData.emotionalState}</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* ストレス状態メッセージ */}
      <motion.div
        className={`p-4 rounded-lg border ${stressMessage.bgColor} ${stressMessage.color}`}
        key={stressMessage.text}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="flex items-center justify-between">
          <span className="font-medium">{stressMessage.text}</span>
          <span className="text-sm opacity-75">
            {Math.round(stressData.stressLevel * 100)}%
          </span>
        </div>
      </motion.div>

      {/* フグの説明 */}
      <div className="mt-6 p-4 bg-purple-50 border border-purple-200 rounded-lg">
        <h4 className="font-semibold text-purple-800 mb-2">🐡 フグちゃんの見方</h4>
        <div className="text-purple-700 text-sm space-y-1">
          <div>• <strong>大きさ</strong>: ストレスレベルに応じて膨らみます</div>
          <div>• <strong>色</strong>: 感情状態を表現（青=冷静、緑=普通、オレンジ=ストレス、赤=不安）</div>
          <div>• <strong>トゲ</strong>: 心拍数の変動度を表示</div>
          <div>• <strong>表情</strong>: リアルタイムの感情状態を反映</div>
          <div>• <strong>パーティクル</strong>: 高心拍数時に出現</div>
        </div>
      </div>

      {/* 学術的解説 */}
      <div className="mt-4 p-4 bg-gray-50 border border-gray-200 rounded-lg">
        <h4 className="font-semibold text-gray-800 mb-2">📚 認知負荷軽減設計</h4>
        <p className="text-gray-700 text-sm">
          複雑な生理学的データを直感的に理解できるよう、
          かわいいフグのメタファーを使用。視覚的変化により
          ユーザーの認知負荷を最小限に抑えながら、
          正確なストレス情報を伝達します。
        </p>
      </div>
    </div>
  )
}