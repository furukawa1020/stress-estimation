import Image from "next/image";

export default function Home() {
  return (
    'use client'

import { useState, useRef, useEffect } from 'react'
import { StressDetectionSystem } from '@/components/StressDetectionSystem'
import { FuguVisualization } from '@/components/FuguVisualization'
import { motion } from 'framer-motion'

export default function Home() {
  const [isSystemActive, setIsSystemActive] = useState(false)
  const [stressData, setStressData] = useState({
    heartRate: 0,
    stressLevel: 0,
    emotionalState: 'neutral' as 'calm' | 'neutral' | 'stressed' | 'anxious',
    confidence: 0,
    timestamp: Date.now()
  })

  const handleStressDataUpdate = (newData: typeof stressData) => {
    setStressData(newData)
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-4">
      <div className="container mx-auto max-w-6xl">
        {/* ヘッダー */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-8"
        >
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent mb-4">
            🔬 次世代ストレス推定システム
          </h1>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto">
            WebRTC + Transformer.js による学術レベルのリアルタイム生理学的ストレス検出
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* 左側: ストレス検出システム */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <StressDetectionSystem
              isActive={isSystemActive}
              onToggle={setIsSystemActive}
              onStressDataUpdate={handleStressDataUpdate}
            />
          </motion.div>

          {/* 右側: フグ可視化システム */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <FuguVisualization stressData={stressData} />
          </motion.div>
        </div>

        {/* データ表示パネル */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-8 bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-gray-200"
        >
          <h3 className="text-xl font-semibold mb-4 text-gray-800">📊 リアルタイム生理学的指標</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-red-50 rounded-lg">
              <div className="text-2xl font-bold text-red-600">{stressData.heartRate}</div>
              <div className="text-sm text-gray-600">心拍数 (BPM)</div>
            </div>
            <div className="text-center p-4 bg-orange-50 rounded-lg">
              <div className="text-2xl font-bold text-orange-600">{Math.round(stressData.stressLevel * 100)}%</div>
              <div className="text-sm text-gray-600">ストレスレベル</div>
            </div>
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{stressData.emotionalState}</div>
              <div className="text-sm text-gray-600">感情状態</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{Math.round(stressData.confidence * 100)}%</div>
              <div className="text-sm text-gray-600">信頼度</div>
            </div>
          </div>
        </motion.div>

        {/* 学術的説明 */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="mt-8 bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-gray-200"
        >
          <h3 className="text-xl font-semibold mb-4 text-gray-800">🎓 技術説明</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-gray-700">
            <div>
              <h4 className="font-semibold text-purple-600 mb-2">🫀 心拍数測定 (rPPG)</h4>
              <p>リモート光容積脈波解析によりカメラから非接触で心拍数を検出。顔面の微細な血流変化を解析。</p>
            </div>
            <div>
              <h4 className="font-semibold text-purple-600 mb-2">😊 表情分析</h4>
              <p>Transformer.jsによる深層学習ベースの表情認識。微細な表情変化からストレス状態を推定。</p>
            </div>
            <div>
              <h4 className="font-semibold text-purple-600 mb-2">👁️ 瞳孔径変化</h4>
              <p>自律神経活動の指標として瞳孔径変化をリアルタイム検出。ストレス応答の早期発見。</p>
            </div>
            <div>
              <h4 className="font-semibold text-purple-600 mb-2">🎭 マイクロエクスプレッション</h4>
              <p>無意識の微細表情変化を高周波数解析。隠れたストレス反応を検出。</p>
            </div>
          </div>
        </motion.div>
      </div>
    </main>
  )
}" "}
            <code className="bg-black/[.05] dark:bg-white/[.06] font-mono font-semibold px-1 py-0.5 rounded">
              src/app/page.tsx
            </code>
            .
          </li>
          <li className="tracking-[-.01em]">
            Save and see your changes instantly.
          </li>
        </ol>

        <div className="flex gap-4 items-center flex-col sm:flex-row">
          <a
            className="rounded-full border border-solid border-transparent transition-colors flex items-center justify-center bg-foreground text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] font-medium text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 sm:w-auto"
            href="https://vercel.com/new?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              className="dark:invert"
              src="/vercel.svg"
              alt="Vercel logomark"
              width={20}
              height={20}
            />
            Deploy now
          </a>
          <a
            className="rounded-full border border-solid border-black/[.08] dark:border-white/[.145] transition-colors flex items-center justify-center hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] hover:border-transparent font-medium text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 w-full sm:w-auto md:w-[158px]"
            href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            Read our docs
          </a>
        </div>
      </main>
      <footer className="row-start-3 flex gap-[24px] flex-wrap items-center justify-center">
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/file.svg"
            alt="File icon"
            width={16}
            height={16}
          />
          Learn
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/window.svg"
            alt="Window icon"
            width={16}
            height={16}
          />
          Examples
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://nextjs.org?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/globe.svg"
            alt="Globe icon"
            width={16}
            height={16}
          />
          Go to nextjs.org →
        </a>
      </footer>
    </div>
  );
}
