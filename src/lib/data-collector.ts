/**
 * DataCollector - 学術研究用データ収集・分析システム
 * 国際学会発表可能な形式でデータを収集・保存・エクスポート
 */

interface StressData {
  heartRate: number
  stressLevel: number
  emotionalState: 'calm' | 'neutral' | 'stressed' | 'anxious'
  confidence: number
  timestamp: number
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

interface SessionData {
  startTime: number
  samples: StressData[]
  metadata: {
    userId: string
    environment: string
    calibrationData: any
  }
}

interface ExportData {
  session: SessionData
  analysis: {
    summary: {
      duration: number
      sampleCount: number
      averageStressLevel: number
      stressEvents: number
      heartRateVariability: number
    }
    statistics: {
      heartRate: { min: number; max: number; mean: number; std: number }
      stressLevel: { min: number; max: number; mean: number; std: number }
      emotionalDistribution: Record<string, number>
      autonomicBalance: { mean: number; std: number }
    }
    scientificMetrics: {
      rmssd: number // Root Mean Square of Successive Differences
      pnn50: number // Percentage of NN50
      stressIndex: number // 独自ストレス指標
      recoveryTime: number // ストレス回復時間
    }
  }
  citations: string[] // 使用した学術手法の引用
}

export class DataCollector {
  private samples: StressData[] = []
  private sessionStart: number = 0
  private isCollecting = false

  constructor() {
    this.sessionStart = Date.now()
  }

  /**
   * データサンプル追加
   */
  addSample(data: StressData): void {
    if (!this.isCollecting) {
      this.isCollecting = true
      this.sessionStart = data.timestamp
    }

    this.samples.push(data)
    
    // リアルタイム異常検知
    this.detectAnomalies(data)
  }

  /**
   * セッションデータエクスポート
   */
  async exportSession(sessionData: SessionData): Promise<ExportData> {
    const analysis = this.performStatisticalAnalysis(sessionData.samples)
    
    const exportData: ExportData = {
      session: sessionData,
      analysis,
      citations: this.getAcademicCitations()
    }

    // 複数形式でエクスポート
    await this.exportToJSON(exportData)
    await this.exportToCSV(sessionData.samples)
    await this.exportToPDF(exportData)
    
    return exportData
  }

  /**
   * 統計分析実行
   */
  private performStatisticalAnalysis(samples: StressData[]): ExportData['analysis'] {
    if (samples.length === 0) {
      return this.getEmptyAnalysis()
    }

    // 基本統計
    const heartRates = samples.map(s => s.heartRate).filter(hr => hr > 0)
    const stressLevels = samples.map(s => s.stressLevel)
    const duration = samples[samples.length - 1].timestamp - samples[0].timestamp

    // HRV分析
    const rmssd = this.calculateRMSSD(heartRates)
    const pnn50 = this.calculatePNN50(heartRates)
    
    // ストレス指標
    const stressIndex = this.calculateStressIndex(samples)
    const recoveryTime = this.calculateRecoveryTime(samples)

    // 感情分布
    const emotionalDistribution = this.calculateEmotionalDistribution(samples)

    return {
      summary: {
        duration: duration,
        sampleCount: samples.length,
        averageStressLevel: this.mean(stressLevels),
        stressEvents: this.countStressEvents(samples),
        heartRateVariability: rmssd
      },
      statistics: {
        heartRate: this.calculateStats(heartRates),
        stressLevel: this.calculateStats(stressLevels),
        emotionalDistribution,
        autonomicBalance: {
          mean: this.mean(samples.map(s => s.autonomicNervousSystem.balance)),
          std: this.std(samples.map(s => s.autonomicNervousSystem.balance))
        }
      },
      scientificMetrics: {
        rmssd,
        pnn50,
        stressIndex,
        recoveryTime
      }
    }
  }

  /**
   * RMSSD計算（HRV指標）
   */
  private calculateRMSSD(heartRates: number[]): number {
    if (heartRates.length < 2) return 0

    const differences = []
    for (let i = 1; i < heartRates.length; i++) {
      differences.push(Math.pow(heartRates[i] - heartRates[i-1], 2))
    }

    const meanSquaredDifference = this.mean(differences)
    return Math.sqrt(meanSquaredDifference)
  }

  /**
   * pNN50計算（HRV指標）
   */
  private calculatePNN50(heartRates: number[]): number {
    if (heartRates.length < 2) return 0

    let nn50Count = 0
    for (let i = 1; i < heartRates.length; i++) {
      if (Math.abs(heartRates[i] - heartRates[i-1]) > 50) {
        nn50Count++
      }
    }

    return (nn50Count / (heartRates.length - 1)) * 100
  }

  /**
   * 独自ストレス指標計算
   */
  private calculateStressIndex(samples: StressData[]): number {
    // 複数の生理学的指標を統合したストレス指標
    let totalStress = 0
    
    for (const sample of samples) {
      // 心拍数の正常範囲からの逸脱度
      const hrStress = Math.abs(sample.heartRate - 75) / 75 * 0.3
      
      // 表情からのストレス度
      const emotionStress = this.getEmotionStressWeight(sample.emotionalState) * 0.3
      
      // 自律神経バランス
      const ansStress = Math.abs(sample.autonomicNervousSystem.balance) * 0.4
      
      totalStress += hrStress + emotionStress + ansStress
    }
    
    return totalStress / samples.length
  }

  /**
   * ストレス回復時間計算
   */
  private calculateRecoveryTime(samples: StressData[]): number {
    // ストレスピークから正常値への回復時間を計算
    let recoveryTimes: number[] = []
    let stressStart = -1
    
    for (let i = 0; i < samples.length; i++) {
      if (samples[i].stressLevel > 0.7 && stressStart === -1) {
        stressStart = i
      } else if (samples[i].stressLevel < 0.3 && stressStart !== -1) {
        recoveryTimes.push(i - stressStart)
        stressStart = -1
      }
    }
    
    return recoveryTimes.length > 0 ? this.mean(recoveryTimes) * 33 : 0 // ms変換
  }

  /**
   * 感情分布計算
   */
  private calculateEmotionalDistribution(samples: StressData[]): Record<string, number> {
    const distribution: Record<string, number> = {
      calm: 0,
      neutral: 0,
      stressed: 0,
      anxious: 0
    }
    
    for (const sample of samples) {
      distribution[sample.emotionalState]++
    }
    
    // パーセンテージに変換
    const total = samples.length
    for (const emotion in distribution) {
      distribution[emotion] = (distribution[emotion] / total) * 100
    }
    
    return distribution
  }

  /**
   * ストレスイベント数カウント
   */
  private countStressEvents(samples: StressData[]): number {
    let events = 0
    let inStressEvent = false
    
    for (const sample of samples) {
      if (sample.stressLevel > 0.6 && !inStressEvent) {
        events++
        inStressEvent = true
      } else if (sample.stressLevel < 0.4) {
        inStressEvent = false
      }
    }
    
    return events
  }

  /**
   * 異常検知
   */
  private detectAnomalies(data: StressData): void {
    const anomalies = []
    
    // 心拍数異常
    if (data.heartRate > 120 || data.heartRate < 50) {
      anomalies.push(`Abnormal heart rate: ${data.heartRate} BPM`)
    }
    
    // 極端なストレスレベル
    if (data.stressLevel > 0.9) {
      anomalies.push(`High stress level detected: ${(data.stressLevel * 100).toFixed(1)}%`)
    }
    
    if (anomalies.length > 0) {
      console.warn('Anomalies detected:', anomalies)
    }
  }

  /**
   * JSON形式エクスポート
   */
  private async exportToJSON(data: ExportData): Promise<void> {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `stress_analysis_${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  /**
   * CSV形式エクスポート
   */
  private async exportToCSV(samples: StressData[]): Promise<void> {
    const headers = [
      'timestamp', 'heartRate', 'stressLevel', 'emotionalState', 'confidence',
      'pupilDiameter', 'headPose_yaw', 'headPose_pitch', 'headPose_roll',
      'ans_sympathetic', 'ans_parasympathetic', 'ans_balance'
    ]
    
    const csvContent = [
      headers.join(','),
      ...samples.map(sample => [
        sample.timestamp,
        sample.heartRate,
        sample.stressLevel,
        sample.emotionalState,
        sample.confidence,
        sample.pupilDiameter,
        sample.headPose.yaw,
        sample.headPose.pitch,
        sample.headPose.roll,
        sample.autonomicNervousSystem.sympathetic,
        sample.autonomicNervousSystem.parasympathetic,
        sample.autonomicNervousSystem.balance
      ].join(','))
    ].join('\n')
    
    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `stress_data_${new Date().toISOString().split('T')[0]}.csv`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  /**
   * PDF形式エクスポート（学術レポート）
   */
  private async exportToPDF(data: ExportData): Promise<void> {
    // 簡易的なPDFレポート生成
    console.log('PDF export feature would be implemented here')
  }

  /**
   * 学術引用リスト
   */
  private getAcademicCitations(): string[] {
    return [
      "Li, X., Chen, J., Zhao, G., & Pietikainen, M. (2014). Remote heart rate measurement from face videos under realistic situations. CVPR.",
      "Gudi, A., Bittner, M., Lochmans, R., & van Gemert, J. (2019). Efficient real-time camera based estimation of heart rate and its variability. ICCVW.",
      "Hassan, M. A., Malik, A. S., Fofi, D., Karasfi, B., & Meriaudeau, F. (2020). Towards health monitoring using remote heart rate measurement using digital camera. Measurement.",
      "Monkaresi, H., Bosch, N., Calvo, R. A., & D'Mello, S. K. (2016). Automated detection of engagement using video-based estimation of facial expressions and heart rate. IEEE Transactions on Affective Computing."
    ]
  }

  // ユーティリティ関数
  private mean(values: number[]): number {
    return values.length > 0 ? values.reduce((sum, val) => sum + val, 0) / values.length : 0
  }

  private std(values: number[]): number {
    const mean = this.mean(values)
    const variance = this.mean(values.map(val => Math.pow(val - mean, 2)))
    return Math.sqrt(variance)
  }

  private calculateStats(values: number[]): { min: number; max: number; mean: number; std: number } {
    return {
      min: Math.min(...values),
      max: Math.max(...values),
      mean: this.mean(values),
      std: this.std(values)
    }
  }

  private getEmotionStressWeight(emotion: string): number {
    const weights = { calm: 0, neutral: 0.2, stressed: 0.8, anxious: 1.0 }
    return weights[emotion as keyof typeof weights] || 0.5
  }

  private getEmptyAnalysis(): ExportData['analysis'] {
    return {
      summary: { duration: 0, sampleCount: 0, averageStressLevel: 0, stressEvents: 0, heartRateVariability: 0 },
      statistics: {
        heartRate: { min: 0, max: 0, mean: 0, std: 0 },
        stressLevel: { min: 0, max: 0, mean: 0, std: 0 },
        emotionalDistribution: {},
        autonomicBalance: { mean: 0, std: 0 }
      },
      scientificMetrics: { rmssd: 0, pnn50: 0, stressIndex: 0, recoveryTime: 0 }
    }
  }
}