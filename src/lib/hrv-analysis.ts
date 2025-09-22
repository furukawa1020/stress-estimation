/**
 * 34パラメータHRV解析システム
 * 学術研究標準の心拍変動解析（2024-2025年基準）
 * 参考文献：Task Force of ESC/NASPE, Physiological Measurement, IEEE TBME
 */

// HRV解析結果型定義
export interface HRVAnalysisResult {
  // 時間領域パラメータ (14個)
  timeDomain: {
    meanNN: number          // 平均NN間隔 (ms)
    sdNN: number           // NN間隔標準偏差 (ms)
    rmssd: number          // 連続NN間隔差の二乗平均平方根 (ms)
    nn50: number           // 50ms以上差のNN間隔ペア数
    pnn50: number          // NN50の全NN間隔に対する割合 (%)
    nn20: number           // 20ms以上差のNN間隔ペア数
    pnn20: number          // NN20の全NN間隔に対する割合 (%)
    cvNN: number           // NN間隔変動係数 (%)
    tinn: number           // 三角補間NN間隔 (ms)
    triangularIndex: number // 三角指数
    meanHR: number         // 平均心拍数 (bpm)
    maxHR: number          // 最大心拍数 (bpm)
    minHR: number          // 最小心拍数 (bpm)
    hrvIndex: number       // HRV指数
  }
  
  // 周波数領域パラメータ (8個)
  frequencyDomain: {
    totalPower: number     // 総パワー (ms²)
    vlf: number           // 超低周波成分 (0.003-0.04Hz) (ms²)
    lf: number            // 低周波成分 (0.04-0.15Hz) (ms²)
    hf: number            // 高周波成分 (0.15-0.4Hz) (ms²)
    lfhfRatio: number     // LF/HF比
    lfNormalized: number  // 正規化LF (nu)
    hfNormalized: number  // 正規化HF (nu)
    peakFrequency: number // ピーク周波数 (Hz)
  }
  
  // 非線形パラメータ (12個)
  nonlinear: {
    // Poincaréプロット解析
    sd1: number           // 短期変動指標 (ms)
    sd2: number           // 長期変動指標 (ms)
    sd1sd2Ratio: number   // SD1/SD2比
    csi: number           // 心臓交感神経指数
    cvi: number           // 心臓迷走神経指数
    
    // エントロピー解析
    sampleEntropy: number // サンプルエントロピー
    approximateEntropy: number // 近似エントロピー
    
    // DFA解析
    dfa1: number          // 短期DFAスケーリング指数
    dfa2: number          // 長期DFAスケーリング指数
    
    // その他非線形指標
    complexity: number    // 複雑性指標
    fractalDimension: number // フラクタル次元
    recurrenceRate: number   // 再帰率
  }
  
  // 品質指標
  qualityMetrics: {
    dataQuality: number   // データ品質 (0-1)
    artifactRatio: number // アーチファクト比率 (%)
    stationarity: number  // 定常性指標 (0-1)
    reliability: number   // 信頼性スコア (0-1)
  }
}

/**
 * 学術レベルHRV解析クラス
 * 34パラメータの完全実装
 */
export class HRVAnalyzer {
  private readonly MIN_NN_INTERVALS = 50 // 最小解析区間数
  private readonly ARTIFACT_THRESHOLD = 0.2 // アーチファクト検出閾値
  
  /**
   * 完全HRV解析実行
   */
  async analyzeHRV(nnIntervals: number[]): Promise<HRVAnalysisResult | null> {
    try {
      // 1. データ品質チェック
      if (nnIntervals.length < this.MIN_NN_INTERVALS) {
        console.warn('Insufficient NN intervals for reliable HRV analysis')
        return null
      }
      
      // 2. アーチファクト除去
      const cleanedIntervals = this.removeArtifacts(nnIntervals)
      
      // 3. 品質評価
      const qualityMetrics = this.assessDataQuality(cleanedIntervals, nnIntervals)
      
      if (qualityMetrics.dataQuality < 0.3) {
        console.warn('Data quality too low for reliable HRV analysis')
        return null
      }
      
      // 4. 各領域解析実行
      const timeDomain = await this.analyzeTimeDomain(cleanedIntervals)
      const frequencyDomain = await this.analyzeFrequencyDomain(cleanedIntervals)
      const nonlinear = await this.analyzeNonlinear(cleanedIntervals)
      
      return {
        timeDomain,
        frequencyDomain,
        nonlinear,
        qualityMetrics
      }
    } catch (error) {
      console.error('HRV analysis error:', error)
      return null
    }
  }

  // ============ 時間領域解析 (14パラメータ) ============

  /**
   * 時間領域HRVパラメータ解析
   */
  private async analyzeTimeDomain(nnIntervals: number[]): Promise<any> {
    const n = nnIntervals.length
    
    // 基本統計量
    const meanNN = this.calculateMean(nnIntervals)
    const sdNN = this.calculateStandardDeviation(nnIntervals, meanNN)
    
    // 連続差分解析
    const differences = this.calculateSuccessiveDifferences(nnIntervals)
    const rmssd = this.calculateRMSSD(differences)
    
    // pNN50/pNN20計算
    const nn50 = differences.filter(diff => Math.abs(diff) > 50).length
    const pnn50 = (nn50 / (n - 1)) * 100
    const nn20 = differences.filter(diff => Math.abs(diff) > 20).length
    const pnn20 = (nn20 / (n - 1)) * 100
    
    // 変動係数
    const cvNN = (sdNN / meanNN) * 100
    
    // 三角指標
    const { tinn, triangularIndex } = this.calculateTriangularIndices(nnIntervals)
    
    // 心拍数指標
    const heartRates = nnIntervals.map(nn => 60000 / nn)
    const meanHR = this.calculateMean(heartRates)
    const maxHR = Math.max(...heartRates)
    const minHR = Math.min(...heartRates)
    
    // HRV指数
    const hrvIndex = this.calculateHRVIndex(nnIntervals)
    
    return {
      meanNN,
      sdNN,
      rmssd,
      nn50,
      pnn50,
      nn20,
      pnn20,
      cvNN,
      tinn,
      triangularIndex,
      meanHR,
      maxHR,
      minHR,
      hrvIndex
    }
  }

  /**
   * 連続NN間隔差分計算
   */
  private calculateSuccessiveDifferences(nnIntervals: number[]): number[] {
    const differences: number[] = []
    for (let i = 1; i < nnIntervals.length; i++) {
      differences.push(nnIntervals[i] - nnIntervals[i - 1])
    }
    return differences
  }

  /**
   * RMSSD計算（連続差分の二乗平均平方根）
   */
  private calculateRMSSD(differences: number[]): number {
    const squaredDiffs = differences.map(diff => diff * diff)
    const meanSquaredDiff = this.calculateMean(squaredDiffs)
    return Math.sqrt(meanSquaredDiff)
  }

  /**
   * 三角指標計算
   */
  private calculateTriangularIndices(nnIntervals: number[]): { tinn: number; triangularIndex: number } {
    // ヒストグラム作成（8msビン幅）
    const binWidth = 8 // ms
    const min = Math.min(...nnIntervals)
    const max = Math.max(...nnIntervals)
    const numBins = Math.ceil((max - min) / binWidth)
    
    const histogram = new Array(numBins).fill(0)
    
    for (const interval of nnIntervals) {
      const binIndex = Math.floor((interval - min) / binWidth)
      if (binIndex >= 0 && binIndex < numBins) {
        histogram[binIndex]++
      }
    }
    
    // 三角補間
    const maxCount = Math.max(...histogram)
    const modalBin = histogram.indexOf(maxCount)
    
    // TINN計算（三角補間NN間隔）
    const tinn = this.calculateTINN(histogram, modalBin, binWidth)
    
    // 三角指数計算
    const triangularIndex = nnIntervals.length / maxCount
    
    return { tinn, triangularIndex }
  }

  /**
   * TINN計算
   */
  private calculateTINN(histogram: number[], modalBin: number, binWidth: number): number {
    // 三角形の底辺長さを計算
    let leftBase = modalBin
    let rightBase = modalBin
    
    // 左端探索
    for (let i = modalBin - 1; i >= 0; i--) {
      if (histogram[i] === 0) {
        leftBase = i + 1
        break
      }
      leftBase = i
    }
    
    // 右端探索
    for (let i = modalBin + 1; i < histogram.length; i++) {
      if (histogram[i] === 0) {
        rightBase = i - 1
        break
      }
      rightBase = i
    }
    
    return (rightBase - leftBase) * binWidth
  }

  /**
   * HRV指数計算
   */
  private calculateHRVIndex(nnIntervals: number[]): number {
    const histogram = this.createNNHistogram(nnIntervals, 8)
    const totalCount = nnIntervals.length
    const modalCount = Math.max(...histogram)
    
    return totalCount / modalCount
  }

  // ============ 周波数領域解析 (8パラメータ) ============

  /**
   * 周波数領域HRVパラメータ解析
   */
  private async analyzeFrequencyDomain(nnIntervals: number[]): Promise<any> {
    // RR間隔時系列を等間隔リサンプリング
    const samplingRate = 4 // 4Hz
    const resampledSeries = this.resampleNNIntervals(nnIntervals, samplingRate)
    
    // パワースペクトル密度計算
    const psd = this.calculatePSD(resampledSeries, samplingRate)
    
    // 周波数帯域別パワー計算
    const vlf = this.calculateBandPower(psd, 0.003, 0.04, samplingRate) // 0.003-0.04Hz
    const lf = this.calculateBandPower(psd, 0.04, 0.15, samplingRate)   // 0.04-0.15Hz
    const hf = this.calculateBandPower(psd, 0.15, 0.4, samplingRate)    // 0.15-0.4Hz
    const totalPower = vlf + lf + hf
    
    // 正規化パラメータ
    const lfNormalized = (lf / (lf + hf)) * 100
    const hfNormalized = (hf / (lf + hf)) * 100
    const lfhfRatio = lf / (hf + 1e-10) // ゼロ除算回避
    
    // ピーク周波数
    const peakFrequency = this.findPeakFrequency(psd, 0.04, 0.4, samplingRate)
    
    return {
      totalPower,
      vlf,
      lf,
      hf,
      lfhfRatio,
      lfNormalized,
      hfNormalized,
      peakFrequency
    }
  }

  /**
   * NN間隔等間隔リサンプリング
   */
  private resampleNNIntervals(nnIntervals: number[], targetRate: number): number[] {
    // 累積時間計算
    const timestamps = [0]
    for (let i = 0; i < nnIntervals.length; i++) {
      timestamps.push(timestamps[timestamps.length - 1] + nnIntervals[i])
    }
    
    // 等間隔時間軸生成
    const duration = timestamps[timestamps.length - 1]
    const numSamples = Math.floor(duration * targetRate / 1000)
    const dt = 1000 / targetRate // ms
    
    const resampled: number[] = []
    
    for (let i = 0; i < numSamples; i++) {
      const targetTime = i * dt
      
      // 線形補間
      let index = 0
      for (let j = 1; j < timestamps.length; j++) {
        if (timestamps[j] > targetTime) {
          index = j - 1
          break
        }
      }
      
      if (index < nnIntervals.length) {
        const t1 = timestamps[index]
        const t2 = timestamps[index + 1]
        const nn1 = nnIntervals[index]
        const nn2 = index + 1 < nnIntervals.length ? nnIntervals[index + 1] : nn1
        
        const weight = (targetTime - t1) / (t2 - t1 + 1e-10)
        const interpolatedNN = nn1 + weight * (nn2 - nn1)
        resampled.push(interpolatedNN)
      }
    }
    
    return resampled
  }

  /**
   * パワースペクトル密度計算（Welch法）
   */
  private calculatePSD(signal: number[], samplingRate: number): number[] {
    // デトレンド（線形除去）
    const detrended = this.detrend(signal)
    
    // ハミング窓適用
    const windowed = this.applyHammingWindow(detrended)
    
    // FFT計算
    const fftResult = this.computeFFT(windowed)
    
    // パワースペクトル密度
    const psd = fftResult.slice(0, Math.floor(fftResult.length / 2)).map(complex => {
      const magnitude = Math.sqrt(complex.real * complex.real + complex.imag * complex.imag)
      return magnitude * magnitude / (samplingRate * signal.length)
    })
    
    return psd
  }

  /**
   * 帯域パワー計算
   */
  private calculateBandPower(psd: number[], lowFreq: number, highFreq: number, samplingRate: number): number {
    const df = samplingRate / (2 * psd.length)
    const lowIndex = Math.floor(lowFreq / df)
    const highIndex = Math.ceil(highFreq / df)
    
    let power = 0
    for (let i = lowIndex; i <= highIndex && i < psd.length; i++) {
      power += psd[i] * df
    }
    
    return power
  }

  // ============ 非線形解析 (12パラメータ) ============

  /**
   * 非線形HRVパラメータ解析
   */
  private async analyzeNonlinear(nnIntervals: number[]): Promise<any> {
    // Poincaréプロット解析
    const poincare = this.analyzePoincarePlot(nnIntervals)
    
    // エントロピー解析
    const sampleEntropy = this.calculateSampleEntropy(nnIntervals, 2, 0.2)
    const approximateEntropy = this.calculateApproximateEntropy(nnIntervals, 2, 0.2)
    
    // DFA解析
    const { dfa1, dfa2 } = this.calculateDFA(nnIntervals)
    
    // 複雑性解析
    const complexity = this.calculateComplexity(nnIntervals)
    const fractalDimension = this.calculateFractalDimension(nnIntervals)
    const recurrenceRate = this.calculateRecurrenceRate(nnIntervals)
    
    return {
      ...poincare,
      sampleEntropy,
      approximateEntropy,
      dfa1,
      dfa2,
      complexity,
      fractalDimension,
      recurrenceRate
    }
  }

  /**
   * Poincaréプロット解析
   */
  private analyzePoincarePlot(nnIntervals: number[]): any {
    const n = nnIntervals.length
    if (n < 2) return { sd1: 0, sd2: 0, sd1sd2Ratio: 0, csi: 0, cvi: 0 }
    
    // RR(n) vs RR(n+1)プロット
    const x = nnIntervals.slice(0, -1) // RR(n)
    const y = nnIntervals.slice(1)     // RR(n+1)
    
    // 主軸・副軸方向分散計算
    const meanX = this.calculateMean(x)
    const meanY = this.calculateMean(y)
    
    // 共分散行列
    const cov = this.calculateCovariance(x, y, meanX, meanY)
    const varX = this.calculateVariance(x, meanX)
    const varY = this.calculateVariance(y, meanY)
    
    // 固有値計算
    const trace = varX + varY
    const det = varX * varY - cov * cov
    const lambda1 = (trace + Math.sqrt(trace * trace - 4 * det)) / 2
    const lambda2 = (trace - Math.sqrt(trace * trace - 4 * det)) / 2
    
    // SD1, SD2計算
    const sd1 = Math.sqrt(lambda2) // 短期変動
    const sd2 = Math.sqrt(lambda1) // 長期変動
    const sd1sd2Ratio = sd1 / (sd2 + 1e-10)
    
    // 心臓自律神経指数
    const csi = sd2 / sd1 // 交感神経指数
    const cvi = Math.log10(sd1 * sd2) // 迷走神経指数
    
    return { sd1, sd2, sd1sd2Ratio, csi, cvi }
  }

  /**
   * サンプルエントロピー計算
   */
  private calculateSampleEntropy(data: number[], m: number, r: number): number {
    const n = data.length
    let A = 0, B = 0
    
    // 相対許容範囲計算
    const tolerance = r * this.calculateStandardDeviation(data)
    
    for (let i = 0; i < n - m; i++) {
      for (let j = i + 1; j < n - m; j++) {
        // m長さパターンマッチング
        let matchM = true
        for (let k = 0; k < m; k++) {
          if (Math.abs(data[i + k] - data[j + k]) > tolerance) {
            matchM = false
            break
          }
        }
        
        if (matchM) {
          B++
          
          // m+1長さパターンマッチング
          if (Math.abs(data[i + m] - data[j + m]) <= tolerance) {
            A++
          }
        }
      }
    }
    
    return A > 0 ? -Math.log(A / B) : Infinity
  }

  /**
   * 近似エントロピー計算
   */
  private calculateApproximateEntropy(data: number[], m: number, r: number): number {
    const n = data.length
    const tolerance = r * this.calculateStandardDeviation(data)
    
    const phi = (m: number): number => {
      let sum = 0
      
      for (let i = 0; i < n - m + 1; i++) {
        let matches = 0
        
        for (let j = 0; j < n - m + 1; j++) {
          let match = true
          for (let k = 0; k < m; k++) {
            if (Math.abs(data[i + k] - data[j + k]) > tolerance) {
              match = false
              break
            }
          }
          if (match) matches++
        }
        
        if (matches > 0) {
          sum += Math.log(matches / (n - m + 1))
        }
      }
      
      return sum / (n - m + 1)
    }
    
    return phi(m) - phi(m + 1)
  }

  /**
   * DFA（Detrended Fluctuation Analysis）計算
   */
  private calculateDFA(nnIntervals: number[]): { dfa1: number; dfa2: number } {
    const n = nnIntervals.length
    const mean = this.calculateMean(nnIntervals)
    
    // 積分時系列作成
    const integrated = new Array(n)
    integrated[0] = nnIntervals[0] - mean
    for (let i = 1; i < n; i++) {
      integrated[i] = integrated[i - 1] + (nnIntervals[i] - mean)
    }
    
    // スケール範囲設定
    const minScale = 4
    const maxScale = Math.floor(n / 4)
    const scales: number[] = []
    const fluctuations: number[] = []
    
    for (let scale = minScale; scale <= maxScale; scale *= 1.2) {
      const s = Math.floor(scale)
      if (s >= n / 4) break
      
      scales.push(s)
      
      // ボックス分割・線形除去
      const numBoxes = Math.floor(n / s)
      let sumSquaredError = 0
      
      for (let box = 0; box < numBoxes; box++) {
        const start = box * s
        const end = start + s
        
        // 線形回帰
        const y = integrated.slice(start, end)
        const x = Array.from({ length: s }, (_, i) => i)
        const { slope, intercept } = this.linearRegression(x, y)
        
        // デトレンド誤差計算
        for (let i = 0; i < s; i++) {
          const predicted = slope * i + intercept
          const error = y[i] - predicted
          sumSquaredError += error * error
        }
      }
      
      const fluctuation = Math.sqrt(sumSquaredError / (numBoxes * s))
      fluctuations.push(fluctuation)
    }
    
    // 対数線形回帰でスケーリング指数計算
    const logScales = scales.map(s => Math.log(s))
    const logFluctuations = fluctuations.map(f => Math.log(f))
    
    // 短期DFA (α1: スケール4-11)
    const shortRange = this.extractRange(logScales, logFluctuations, Math.log(4), Math.log(11))
    const dfa1 = shortRange.length > 1 ? this.linearRegression(shortRange.x, shortRange.y).slope : 1.0
    
    // 長期DFA (α2: スケール11以上)
    const longRange = this.extractRange(logScales, logFluctuations, Math.log(11), Math.log(maxScale))
    const dfa2 = longRange.length > 1 ? this.linearRegression(longRange.x, longRange.y).slope : 1.0
    
    return { dfa1, dfa2 }
  }

  // ============ ヘルパーメソッド ============

  private removeArtifacts(nnIntervals: number[]): number[] {
    // 生理学的範囲チェック（300-2000ms）
    const physiological = nnIntervals.filter(nn => nn >= 300 && nn <= 2000)
    
    // 外れ値除去（3σルール）
    const mean = this.calculateMean(physiological)
    const std = this.calculateStandardDeviation(physiological, mean)
    const threshold = 3 * std
    
    return physiological.filter(nn => Math.abs(nn - mean) <= threshold)
  }

  private assessDataQuality(cleanedIntervals: number[], originalIntervals: number[]): any {
    const artifactRatio = ((originalIntervals.length - cleanedIntervals.length) / originalIntervals.length) * 100
    const dataQuality = Math.max(0, 1 - artifactRatio / 50) // 50%未満で品質低下
    
    // 定常性チェック（分散の変化）
    const stationarity = this.assessStationarity(cleanedIntervals)
    
    // 信頼性スコア
    const reliability = (dataQuality * 0.6 + stationarity * 0.4)
    
    return { dataQuality, artifactRatio, stationarity, reliability }
  }

  private assessStationarity(intervals: number[]): number {
    const segmentSize = Math.floor(intervals.length / 4)
    if (segmentSize < 10) return 1.0
    
    const variances = []
    for (let i = 0; i < 4; i++) {
      const start = i * segmentSize
      const end = start + segmentSize
      const segment = intervals.slice(start, end)
      const variance = this.calculateVariance(segment)
      variances.push(variance)
    }
    
    const meanVar = this.calculateMean(variances)
    const varOfVar = this.calculateVariance(variances, meanVar)
    const cv = Math.sqrt(varOfVar) / meanVar
    
    return Math.max(0, 1 - cv)
  }

  // 基本統計関数
  private calculateMean(data: number[]): number {
    return data.reduce((sum, val) => sum + val, 0) / data.length
  }

  private calculateStandardDeviation(data: number[], mean?: number): number {
    const m = mean ?? this.calculateMean(data)
    const variance = data.reduce((sum, val) => sum + Math.pow(val - m, 2), 0) / data.length
    return Math.sqrt(variance)
  }

  private calculateVariance(data: number[], mean?: number): number {
    const m = mean ?? this.calculateMean(data)
    return data.reduce((sum, val) => sum + Math.pow(val - m, 2), 0) / data.length
  }

  private calculateCovariance(x: number[], y: number[], meanX: number, meanY: number): number {
    let sum = 0
    for (let i = 0; i < x.length; i++) {
      sum += (x[i] - meanX) * (y[i] - meanY)
    }
    return sum / x.length
  }

  private createNNHistogram(nnIntervals: number[], binWidth: number): number[] {
    const min = Math.min(...nnIntervals)
    const max = Math.max(...nnIntervals)
    const numBins = Math.ceil((max - min) / binWidth)
    const histogram = new Array(numBins).fill(0)
    
    for (const interval of nnIntervals) {
      const binIndex = Math.floor((interval - min) / binWidth)
      if (binIndex >= 0 && binIndex < numBins) {
        histogram[binIndex]++
      }
    }
    
    return histogram
  }

  private detrend(signal: number[]): number[] {
    const n = signal.length
    const x = Array.from({ length: n }, (_, i) => i)
    const { slope, intercept } = this.linearRegression(x, signal)
    
    return signal.map((val, i) => val - (slope * i + intercept))
  }

  private applyHammingWindow(signal: number[]): number[] {
    const n = signal.length
    return signal.map((val, i) => {
      const window = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (n - 1))
      return val * window
    })
  }

  private linearRegression(x: number[], y: number[]): { slope: number; intercept: number } {
    const n = x.length
    const sumX = x.reduce((a, b) => a + b, 0)
    const sumY = y.reduce((a, b) => a + b, 0)
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0)
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0)
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
    const intercept = (sumY - slope * sumX) / n
    
    return { slope, intercept }
  }

  private extractRange(x: number[], y: number[], minX: number, maxX: number): { x: number[]; y: number[]; length: number } {
    const indices = x.map((val, i) => ({ val, i }))
                     .filter(item => item.val >= minX && item.val <= maxX)
    
    return {
      x: indices.map(item => x[item.i]),
      y: indices.map(item => y[item.i]),
      length: indices.length
    }
  }

  private findPeakFrequency(psd: number[], lowFreq: number, highFreq: number, samplingRate: number): number {
    const df = samplingRate / (2 * psd.length)
    const lowIndex = Math.floor(lowFreq / df)
    const highIndex = Math.ceil(highFreq / df)
    
    let maxPower = 0
    let peakIndex = lowIndex
    
    for (let i = lowIndex; i <= highIndex && i < psd.length; i++) {
      if (psd[i] > maxPower) {
        maxPower = psd[i]
        peakIndex = i
      }
    }
    
    return peakIndex * df
  }

  private calculateComplexity(intervals: number[]): number {
    // サンプルエントロピーベースの複雑性指標
    return this.calculateSampleEntropy(intervals, 2, 0.15)
  }

  private calculateFractalDimension(intervals: number[]): number {
    // Higuchi法によるフラクタル次元計算（簡略版）
    const kmax = 5
    let logSum = 0
    let logkSum = 0
    let count = 0
    
    for (let k = 1; k <= kmax; k++) {
      let lk = 0
      const m = Math.floor((intervals.length - 1) / k)
      
      for (let i = 0; i < k; i++) {
        let lm = 0
        for (let j = 1; j <= m; j++) {
          if (i + j * k < intervals.length) {
            lm += Math.abs(intervals[i + j * k] - intervals[i + (j - 1) * k])
          }
        }
        lk += lm * (intervals.length - 1) / (m * k)
      }
      
      lk = lk / k
      if (lk > 0) {
        logSum += Math.log(lk)
        logkSum += Math.log(k)
        count++
      }
    }
    
    // 線形回帰の傾き = -フラクタル次元
    const slope = count > 1 ? -(logSum - count * logkSum / count) / (logkSum - count * logkSum / count) : 1.5
    return Math.abs(slope)
  }

  private calculateRecurrenceRate(intervals: number[]): number {
    const n = intervals.length
    const threshold = 0.1 * this.calculateStandardDeviation(intervals)
    let recurrences = 0
    
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(intervals[i] - intervals[j]) < threshold) {
          recurrences++
        }
      }
    }
    
    return recurrences / (n * (n - 1) / 2)
  }

  // FFT実装（再利用）
  private computeFFT(signal: number[]): Complex[] {
    const n = signal.length
    const paddedLength = Math.pow(2, Math.ceil(Math.log2(n)))
    const paddedSignal = [...signal, ...new Array(paddedLength - n).fill(0)]
    
    return this.fftRecursive(paddedSignal.map(val => ({ real: val, imag: 0 })))
  }

  private fftRecursive(x: Complex[]): Complex[] {
    const n = x.length
    if (n <= 1) return x
    
    const even = this.fftRecursive(x.filter((_, i) => i % 2 === 0))
    const odd = this.fftRecursive(x.filter((_, i) => i % 2 === 1))
    
    const result: Complex[] = new Array(n)
    
    for (let k = 0; k < n / 2; k++) {
      const angle = -2 * Math.PI * k / n
      const w = { real: Math.cos(angle), imag: Math.sin(angle) }
      const oddMultiplied = this.complexMultiply(w, odd[k])
      
      result[k] = this.complexAdd(even[k], oddMultiplied)
      result[k + n / 2] = this.complexSubtract(even[k], oddMultiplied)
    }
    
    return result
  }

  private complexAdd(a: Complex, b: Complex): Complex {
    return { real: a.real + b.real, imag: a.imag + b.imag }
  }

  private complexSubtract(a: Complex, b: Complex): Complex {
    return { real: a.real - b.real, imag: a.imag - b.imag }
  }

  private complexMultiply(a: Complex, b: Complex): Complex {
    return {
      real: a.real * b.real - a.imag * b.imag,
      imag: a.real * b.imag + a.imag * b.real
    }
  }
}

interface Complex {
  real: number
  imag: number
}