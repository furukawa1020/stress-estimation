/**
 * WebRTCManager - リアルタイムカメラアクセス管理
 * 学術研究レベルの精度でWebRTCストリームを制御
 */

export class WebRTCManager {
  private stream: MediaStream | null = null
  private constraints: MediaStreamConstraints
  private isInitialized = false

  constructor() {
    // 学術研究レベルの高品質設定
    this.constraints = {
      video: {
        width: { ideal: 1920, min: 640 },
        height: { ideal: 1080, min: 480 },
        frameRate: { ideal: 60, min: 30 },
        facingMode: 'user',
        // 研究用高精度設定
        advanced: [
          { width: 1920, height: 1080 },
          { aspectRatio: 16/9 },
          { frameRate: 60 }
        ] as any
      },
      audio: false // 今回は動画のみ
    }
  }

  /**
   * システム初期化
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return

    // ブラウザ対応確認
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('WebRTC is not supported in this browser')
    }

    // MediaDevices API の利用可能性確認
    try {
      const devices = await navigator.mediaDevices.enumerateDevices()
      const videoDevices = devices.filter(device => device.kind === 'videoinput')
      
      if (videoDevices.length === 0) {
        throw new Error('No video input devices found')
      }

      console.log(`Found ${videoDevices.length} video input device(s):`, videoDevices)
    } catch (error) {
      throw new Error(`Failed to enumerate devices: ${error}`)
    }

    this.isInitialized = true
  }

  /**
   * カメラストリーム開始
   */
  async startCamera(): Promise<MediaStream> {
    if (!this.isInitialized) {
      await this.initialize()
    }

    try {
      // 高解像度での取得を試行
      this.stream = await navigator.mediaDevices.getUserMedia(this.constraints)
      
      // ストリーム情報をログ出力（研究用）
      const videoTrack = this.stream.getVideoTracks()[0]
      const settings = videoTrack.getSettings()
      console.log('Camera stream settings:', settings)
      
      return this.stream
    } catch (error) {
      // 高解像度が失敗した場合、フォールバック設定で再試行
      console.warn('High resolution failed, trying fallback settings:', error)
      
      const fallbackConstraints: MediaStreamConstraints = {
        video: {
          width: { ideal: 1280, min: 640 },
          height: { ideal: 720, min: 480 },
          frameRate: { ideal: 30, min: 15 },
          facingMode: 'user'
        },
        audio: false
      }

      try {
        this.stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints)
        return this.stream
      } catch (fallbackError) {
        throw new Error(`Failed to access camera: ${fallbackError}`)
      }
    }
  }

  /**
   * カメラストリーム停止
   */
  async stopCamera(): Promise<void> {
    if (this.stream) {
      this.stream.getTracks().forEach(track => {
        track.stop()
        console.log(`Stopped track: ${track.kind} (${track.label})`)
      })
      this.stream = null
    }
  }

  /**
   * 現在のストリーム取得
   */
  getStream(): MediaStream | null {
    return this.stream
  }

  /**
   * カメラ設定変更
   */
  async updateConstraints(newConstraints: Partial<MediaStreamConstraints>): Promise<void> {
    this.constraints = { ...this.constraints, ...newConstraints }
    
    if (this.stream) {
      await this.stopCamera()
      await this.startCamera()
    }
  }

  /**
   * 利用可能なカメラデバイス一覧取得
   */
  async getAvailableCameras(): Promise<MediaDeviceInfo[]> {
    if (!this.isInitialized) {
      await this.initialize()
    }

    const devices = await navigator.mediaDevices.enumerateDevices()
    return devices.filter(device => device.kind === 'videoinput')
  }

  /**
   * 特定のカメラデバイスに切り替え
   */
  async switchCamera(deviceId: string): Promise<void> {
    const currentVideo = this.constraints.video as MediaTrackConstraints
    this.constraints.video = {
      ...currentVideo,
      deviceId: { exact: deviceId }
    }

    if (this.stream) {
      await this.stopCamera()
      await this.startCamera()
    }
  }

  /**
   * ストリーム統計情報取得（研究用）
   */
  async getStreamStats(): Promise<any> {
    if (!this.stream) return null

    const videoTrack = this.stream.getVideoTracks()[0]
    if (!videoTrack) return null

    return {
      settings: videoTrack.getSettings(),
      capabilities: videoTrack.getCapabilities ? videoTrack.getCapabilities() : null,
      constraints: videoTrack.getConstraints(),
      readyState: videoTrack.readyState,
      enabled: videoTrack.enabled,
      muted: videoTrack.muted
    }
  }

  /**
   * リソース解放
   */
  dispose(): void {
    this.stopCamera()
    this.isInitialized = false
  }
}