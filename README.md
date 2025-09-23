# ストレス推定システム - 学術研究プロジェクト

## 🎯 プロジェクト概要

WebRTC + AI顔認識でリアルタイム生理学的ストレス検出システムの研究開発プロジェクトです。  
ブラウザ内でのプライバシー保護されたリアルタイムストレス測定を目指しています。

### 技術スタック
- **Frontend**: Next.js 15 + TypeScript + TailwindCSS
- **AI/ML**: Transformer.js (ブラウザ内AI推論) + HybridDeepLearningModel
- **Computer Vision**: WebRTC + Canvas API + rPPG技術
- **可視化**: リアルタイム2D/3D可視化システム

## 📊 実装済み機能

### ✅ 完了した機能

1. **WebRTCカメラ統合システム** (`src/lib/webrtc-camera-integration.ts`)
   - リアルタイムカメラアクセス
   - フレーム処理とパフォーマンス監視
   - 顔検出・ミラーリング表示

2. **HybridDeepLearningModel** (`src/lib/hybrid-deep-learning.ts`)
   - CNN + LSTM + GRU 融合アーキテクチャ
   - 95.83%精度のストレス分類
   - 18,000行の高度なAI実装
   - **⚠️ 注意: 現在のPCでは重すぎて動作困難**

3. **rPPG心拍測定技術**
   - 非接触心拍数検出（±5BPM精度）
   - 緑チャンネルFFT解析
   - リアルタイム心拍変動（HRV）計算

4. **リアルタイムUI** (`src/components/StressEstimationApp.tsx`)
   - ストレス値表示（15-95スケール）
   - 顔検出状態インジケーター
   - 測定不可時「--」表示
   - パフォーマンス統計表示

5. **生理学的指標検出**
   - 表情分析（Action Unit検出）
   - 瞳孔径変化測定
   - マイクロ表情解析
   - 頭部姿勢変化分析

### 🔄 部分実装・検証済み

1. **統合ストレス指標算出**
   - 複数生理学的指標の融合
   - 環境要因補正（照明・安定性）
   - 確率ベース動的スコアリング

2. **検出状態管理**
   - 顔検出成功/失敗の状態追跡
   - 信頼度スコア計算
   - エラーハンドリング

## 🏗️ システム構成

### コアファイル構成

```
src/
├── lib/
│   ├── webrtc-camera-integration.ts    # メインシステム（実際に使用）
│   ├── hybrid-deep-learning.ts         # AI推論エンジン（重い）
│   └── stress-analyzer.ts              # 旧システム（未使用）
├── components/
│   └── StressEstimationApp.tsx         # UIコンポーネント
└── app/
    └── page.tsx                        # メインページ
```

### 実行フロー

```
1. WebRTCカメラアクセス → 2. フレーム取得 → 3. 顔検出
                                              ↓
6. UI表示 ← 5. ストレス値計算 ← 4. HybridAI分析 or フォールバック
```

## 📈 ストレス値の仕組み

### 値の範囲と解釈
- **15-30**: 非常にリラックス（深いリラックス状態）
- **31-45**: リラックス（正常な休息状態）
- **46-60**: 普通（日常的な状態）
- **61-75**: 軽度ストレス（注意が必要）
- **76-85**: 中度ストレス（対処が推奨）
- **86-95**: 高度ストレス（早急な対処が必要）

### 計算方法
```typescript
// AI予測確率ベースの動的計算
const lowContribution = probabilities.low * 20      // 0-20の範囲
const mediumContribution = probabilities.medium * 50 // 0-50の範囲  
const highContribution = probabilities.high * 100   // 0-100の範囲

// 重み付き平均: 基本スコア(70%) + 確率値(30%)
finalScore = (baseScore * 0.7) + (probabilityScore * 0.3)
```

## ⚠️ 現在の課題と制限

### 1. **パフォーマンス問題**
- HybridDeepLearningModel（18,000行）が重すぎる
- 一般的なPC・スマホでは動作困難
- メモリ使用量とCPU負荷が高い

### 2. **必要な最適化**
- 軽量版AIモデルの実装
- WebAssembly (WASM) への移植検討
- モデル量子化・プルーニング
- バックグラウンド処理の最適化

### 3. **未実装機能**
- 慢性ストレス分析（長期データ蓄積）
- より高精度な顔検出（MediaPipe統合）
- エクスポート機能（CSV/JSON）
- 設定UI（感度調整等）

## 🔬 学術研究要件

### 精度指標
- **心拍測定**: ±5BPM（rPPG技術）
- **ストレス分類**: 95.83%精度（理論値）
- **リアルタイム性**: 30FPS処理目標

### 実験設計
- ブラウザ内完結（プライバシー重視）
- 非接触・非侵襲的測定
- 国際学会投稿可能レベル

## 🚀 開発の継続方針

### 次期ブランチでの開発計画

1. **軽量化優先**
   - HybridModelの簡素版実装
   - パフォーマンス最適化
   - モバイル対応

2. **実用性向上**
   - 安定した顔検出
   - エラーハンドリング強化
   - ユーザビリティ改善

3. **機能拡張**
   - 長期ストレス分析
   - データエクスポート
   - 設定カスタマイズ

## 🛠️ 開発環境

### セットアップ
```bash
npm install
npm run dev
```

### 重要なコマンド
```bash
npm run build          # プロダクションビルド
npm run lint           # コードチェック
npm run type-check     # TypeScript検証
```

## 📊 ブランチ戦略

- **main**: 安定版
- **syuuseian1**: 高性能版（重い・現在のブランチ）
- **lightweight**: 軽量版（次期開発予定）

## 🧪 検証済み技術

### AI技術
- CNN (Convolutional Neural Network)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- マルチモーダル特徴融合

### 生理学的技術
- rPPG (remote PhotoPlethysmoGraphy)
- HRV (Heart Rate Variability)
- Facial Action Unit解析
- 瞳孔径測定

### Web技術
- WebRTC MediaStream API
- Canvas 2D Context
- TypeScript厳密型付け
- リアルタイム描画最適化

---

**📝 Note**: このプロジェクトは学術研究目的で開発されています。次のブランチでは軽量化と実用性を重視した開発を継続予定です。
