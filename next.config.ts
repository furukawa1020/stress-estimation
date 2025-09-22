import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Vercel対応: Turbopack設定を環境別に分離
  ...(process.env.NODE_ENV === 'development' && {
    experimental: {
      turbo: {
        root: __dirname,
      },
    },
  }),
  
  // Vercel最適化設定
  poweredByHeader: false,
  reactStrictMode: true,
  
  // WebRTC・カメラアクセス用セキュリティヘッダー
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Permissions-Policy',
            value: 'camera=self, microphone=self',
          },
        ],
      },
    ];
  },
};

export default nextConfig;
