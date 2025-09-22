/** @type {import('next').NextConfig} */
const nextConfig = {
  turbo: {
    moduleIdStrategy: 'deterministic',
  },
  headers: async () => [
    {
      source: '/(.*)',
      headers: [
        {
          key: 'Cross-Origin-Embedder-Policy',
          value: 'require-corp',
        },
        {
          key: 'Cross-Origin-Opener-Policy',
          value: 'same-origin',
        },
        {
          key: 'Permissions-Policy',
          value: 'camera=(self)',
        },
      ],
    },
  ],
  outputFileTracingRoot: process.cwd(),
  serverComponentsExternalPackages: [
    '@tensorflow/tfjs',
    '@tensorflow/tfjs-node',
    '@vladmandic/face-api',
  ],
  experimental: {
    turbo: {},
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/mediapipe/:path*',
        destination: '/api/mediapipe/:path*',
      },
    ];
  },
  transpilePackages: ['@tensorflow/tfjs'],
  webpack: (config, { isServer }) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      '@': require('path').resolve(__dirname, 'src'),
    };

    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
        crypto: false,
      };
    }

    return config;
  },
};

module.exports = nextConfig;