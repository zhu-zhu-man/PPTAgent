const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    host: '0.0.0.0',  // 允许外部访问
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:9297',
        changeOrigin: true,
        secure: false,
        logLevel: 'debug',
        onProxyReq: (proxyReq, req, res) => {
          console.log('Proxying request:', req.method, req.url, '->', proxyReq.path);
        },
        onError: (err, req, res) => {
          console.error('Proxy error:', err.message);
        }
      },
      '/wsapi': {
        target: 'ws://127.0.0.1:9297',
        ws: true,
        changeOrigin: true,
        secure: false,
        logLevel: 'debug'
      },
    },
    port: 8088
  },
  css: {
    loaderOptions: {
      postcss: { postcssOptions: { config: false } }
    }
  }
})
