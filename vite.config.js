import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      usePolling: true, // 도커 환경에서 파일 변경을 주기적으로 강제 확인하게 만드는 핵심 옵션
    }
  }
})