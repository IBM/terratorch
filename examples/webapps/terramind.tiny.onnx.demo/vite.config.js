import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  base: "/",
  build: {
    outDir: "dist", // default, but explicit
    emptyOutDir: true // clean dist before build
  },
  assetsInclude: ["**/*.onnx"],
  plugins: [
    VitePWA({
      strategies: "injectManifest",
      srcDir: ".",             // where sw_elephants.js lives
      filename: "sw_elephants.js",
      includeAssets: [
        "/icon-192.png",
        "/icon-512.png",
        "/model_tm_tiny_aed_elephants.onnx",
        "/tile_0_0_0.png",
        "/tile_0_1152_3072.png",
        "/tile_0_768_3072.png",
        "/tile_1260_0_2688.png",
        "/tile_100_3840_1536.png",
        "/tile_100_4224_1152.png",
        "/tile_100_4224_1536.png",
        "/tile_100_4224_768.png",
        "/tile_100_4608_0.png",
        "/tile_1261_4224_768.png",
        "/tile_1013_3072_3072.png",
        "/tile_101_3072_384.png",
        "/index.html",
      ],
      manifest: {
        name: "TerraMind.tiny wild elephant object detection",
        short_name: "FindElephant",
        start_url: "/",
        display: "standalone",
        background_color: "#ffffff",
        theme_color: "#317EFB",
        icons: [
          { src: "/icon-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
          { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "any maskable" }
        ]
      }
    })
  ]
});
