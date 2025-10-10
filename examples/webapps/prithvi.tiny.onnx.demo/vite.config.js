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
      filename: "sw_nasa.js",
      includeAssets: [
        "/index.html",
        "/model_tm_tiny_aed_chesapeake.onnx",
        "/icon-nasa-192.png",
        "/icon-512.png",
        "/sample_3_x.bin",
        "/sample_3_y.png",
        "/sample_3_rgb.png",
        "/sample_16_x.bin",
        "/sample_16_y.png",
        "/sample_16_rgb.png",
        "/sample_18_x.bin",
        "/sample_18_y.png",
        "/sample_18_rgb.png",
        "/sample_21_x.bin",
        "/sample_21_y.png",
        "/sample_21_rgb.png",
      ],
      manifest: {
        name: "Prithvi.tiny LULC segmentation",
        short_name: "PrithviLULC",
        start_url: "/",
        display: "standalone",
        background_color: "#ffffff",
        theme_color: "#317EFB",
        icons: [
          { src: "/icon-nasa-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
          { src: "/icon-nasa-512.png", sizes: "512x512", type: "image/png", purpose: "any maskable" }
        ]
      }
    })
  ]
});
