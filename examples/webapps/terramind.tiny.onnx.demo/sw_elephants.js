import { precacheAndRoute } from "workbox-precaching"

// This line is required for Workbox injection
precacheAndRoute(self.__WB_MANIFEST)

const CACHE_NAME = "tm-tiny-od-nzcattle-pwa"
const urlsToCache = [
  "/model_tm_tiny_aed_elephants.onnx",
  "/icon-192.png",
  "/icon-512.png",
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
]

/** 
self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(urlsToCache))
  )
})

self.addEventListener("fetch", event => {
  event.respondWith(
    caches.match(event.request).then(
      response => response || fetch(event.request)
    )
  )
})
*/