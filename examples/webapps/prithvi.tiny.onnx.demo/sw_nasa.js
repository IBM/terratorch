import { precacheAndRoute } from "workbox-precaching"

// This line is required for Workbox injection
precacheAndRoute(self.__WB_MANIFEST)

const CACHE_NAME = "prithvi-nasa-pwa";
const urlsToCache = [
  "/",
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
];

/**
self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener("fetch", event => {
  event.respondWith(
    caches.match(event.request).then(response => response || fetch(event.request))
  );
});
 */
