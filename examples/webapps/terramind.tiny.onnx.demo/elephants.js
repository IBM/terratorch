import "@carbon/web-components/es/components/accordion/index.js";
import "@carbon/web-components/es/components/textarea/index.js";
import "@carbon/web-components/es/components/skeleton-placeholder/index.js";
import "@carbon/web-components/es/components/progress-bar/index.js";

const log = (msg) => {
  document.getElementById("log").value += msg + "\n";
};
let session = null;

async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP error! ${response.status}`);
  const contentLength = response.headers.get("Content-Length");
  if (!contentLength) return await response.arrayBuffer();

  const total = parseInt(contentLength, 10);
  let loaded = 0;
  const reader = response.body.getReader();
  const chunks = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded / total);
  }
  return new Blob(chunks);
}

async function createInputTensorFromUrl(url) {
  log("Downloading and processing image...");
  const response = await fetch(url);
  const blob = await response.blob();
  const imageBitmap = await createImageBitmap(blob);
  document.getElementById("inputImage").src = URL.createObjectURL(blob);
  if (window.innerWidth < 672) {
    document.getElementById("inputImage").scrollIntoView();
  }
  return await imageBitmapToTensor(imageBitmap);
}

function drawBoxesOnImage(boxes) {
  log("Start drawing boxes..");
  const canvas = document.getElementById("overlayCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  boxes.forEach((det) => {
    let [x1, y1, x2, y2] = det.box;
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  });
}

function iou(boxA, boxB) {
  const [x1, y1, x2, y2] = boxA;
  const [x1b, y1b, x2b, y2b] = boxB;
  const xx1 = Math.max(x1, x1b);
  const yy1 = Math.max(y1, y1b);
  const xx2 = Math.min(x2, x2b);
  const yy2 = Math.min(y2, y2b);
  const w = Math.max(0, xx2 - xx1);
  const h = Math.max(0, yy2 - yy1);
  const inter = w * h;
  const areaA = (x2 - x1) * (y2 - y1);
  const areaB = (x2b - x1b) * (y2b - y1b);
  return inter / (areaA + areaB - inter);
}

function nms(dets, iouThreshold) {
  const keep = [];
  while (dets.length > 0) {
    const best = dets.shift();
    keep.push(best);
    dets = dets.filter((d) => iou(best.box, d.box) < iouThreshold);
  }
  return keep;
}

async function run(inputTensor) {
  const canvas = document.getElementById("overlayCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  log("Running inference...");
  var startTime = performance.now();
  const feeds = { [session.inputNames[0]]: inputTensor };
  const results = await session.run(feeds);
  var endTime = performance.now();
  log(`Done: ${endTime - startTime} ms`);
  const boxes = results[session.outputNames[0]].data;
  const scores = results[session.outputNames[1]].data;
  const labels = results[session.outputNames[2]].data;
  let dets = [];
  for (let i = 0; i < scores.length; i++) {
    const box = [boxes[i * 4 + 0], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3]];
    if (scores[i] > 0.6) {
      dets.push({ box, score: scores[i], label: Number(labels[i]) });
    }
  }
  dets.sort((a, b) => b.score - a.score);
  const selected = nms(dets, 0.5);
  log("Selected boxes after NMS: " + selected.length);
  drawBoxesOnImage(selected);
}

async function imageBitmapToTensor(imageBitmap) {
  const width = 512;
  const height = 512;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imageBitmap, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);
  const inputData = new Float32Array(1 * 3 * width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      inputData[0 * width * height + y * width + x] = data[i] / 255.0;
      inputData[1 * width * height + y * width + x] = data[i + 1] / 255.0;
      inputData[2 * width * height + y * width + x] = data[i + 2] / 255.0;
    }
  }
  return new ort.Tensor("float32", inputData, [1, 3, height, width]);
}

async function init() {
  const overlay = document.getElementById("overlay");
  const progressBar = document.getElementById("progressBar");

  const modelUrl = "model_tm_tiny_aed_elephants.onnx";
  log("Downloading model: " + modelUrl);
  const grid = document.getElementById("sampleGrid");
  grid.innerHTML = "";
  const sampleImages = [
    "tile_0_0_0.png",
    "tile_0_1536_1920.png",
    "tile_3_1536_768.png",
    "tile_9_1152_3072.png",
    "tile_14_768_2688.png",
    "tile_4_3072_2304.png",
    "tile_17_768_1152.png",
    "tile_15_1920_0.png",
    "tile_11_3840_1152.png",
    "tile_10_1536_768.png",
    "tile_0_1536_2304.png",
    "tile_0_384_3072.png",
  ];
  sampleImages.forEach((url) => {
    const skeleton = document.createElement("cds-skeleton-placeholder");
    grid.appendChild(skeleton);
  });

  const blob = await fetchWithProgress(modelUrl, (progress) => {
    progressBar.value = (progress * 100).toFixed(1);
  });

  const arrayBuffer = await blob.arrayBuffer();
  session = await ort.InferenceSession.create(arrayBuffer);
  grid.innerHTML = ""; // clear in case of reload
  progressBar.remove()
  sampleImages.forEach((url) => {
    const img = document.createElement("img");
    img.src = url;
    img.addEventListener("click", async () => {
      document.getElementById("loadUrlBtn").innerText = "Running inference...";
      const startTime = performance.now();
      const inputTensor = await createInputTensorFromUrl(url);
      await run(inputTensor);
      const endTime = performance.now();
      document.getElementById("loadUrlBtn").innerText = `Done: ${Math.round(endTime - startTime)} ms`;
    });
    grid.appendChild(img);
  });

  //   overlay.style.display = "none"; // unlock UI
  document.getElementById("loadUrlBtn").innerText = "Model downloaded. Please click on any image";
  log("Model loaded.");
}

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw_elephants.js")
      .then(reg => console.log("ServiceWorker registered:", reg))
      .catch(err => console.error("ServiceWorker registration failed:", err));
  });
}


init();
