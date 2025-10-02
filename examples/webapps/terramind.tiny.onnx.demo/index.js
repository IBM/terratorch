const log = (msg) => {
  document.getElementById("log").textContent += msg + "\n";
};
let session = null;

const defaultImageUrl =
  "https://terramind-tiny-onnx-mobile-benchmark.s3.eu-de.cloud-object-storage.appdomain.cloud/nzcattleex1.jpg";

// Add 12 sample images
const sampleImages = [
  "https://terramind-tiny-onnx-mobile-benchmark.s3.eu-de.cloud-object-storage.appdomain.cloud/nzcattleex1.jpg",
  "https://terramind-tiny-onnx-mobile-benchmark.s3.eu-de.cloud-object-storage.appdomain.cloud/nzcattleex2.jpg",
  "https://terramind-tiny-onnx-mobile-benchmark.s3.eu-de.cloud-object-storage.appdomain.cloud/nzcattleex3.jpg",
  "https://terramind-tiny-onnx-mobile-benchmark.s3.eu-de.cloud-object-storage.appdomain.cloud/nzcattleex4.jpg",
  "https://terramind-tiny-onnx-mobile-benchmark.s3.eu-de.cloud-object-storage.appdomain.cloud/nzcattleex5.jpg",
  "https://terramind-tiny-onnx-mobile-benchmark.s3.eu-de.cloud-object-storage.appdomain.cloud/nzcattleex6.jpg",
  "https://terramind-tiny-onnx-mobile-benchmark.s3.eu-de.cloud-object-storage.appdomain.cloud/nzcattleex7.jpg",
  "https://terramind-tiny-onnx-mobile-benchmark.s3.eu-de.cloud-object-storage.appdomain.cloud/nzcattleex8.jpg",
  "https://terramind-tiny-onnx-mobile-benchmark.s3.eu-de.cloud-object-storage.appdomain.cloud/nzcattleex9.jpg",
  "https://terramind-tiny-onnx-mobile-benchmark.s3.eu-de.cloud-object-storage.appdomain.cloud/nzcattleex10.jpg",
  "https://terramind-tiny-onnx-mobile-benchmark.s3.eu-de.cloud-object-storage.appdomain.cloud/nzcattleex11.jpg",
  "https://terramind-tiny-onnx-mobile-benchmark.s3.eu-de.cloud-object-storage.appdomain.cloud/nzcattleex12.jpg",
];

const grid = document.getElementById("sampleGrid");
sampleImages.forEach((url) => {
  const img = document.createElement("img");
  img.src = url;
  img.addEventListener("click", async () => {
    if (!document.getElementById("sampleGrid").disabled) {
      document.getElementById("loadUrlBtn").innerText = "Running inference...";
      document.getElementById("sampleGrid").disabled = true;
      var startTime = performance.now();
      const inputTensor = await createInputTensorFromUrl(url);
      run(inputTensor);
      var endTime = performance.now();
      document.getElementById("loadUrlBtn").innerText = `Done: ${endTime - startTime} ms`;
      document.getElementById("sampleGrid").disabled = false;
    }
  });
  grid.appendChild(img);
});

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

async function createInputTensorFromUrl(url) {
  log("Downloading and processing image...");
  const response = await fetch(url);
  const blob = await response.blob();
  const imageBitmap = await createImageBitmap(blob);
  document.getElementById("inputImage").src = URL.createObjectURL(blob);
  return await imageBitmapToTensor(imageBitmap);
}

async function createInputTensorFromFile(file) {
  log("Processing uploaded image...");
  const blobUrl = URL.createObjectURL(file);
  const imageBitmap = await createImageBitmap(file);
  document.getElementById("inputImage").src = blobUrl;
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

/**document.getElementById('loadUrlBtn').addEventListener('click', async () => {
      const inputTensor = await createInputTensorFromUrl(defaultImageUrl);
      run(inputTensor);
    });

    document.getElementById('imageUpload').addEventListener('change', async (event) => {
      if (event.target.files.length > 0) {
        const file = event.target.files[0];
        const inputTensor = await createInputTensorFromFile(file);
        run(inputTensor);
      }
    });**/

async function init() {
  document.getElementById("sampleGrid").disabled = true;

  //document.getElementById('loadUrlBtn').disabled = true;
  //document.getElementById('imageUpload').disabled = true;
  const modelUrl = "model_tm_tiny_nzcattle.onnx";
  log("Downloading model: " + modelUrl);
  session = await ort.InferenceSession.create(modelUrl);
  document.getElementById("sampleGrid").disabled = false;

  document.getElementById("loadUrlBtn").innerText = "Model downloaded. Please click on any image";
  log("Model loaded.");
  //document.getElementById('loadUrlBtn').disabled = false;
  //document.getElementById('imageUpload').disabled = false;
}

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw_elephants.js")
      .then(reg => console.log("ServiceWorker registered:", reg))
      .catch(err => console.error("ServiceWorker registration failed:", err));
  });
}

init();
