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
  const total = parseInt(response.headers.get("Content-Length") || "0", 10);
  if (!total) return await response.arrayBuffer();
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded / total);
  }
  return new Blob(chunks);
}

function logHistogram(data, numBins = 10) {
  // Compute min and max iteratively
  let min = data[0],
    max = data[0];
  for (let i = 1; i < data.length; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }

  const binSize = (max - min) / numBins;
  const bins = new Array(numBins).fill(0);

  for (let i = 0; i < data.length; i++) {
    let bin = Math.floor((data[i] - min) / binSize);
    if (bin === numBins) bin = numBins - 1; // include max value
    bins[bin]++;
  }

  console.log("Histogram:");
  bins.forEach((count, idx) => {
    const rangeStart = (min + idx * binSize).toFixed(2);
    const rangeEnd = (min + (idx + 1) * binSize).toFixed(2);
    console.log(`${rangeStart} - ${rangeEnd}: ${count}`);
  });
}

async function createInputTensorFromBin(url, channelOrder = ["RED", "GREEN", "BLUE", "NIR"]) {
  log("Downloading and processing .bin...");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP error ${response.status}`);
  const arrayBuffer = await response.arrayBuffer();
  let data = new Float32Array(arrayBuffer);

  const H = 224,
    W = 224,
    C = 4;
  if (data.length !== H * W * C) {
    throw new Error(`Bin size mismatch! Got ${data.length}, expected ${H * W * C}`);
  }

  // reshape into [C,H,W]
  let tensorData = [];
  for (let c = 0; c < C; c++) {
    const channelData = data.slice(c * H * W, (c + 1) * H * W);
    tensorData.push(channelData);
  }

  // reorder channels according to channelOrder
  const channelNames = ["RED", "GREEN", "BLUE", "NIR"]; // original order
  const reorderedData = [];
  for (let ch of channelOrder) {
    const idx = channelNames.indexOf(ch);
    if (idx === -1) throw new Error(`Unknown channel "${ch}"`);
    reorderedData.push(...tensorData[idx]);
  }
  logHistogram(reorderedData);
  return new ort.Tensor("float32", new Float32Array(reorderedData), [1, C, H, W]);
}

// Draw segmentation mask with semi-transparent colors
function drawSegmentationMask(mask, width, height, canvas) {
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(width, height);
  const colors = [
    [0, 0, 255], // class 0
    [10, 18, 255], // class 1 water
    [34, 139, 34], // class 2 trees
    [124, 252, 0], // class 3 streets
    [128, 128, 128], // class 4
    [255, 0, 0], // class 5
    [255, 255, 0], // class 6
  ];
  for (let i = 0; i < width * height; i++) {
    const [r, g, b] = colors[mask[i] % colors.length];
    const j = i * 4;
    imageData.data[j] = r;
    imageData.data[j + 1] = g;
    imageData.data[j + 2] = b;
    imageData.data[j + 3] = 128; // semi-transparent
  }
  ctx.putImageData(imageData, 0, 0);
}

// Run model inference and visualize output
async function run(inputTensor) {
  log("Running segmentation inference...");
  const feeds = { [session.inputNames[0]]: inputTensor };
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]];
  const logits = output.data;
  const width = 224,
    height = 224,
    numPixels = width * height,
    numClasses = logits.length / numPixels;
  const mask = new Uint8Array(numPixels);
  for (let i = 0; i < numPixels; i++) {
    let maxVal = -Infinity,
      maxClass = 0;
    for (let c = 0; c < numClasses; c++) {
      const val = logits[c * numPixels + i];
      if (val > maxVal) {
        maxVal = val;
        maxClass = c;
      }
    }
    mask[i] = maxClass;
  }
  return mask;
}

async function init() {
  const overlay = document.getElementById("overlay");
  const progressBar = document.getElementById("progressBar");
  const modelUrl = "model_chesapeake.onnx";
  log("Downloading model: " + modelUrl);
  const grid = document.getElementById("sampleGrid");
  grid.innerHTML = "";

  let sampleFiles = [
    { rgb: "sample_3_rgb.png", bin: "sample_3_x.bin", gtMask: "sample_3_y.png", predMask: "sample_3_pred_mask.png" },
    {
      rgb: "sample_16_rgb.png",
      bin: "sample_16_x.bin",
      gtMask: "sample_16_y.png",
      predMask: "sample_16_pred_mask.png",
    },
    {
      rgb: "sample_18_rgb.png",
      bin: "sample_18_x.bin",
      gtMask: "sample_18_y.png",
      predMask: "sample_18_pred_mask.png",
    },
    {
      rgb: "sample_21_rgb.png",
      bin: "sample_21_x.bin",
      gtMask: "sample_21_y.png",
      predMask: "sample_21_pred_mask.png",
    },
    {
      rgb: "sample_23_rgb.png",
      bin: "sample_23_x.bin",
      gtMask: "sample_23_y.png",
      predMask: "sample_23_pred_mask.png",
    },
    {
      rgb: "sample_37_rgb.png",
      bin: "sample_37_x.bin",
      gtMask: "sample_37_y.png",
      predMask: "sample_37_pred_mask.png",
    },
    {
      rgb: "sample_39_rgb.png",
      bin: "sample_39_x.bin",
      gtMask: "sample_39_y.png",
      predMask: "sample_39_pred_mask.png",
    },
    {
      rgb: "sample_40_rgb.png",
      bin: "sample_40_x.bin",
      gtMask: "sample_40_y.png",
      predMask: "sample_40_pred_mask.png",
    },
    {
      rgb: "sample_42_rgb.png",
      bin: "sample_42_x.bin",
      gtMask: "sample_42_y.png",
      predMask: "sample_42_pred_mask.png",
    },
    {
      rgb: "sample_49_rgb.png",
      bin: "sample_49_x.bin",
      gtMask: "sample_49_y.png",
      predMask: "sample_49_pred_mask.png",
    },
    {
      rgb: "sample_51_rgb.png",
      bin: "sample_51_x.bin",
      gtMask: "sample_51_y.png",
      predMask: "sample_51_pred_mask.png",
    },
    {
      rgb: "sample_80_rgb.png",
      bin: "sample_80_x.bin",
      gtMask: "sample_80_y.png",
      predMask: "sample_80_pred_mask.png",
    },
    {
      rgb: "sample_108_rgb.png",
      bin: "sample_108_x.bin",
      gtMask: "sample_108_y.png",
      predMask: "sample_108_pred_mask.png",
    },
    {
      rgb: "sample_111_rgb.png",
      bin: "sample_111_x.bin",
      gtMask: "sample_111_y.png",
      predMask: "sample_111_pred_mask.png",
    },
    {
      rgb: "sample_112_rgb.png",
      bin: "sample_112_x.bin",
      gtMask: "sample_112_y.png",
      predMask: "sample_112_pred_mask.png",
    },
    {
      rgb: "sample_113_rgb.png",
      bin: "sample_113_x.bin",
      gtMask: "sample_113_y.png",
      predMask: "sample_113_pred_mask.png",
    },
  ];
  sampleFiles.forEach(({ rgb }, index) => {
    const skeleton = document.createElement("cds-skeleton-placeholder");
    grid.appendChild(skeleton);
  });

  const blob = await fetchWithProgress(modelUrl, (p) => {
    progressBar.value = (p * 100).toFixed(1);
  });
  const arrayBuffer = await blob.arrayBuffer();
  session = await ort.InferenceSession.create(arrayBuffer);
  grid.innerHTML = "";
  progressBar.remove();

  sampleFiles.forEach(({ rgb }, index) => {
    const img = document.createElement("img");
    img.src = rgb;
    img.addEventListener("click", async () => {
      document.getElementById("loadUrlBtn").innerText = "Running inference...";
      const inputTensor = await createInputTensorFromBin(sampleFiles[index].bin);
      const mask = await run(inputTensor);
      addResultRow(sampleFiles[index], mask);
      document.getElementById("loadUrlBtn").innerText = "Done";
    });
    grid.appendChild(img);
  });

  document.getElementById("loadUrlBtn").innerText = "Model downloaded. Please click on any image";
  log("Model loaded.");
}

// Adds a results row with all requested images/canvases
function addResultRow({ rgb, gtMask, predMask }, mask) {
  const container = document.getElementById("resultsContainer");
  container.replaceChildren(); // clears everything

  const row = document.createElement("div");
  row.className = "results-row";

  // Input RGB image (same as clicked preview)
  var col = document.createElement("div");
  var title = document.createElement("h5");
  title.innerText = "RGB-Preview";
  title.style.textAlign = "center";
  col.appendChild(title);
  const inputImg = document.createElement("img");
  inputImg.src = rgb;
  col.appendChild(inputImg);
  row.appendChild(col);

  /**
      // Ground truth mask
      const gtImg = document.createElement("img");
      gtImg.src = gtMask;
      row.appendChild(gtImg);
      **/

  /**
      // Predicted mask image (from Python saved pred_mask.png)
      if (predMask) {
        const predMaskImg = document.createElement("img");
        predMaskImg.src = predMask;
        row.appendChild(predMaskImg);
      }
      */

  // Model output mask canvas
  var col = document.createElement("div");
  var title = document.createElement("h5");
  title.innerText = "Prediction";
  title.style.textAlign = "center";
  col.appendChild(title);
  const modelMaskCanvas = document.createElement("canvas");
  modelMaskCanvas.width = 224;
  modelMaskCanvas.height = 224;
  col.appendChild(modelMaskCanvas);
  row.appendChild(col);

  // Overlay container: base RGB + mask canvas on top
  const overlayDiv = document.createElement("div");
  overlayDiv.className = "overlay-container";

  const baseRgbImg = document.createElement("img");
  baseRgbImg.src = rgb;
  overlayDiv.appendChild(baseRgbImg);

  var col = document.createElement("div");
  var title = document.createElement("h5");
  title.innerText = "Overlay";
  title.style.textAlign = "center";
  col.appendChild(title);
  const overlayCanvas = document.createElement("canvas");
  overlayCanvas.width = 224;
  overlayCanvas.height = 224;
  overlayDiv.appendChild(overlayCanvas);
  col.appendChild(overlayDiv);
  row.appendChild(col);

  container.prepend(row); // prepending to show latest on top

  // Draw masks on respective canvases
  drawSegmentationMask(mask, 224, 224, modelMaskCanvas);
  drawSegmentationMask(mask, 224, 224, overlayCanvas);
  if (window.innerWidth < 672) {
    container.scrollIntoView();
  }
}

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw_nasa.js").then(
      (reg) => console.log("ServiceWorker registered:", reg),
      (err) => console.error("ServiceWorker registration failed:", err)
    );
  });
}

init();
