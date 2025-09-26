# TM Tiny Web Demo
This is a PWA (Progressive Web App) demo to run a fine-tuned TerraMind.tiny model as ONNX withing a browser/mobile device. It uses the WASM (Web Assembly) ONNX execution engine.

The model was fine-tuned using the The Aerial Elephant Dataset: https://zenodo.org/records/3234780


Just copy the contents of the /dist folder to a web server/servable COS bucket of your choise. Or simply run 'python3 -m http.server' from the /dist folder to test locally.

Please also copy the ONNX model to the /dist folder before. You can get it from: https://terramind-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/model_tm_tiny_aed_elephants.onnx

