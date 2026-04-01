"use strict";
importScripts("tokenizer.js", "transformer.js", "webgl-matmul.js", "webgl-ops.js", "gpu-trainer.js");

let tokenizer, transformer, optimizer, gpuTrainer;
let textData;
let training = false;
let step = 0;
let stockPrompts = [];
let computeMode = "cpu"; // "gpu-full", "gpu-matmul", "cpu"

function initFullGPU(config, lr) {
    try {
        const canvas = new OffscreenCanvas(1, 1);
        const gl = canvas.getContext("webgl2");
        if (!gl) return false;

        const trainer = new GPUTrainer(gl, config);
        // Initialize from a fresh CPU model to get random weights
        const cpuModel = new Transformer(config);
        trainer.uploadWeights(cpuModel.params);
        trainer.lr = lr;

        // Quick sanity test: one training step on dummy data
        const seqLen = Math.min(4, config.maxSeqLen);
        const testIn = new Uint16Array(seqLen).fill(65);
        const testTgt = new Uint16Array(seqLen).fill(66);
        const loss = trainer.trainStep(testIn, testTgt);
        if (isNaN(loss) || !isFinite(loss)) return false;

        // Re-initialize with fresh weights (the sanity test modified them)
        trainer.uploadWeights(cpuModel.params);
        trainer.lr = lr;
        gpuTrainer = trainer;
        return true;
    } catch (e) {
        return false;
    }
}

function initGPUMatmul() {
    try {
        const canvas = new OffscreenCanvas(1, 1);
        const gl = canvas.getContext("webgl2");
        if (!gl) return false;
        const gpuMM = new WebGLMatMul(gl);
        const a = new Float32Array([1, 2, 3, 4]);
        const b = new Float32Array([5, 6, 7, 8]);
        const c = gpuMM.compute(a, b, 2, 2, 2);
        const expected = [19, 22, 43, 50];
        for (let i = 0; i < 4; i++)
            if (Math.abs(c[i] - expected[i]) > 0.01) return false;
        setMatmulImpl(function (A, B, M, K, N, transA, transB) {
            return gpuMM.compute(A, B, M, K, N, transA, transB);
        });
        return true;
    } catch (e) {
        return false;
    }
}

self.onmessage = function (e) {
    const msg = e.data;
    switch (msg.type) {
        case "init": {
            tokenizer = new BPETokenizer(msg.merges || []);
            const lr = msg.lr || 3e-4;

            // Try full GPU pipeline first
            if (initFullGPU(msg.config, lr)) {
                computeMode = "gpu-full";
            } else {
                // Fall back to CPU transformer with optional GPU matmul
                if (initGPUMatmul()) computeMode = "gpu-matmul";
                else computeMode = "cpu";
                transformer = new Transformer(msg.config);
                optimizer = new AdamW(transformer, { lr });
            }

            textData = new Uint16Array(msg.textData);
            stockPrompts = msg.stockPrompts || [];
            const totalParams = gpuTrainer ? gpuTrainer.totalParams : transformer.totalParams;
            self.postMessage({
                type: "ready",
                totalParams,
                config: msg.config,
                gpu: computeMode,
            });
            break;
        }
        case "start":
            training = true;
            trainLoop();
            break;

        case "stop":
            training = false;
            break;

        case "generate": {
            const tokens = tokenizer.encode(msg.prompt);
            let output;
            if (gpuTrainer) {
                output = gpuTrainer.generate(tokens, msg.maxTokens || 200, msg.temperature || 0.8);
            } else {
                output = transformer.generate(tokens, msg.maxTokens || 200, msg.temperature || 0.8);
            }
            self.postMessage({ type: "generated", text: tokenizer.decode(output), id: msg.id });
            break;
        }
        case "setLR":
            if (gpuTrainer) gpuTrainer.lr = msg.lr;
            else if (optimizer) optimizer.lr = msg.lr;
            break;

        case "save": {
            let weights, optM, optV, optT, lr;
            if (gpuTrainer) {
                weights = gpuTrainer.downloadWeights();
                const os = gpuTrainer.downloadOptimizer();
                optM = os.m; optV = os.v; optT = os.t;
                lr = gpuTrainer.lr;
            } else {
                weights = new Float32Array(transformer.params);
                optM = new Float32Array(optimizer.m);
                optV = new Float32Array(optimizer.v);
                optT = optimizer.t;
                lr = optimizer.lr;
            }
            const config = gpuTrainer ? gpuTrainer.config : transformer.config;
            self.postMessage({ type: "saveData", weights, optM, optV, optT, step, lr, config, merges: tokenizer.merges });
            break;
        }

        case "loadModel": {
            const { config, weights, optM, optV, optT, savedStep, lr, merges } = msg;
            step = savedStep || 0;
            tokenizer = new BPETokenizer(merges || []);
            if (gpuTrainer) {
                gpuTrainer.uploadWeights(weights);
                if (optM && optV) gpuTrainer.uploadOptimizer(optM, optV, optT || 0);
                gpuTrainer.lr = lr || 3e-4;
            } else {
                transformer = new Transformer(config);
                transformer.params.set(weights);
                optimizer = new AdamW(transformer, { lr: lr || 3e-4 });
                if (optM && optV) {
                    optimizer.m.set(optM);
                    optimizer.v.set(optV);
                    optimizer.t = optT || 0;
                }
            }
            self.postMessage({ type: "modelLoaded", step, config });
            break;
        }

        case "setStockPrompts":
            stockPrompts = msg.prompts;
            break;
    }
};

function trainLoop() {
    if (!training) return;

    const seqLen = gpuTrainer ? gpuTrainer.config.maxSeqLen : transformer.config.maxSeqLen;
    if (textData.length < seqLen + 1) {
        self.postMessage({ type: "error", message: "Training data too short for sequence length" });
        training = false;
        return;
    }

    const t0 = performance.now();

    const maxStart = textData.length - seqLen - 1;
    const startIdx = Math.floor(Math.random() * maxStart);
    const input = textData.subarray(startIdx, startIdx + seqLen);
    const target = textData.subarray(startIdx + 1, startIdx + seqLen + 1);

    let loss;
    if (gpuTrainer) {
        loss = gpuTrainer.trainStep(input, target);
    } else {
        const { logits, cache } = transformer.forward(input);
        const res = transformer.lossAndGrad(logits, target, seqLen);
        loss = res.loss;
        transformer.backward(cache, res.dLogits);
        optimizer.step();
    }
    step++;

    const elapsed = performance.now() - t0;
    const charsPerSec = Math.round(seqLen / (elapsed / 1000));

    self.postMessage({ type: "step", step, loss, elapsed, charsPerSec, lr: gpuTrainer ? gpuTrainer.lr : optimizer.lr });

    if (step % 100 === 0) {
        if (gpuTrainer) {
            self.postMessage({ type: "weights", weights: gpuTrainer.downloadWeights() });
        } else {
            self.postMessage({ type: "weights", weights: new Float32Array(transformer.params) });
        }
    }

    if (step % 1000 === 0 && stockPrompts.length > 0) {
        evaluateStockPrompts();
    }

    setTimeout(trainLoop, 0);
}

function evaluateStockPrompts() {
    const results = stockPrompts.map((prompt) => {
        const tokens = tokenizer.encode(prompt);
        let output;
        if (gpuTrainer) {
            output = gpuTrainer.generate(tokens, 100, 0.8);
        } else {
            output = transformer.generate(tokens, 100, 0.8);
        }
        return { prompt, completion: tokenizer.decode(output) };
    });
    self.postMessage({ type: "stockPrompts", results, step });
}
