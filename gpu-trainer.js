'use strict';
// Full GPU training pipeline. Mirrors the CPU Transformer forward/backward
// but keeps all data as GPU textures between operations.

class GPUTrainer {
    constructor(gl, config) {
        this.ops = new WebGLOps(gl);
        this.config = config;
        this.lr = 3e-4;
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.eps = 1e-8;
        this.weightDecay = 0.01;
        this.adamT = 0;

        const { vocabSize, dModel, nHeads, nLayers, dFF, maxSeqLen } = config;
        this.dHead = dModel / nHeads;

        // Build parameter definitions (mirrors Transformer._buildParamDefs)
        this.paramDefs = [];
        const add = (name, rows, cols, wd) => {
            this.paramDefs.push({ name, rows, cols, size: rows * cols, wd: !!wd });
        };
        add('tok_emb', vocabSize, dModel, true);
        add('pos_emb', maxSeqLen, dModel, true);
        for (let l = 0; l < nLayers; l++) {
            add(`b${l}.ln1.g`, 1, dModel, false);
            add(`b${l}.ln1.b`, 1, dModel, false);
            add(`b${l}.attn.Wq`, dModel, dModel, true);
            add(`b${l}.attn.bq`, 1, dModel, false);
            add(`b${l}.attn.Wk`, dModel, dModel, true);
            add(`b${l}.attn.bk`, 1, dModel, false);
            add(`b${l}.attn.Wv`, dModel, dModel, true);
            add(`b${l}.attn.bv`, 1, dModel, false);
            add(`b${l}.attn.Wo`, dModel, dModel, true);
            add(`b${l}.attn.bo`, 1, dModel, false);
            add(`b${l}.ln2.g`, 1, dModel, false);
            add(`b${l}.ln2.b`, 1, dModel, false);
            add(`b${l}.ffn.W1`, dModel, dFF, true);
            add(`b${l}.ffn.b1`, 1, dFF, false);
            add(`b${l}.ffn.W2`, dFF, dModel, true);
            add(`b${l}.ffn.b2`, 1, dModel, false);
        }
        add('ln_f.g', 1, dModel, false);
        add('ln_f.b', 1, dModel, false);
        add('W_out', dModel, vocabSize, true);
        add('b_out', 1, vocabSize, false);

        // Compute offsets into the flat CPU param array
        let off = 0;
        for (const d of this.paramDefs) { d.offset = off; off += d.size; }
        this.totalParams = off;

        this.w = {};  // weight textures
        this.m = {};  // Adam m textures
        this.v = {};  // Adam v textures
        this.grad = {}; // gradient textures (written each step)
    }

    uploadWeights(params) {
        const ops = this.ops;
        for (const d of this.paramDefs) {
            const data = params.subarray(d.offset, d.offset + d.size);
            if (this.w[d.name]) ops.free(this.w[d.name]);
            if (this.m[d.name]) ops.free(this.m[d.name]);
            if (this.v[d.name]) ops.free(this.v[d.name]);
            this.w[d.name] = ops.fromData(data, d.rows, d.cols);
            this.m[d.name] = ops.zeros(d.rows, d.cols);
            this.v[d.name] = ops.zeros(d.rows, d.cols);
        }
        this.adamT = 0;
    }

    downloadWeights() {
        const params = new Float32Array(this.totalParams);
        for (const d of this.paramDefs) {
            const data = this.ops.read(this.w[d.name]);
            params.set(data, d.offset);
        }
        return params;
    }

    downloadOptimizer() {
        const m = new Float32Array(this.totalParams);
        const v = new Float32Array(this.totalParams);
        for (const d of this.paramDefs) {
            m.set(this.ops.read(this.m[d.name]), d.offset);
            v.set(this.ops.read(this.v[d.name]), d.offset);
        }
        return { m, v, t: this.adamT };
    }

    uploadOptimizer(m, v, t) {
        const ops = this.ops;
        for (const d of this.paramDefs) {
            ops.free(this.m[d.name]);
            ops.free(this.v[d.name]);
            this.m[d.name] = ops.fromData(m.subarray(d.offset, d.offset + d.size), d.rows, d.cols);
            this.v[d.name] = ops.fromData(v.subarray(d.offset, d.offset + d.size), d.rows, d.cols);
        }
        this.adamT = t;
    }

    // --- Forward pass ---
    forward(tokenIdArray) {
        const ops = this.ops;
        const { dModel: D, nHeads: H, nLayers: L, dFF, vocabSize: V } = this.config;
        const dH = this.dHead;
        const S = tokenIdArray.length;
        const scale = 1 / Math.sqrt(dH);
        const cache = { layers: [], S };
        const w = this.w;

        // Token IDs as texture [1, S]
        const tokTex = ops.fromData(new Float32Array(tokenIdArray), 1, S);
        cache.tokTex = tokTex;

        let x = ops.embed(tokTex, w['tok_emb'], w['pos_emb']);

        for (let l = 0; l < L; l++) {
            const lc = {};
            lc.xIn = x;

            // Layer norm 1
            const ln1 = ops.layerNorm(x, w[`b${l}.ln1.g`], w[`b${l}.ln1.b`]);
            lc.ln1 = ln1; // {out, xNorm, stats}
            const lnOut = ln1.out;
            lc.lnOut = lnOut;

            // QKV projections
            const Q = ops.matmulBias(lnOut, w[`b${l}.attn.Wq`], w[`b${l}.attn.bq`], S, D, D);
            const K = ops.matmulBias(lnOut, w[`b${l}.attn.Wk`], w[`b${l}.attn.bk`], S, D, D);
            const Vi = ops.matmulBias(lnOut, w[`b${l}.attn.Wv`], w[`b${l}.attn.bv`], S, D, D);
            lc.Q = Q; lc.K = K; lc.V = Vi;

            // Multi-head attention
            lc.attnW = [];
            let attnConcat = ops.zeros(S, D);
            for (let h = 0; h < H; h++) {
                const Qh = ops.extractHead(Q, h, dH);
                const Kh = ops.extractHead(K, h, dH);
                const Vh = ops.extractHead(Vi, h, dH);
                const scores = ops.matmul(Qh, Kh, S, dH, S, false, true);
                const attn = ops.softmaxCausal(scores, scale);
                lc.attnW.push(attn);
                const outH = ops.matmul(attn, Vh, S, S, dH);
                const newConcat = ops.scatterHead(attnConcat, outH, h, dH);
                ops.free(attnConcat); attnConcat = newConcat;
                ops.free(Qh); ops.free(Kh); ops.free(Vh); ops.free(scores); ops.free(outH);
            }
            lc.attnConcat = attnConcat;

            // Output projection + residual
            const attnOut = ops.matmulBias(attnConcat, w[`b${l}.attn.Wo`], w[`b${l}.attn.bo`], S, D, D);
            x = ops.add(lc.xIn, attnOut);
            ops.free(attnOut);

            lc.xMid = x;

            // Layer norm 2
            const ln2 = ops.layerNorm(x, w[`b${l}.ln2.g`], w[`b${l}.ln2.b`]);
            lc.ln2 = ln2;
            const ln2Out = ln2.out;
            lc.ln2Out = ln2Out;

            // FFN
            const h1 = ops.matmulBias(ln2Out, w[`b${l}.ffn.W1`], w[`b${l}.ffn.b1`], S, D, dFF);
            lc.ffnPre = h1;
            const h1Act = ops.gelu(h1);
            lc.ffnPost = h1Act;
            const h2 = ops.matmulBias(h1Act, w[`b${l}.ffn.W2`], w[`b${l}.ffn.b2`], S, dFF, D);

            // Residual
            x = ops.add(lc.xMid, h2);
            ops.free(h2);

            cache.layers.push(lc);
        }

        // Final layer norm
        const lnF = ops.layerNorm(x, w['ln_f.g'], w['ln_f.b']);
        cache.lnF = lnF;
        cache.xPreFinal = x;
        const xFinal = lnF.out;
        cache.xFinal = xFinal;

        // Output projection
        const logits = ops.matmulBias(xFinal, w['W_out'], w['b_out'], S, D, V);
        return { logits, cache };
    }

    // --- Backward pass ---
    backward(cache, dLogits) {
        const ops = this.ops;
        const { dModel: D, nHeads: H, nLayers: L, dFF, vocabSize: V } = this.config;
        const dH = this.dHead;
        const S = cache.S;
        const scale = 1 / Math.sqrt(dH);
        const w = this.w;
        const g = this.grad;

        // Output projection gradients
        g['W_out'] = ops.matmul(cache.xFinal, dLogits, D, S, V, true);
        g['b_out'] = ops.colSum(dLogits);
        let dx = ops.matmul(dLogits, w['W_out'], S, V, D, false, true);

        // Final layer norm backward
        const { dgamma: dgF, dbeta: dbF } = ops.layerNormBackParams(dx, cache.lnF.xNorm);
        g['ln_f.g'] = dgF; g['ln_f.b'] = dbF;
        const dxLnF = ops.layerNormBackDx(dx, cache.lnF.xNorm, cache.lnF.stats, w['ln_f.g']);
        ops.free(dx); dx = dxLnF;

        for (let l = L - 1; l >= 0; l--) {
            const lc = cache.layers[l];

            // FFN residual: dx flows to both xMid and FFN path
            const dxMid = dx; // save reference (don't free, used for add below)

            // FFN backward
            g[`b${l}.ffn.W2`] = ops.matmul(lc.ffnPost, dx, dFF, S, D, true);
            g[`b${l}.ffn.b2`] = ops.colSum(dx);
            let dh1 = ops.matmul(dx, w[`b${l}.ffn.W2`], S, D, dFF, false, true);
            const dh1g = ops.geluBack(dh1, lc.ffnPre);
            ops.free(dh1); dh1 = dh1g;
            g[`b${l}.ffn.W1`] = ops.matmul(lc.ln2Out, dh1, D, S, dFF, true);
            g[`b${l}.ffn.b1`] = ops.colSum(dh1);
            let dln2 = ops.matmul(dh1, w[`b${l}.ffn.W1`], S, dFF, D, false, true);
            ops.free(dh1);

            // LN2 backward
            const { dgamma: dg2, dbeta: db2 } = ops.layerNormBackParams(dln2, lc.ln2.xNorm);
            g[`b${l}.ln2.g`] = dg2; g[`b${l}.ln2.b`] = db2;
            const dxLn2 = ops.layerNormBackDx(dln2, lc.ln2.xNorm, lc.ln2.stats, w[`b${l}.ln2.g`]);
            ops.free(dln2);

            dx = ops.add(dxMid, dxLn2);
            ops.free(dxMid); ops.free(dxLn2);

            // Attention residual
            const dxIn = dx;

            // Output projection backward
            g[`b${l}.attn.Wo`] = ops.matmul(lc.attnConcat, dx, D, S, D, true);
            g[`b${l}.attn.bo`] = ops.colSum(dx);
            let dConcat = ops.matmul(dx, w[`b${l}.attn.Wo`], S, D, D, false, true);

            // Per-head attention backward
            let dQ = ops.zeros(S, D), dK = ops.zeros(S, D), dV = ops.zeros(S, D);

            for (let h = 0; h < H; h++) {
                const dOutH = ops.extractHead(dConcat, h, dH);
                const Vh = ops.extractHead(lc.V, h, dH);
                const Qh = ops.extractHead(lc.Q, h, dH);
                const Kh = ops.extractHead(lc.K, h, dH);
                const attn = lc.attnW[h];

                const dAttn = ops.matmul(dOutH, Vh, S, dH, S, false, true);
                const dVh = ops.matmul(attn, dOutH, S, S, dH, true);
                const dScores = ops.softmaxBack(dAttn, attn, scale);
                const dQh = ops.matmul(dScores, Kh, S, S, dH);
                const dKh = ops.matmul(dScores, Qh, S, S, dH, true);

                const ndQ = ops.scatterHead(dQ, dQh, h, dH); ops.free(dQ); dQ = ndQ;
                const ndK = ops.scatterHead(dK, dKh, h, dH); ops.free(dK); dK = ndK;
                const ndV = ops.scatterHead(dV, dVh, h, dH); ops.free(dV); dV = ndV;

                ops.free(dOutH); ops.free(Vh); ops.free(Qh); ops.free(Kh);
                ops.free(dAttn); ops.free(dVh); ops.free(dScores); ops.free(dQh); ops.free(dKh);
            }
            ops.free(dConcat);

            // QKV projection gradients
            g[`b${l}.attn.Wq`] = ops.matmul(lc.lnOut, dQ, D, S, D, true);
            g[`b${l}.attn.bq`] = ops.colSum(dQ);
            g[`b${l}.attn.Wk`] = ops.matmul(lc.lnOut, dK, D, S, D, true);
            g[`b${l}.attn.bk`] = ops.colSum(dK);
            g[`b${l}.attn.Wv`] = ops.matmul(lc.lnOut, dV, D, S, D, true);
            g[`b${l}.attn.bv`] = ops.colSum(dV);

            let dln1 = ops.matmul(dQ, w[`b${l}.attn.Wq`], S, D, D, false, true);
            const dln1K = ops.matmul(dK, w[`b${l}.attn.Wk`], S, D, D, false, true);
            const dln1V = ops.matmul(dV, w[`b${l}.attn.Wv`], S, D, D, false, true);
            const dln1Sum = ops.add(dln1, dln1K);
            ops.free(dln1); ops.free(dln1K);
            const dln1Sum2 = ops.add(dln1Sum, dln1V);
            ops.free(dln1Sum); ops.free(dln1V);
            dln1 = dln1Sum2;

            ops.free(dQ); ops.free(dK); ops.free(dV);

            // LN1 backward
            const { dgamma: dg1, dbeta: db1 } = ops.layerNormBackParams(dln1, lc.ln1.xNorm);
            g[`b${l}.ln1.g`] = dg1; g[`b${l}.ln1.b`] = db1;
            const dxLn1 = ops.layerNormBackDx(dln1, lc.ln1.xNorm, lc.ln1.stats, w[`b${l}.ln1.g`]);
            ops.free(dln1);

            dx = ops.add(dxIn, dxLn1);
            ops.free(dxIn); ops.free(dxLn1);
        }

        // Embedding gradients
        g['tok_emb'] = ops.embedBackTok(dx, cache.tokTex, this.config.vocabSize);
        g['pos_emb'] = ops.embedBackPos(dx, this.config.maxSeqLen);
        ops.free(dx);
    }

    freeCache(cache) {
        const ops = this.ops;
        ops.free(cache.tokTex);
        for (const lc of cache.layers) {
            ops.free(lc.xIn); ops.free(lc.lnOut);
            ops.free(lc.ln1.xNorm); ops.free(lc.ln1.stats);
            ops.free(lc.Q); ops.free(lc.K); ops.free(lc.V);
            for (const a of lc.attnW) ops.free(a);
            ops.free(lc.attnConcat); ops.free(lc.xMid);
            ops.free(lc.ln2Out); ops.free(lc.ln2.xNorm); ops.free(lc.ln2.stats);
            ops.free(lc.ffnPre); ops.free(lc.ffnPost);
        }
        ops.free(cache.xPreFinal);
        ops.free(cache.lnF.out); ops.free(cache.lnF.xNorm); ops.free(cache.lnF.stats);
    }

    // --- Optimizer ---
    optimizerStep() {
        this.adamT++;
        const bc1 = 1 - Math.pow(this.beta1, this.adamT);
        const bc2 = 1 - Math.pow(this.beta2, this.adamT);
        const ops = this.ops;

        for (const d of this.paramDefs) {
            const n = d.name;
            const wd = d.wd ? this.weightDecay : 0;
            const result = ops.adamW(this.w[n], this.grad[n], this.m[n], this.v[n],
                this.lr, this.beta1, this.beta2, bc1, bc2, this.eps, wd);
            ops.free(this.w[n]); ops.free(this.m[n]); ops.free(this.v[n]);
            ops.free(this.grad[n]);
            this.w[n] = result.param;
            this.m[n] = result.m;
            this.v[n] = result.v;
        }
    }

    // --- Public API ---

    trainStep(input, target) {
        const ops = this.ops;
        const S = input.length;
        const V = this.config.vocabSize;

        const { logits, cache } = this.forward(input);

        // Loss computation (read back small [S,1] texture)
        const targetTex = ops.fromData(new Float32Array(target), 1, S);
        const lossTex = ops.ceLoss(logits, targetTex);
        const lossArr = ops.read(lossTex);
        let loss = 0;
        for (let i = 0; i < S; i++) loss += lossArr[i];
        loss /= S;
        ops.free(lossTex);

        // Gradient of logits
        const dLogits = ops.ceGrad(logits, targetTex);
        ops.free(targetTex); ops.free(logits);

        // Backward pass
        this.backward(cache, dLogits);
        ops.free(dLogits);
        this.freeCache(cache);

        // Optimizer step
        this.optimizerStep();

        return loss;
    }

    generate(tokenIds, maxNew, temperature) {
        temperature = temperature || 0.8;
        const { maxSeqLen, vocabSize } = this.config;
        const tokens = Array.from(tokenIds);
        const ops = this.ops;

        for (let t = 0; t < maxNew && tokens.length < maxSeqLen; t++) {
            const { logits, cache } = this.forward(new Uint16Array(tokens));
            // Read back last row of logits
            const allLogits = ops.read(logits);
            ops.free(logits);
            this.freeCache(cache);

            const S = tokens.length;
            const lastOff = (S - 1) * vocabSize;
            // Temperature-scaled softmax on CPU (only V=128 values)
            let maxVal = -Infinity;
            for (let i = 0; i < vocabSize; i++)
                maxVal = Math.max(maxVal, allLogits[lastOff + i] / temperature);
            let sumExp = 0;
            for (let i = 0; i < vocabSize; i++)
                sumExp += Math.exp(allLogits[lastOff + i] / temperature - maxVal);

            let r = Math.random(), cumsum = 0;
            for (let i = 0; i < vocabSize; i++) {
                cumsum += Math.exp(allLogits[lastOff + i] / temperature - maxVal) / sumExp;
                if (r < cumsum) { tokens.push(i); break; }
            }
            if (tokens.length === S) tokens.push(0);
        }
        return new Uint16Array(tokens);
    }
}

if (typeof module !== 'undefined') module.exports = { GPUTrainer };
