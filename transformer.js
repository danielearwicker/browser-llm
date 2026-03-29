'use strict';

// ===== Math helpers =====

function randn() {
    const u1 = Math.random(), u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1 || 1e-30)) * Math.cos(2 * Math.PI * u2);
}

// General matrix multiply: C = op(A) @ op(B)
// Logical shapes: op(A) is [M, K], op(B) is [K, N], C is [M, N]
// Storage: A is [transA ? K : M, transA ? M : K], B is [transB ? N : K, transB ? K : N]
function cpuMatmul(A, B, M, K, N, transA, transB) {
    const C = new Float32Array(M * N);
    if (!transA && !transB) {
        for (let i = 0; i < M; i++) {
            const iK = i * K, iN = i * N;
            for (let k = 0; k < K; k++) {
                const a = A[iK + k], kN = k * N;
                for (let j = 0; j < N; j++) C[iN + j] += a * B[kN + j];
            }
        }
    } else if (transA && !transB) {
        for (let k = 0; k < K; k++) {
            const kM = k * M, kN = k * N;
            for (let i = 0; i < M; i++) {
                const a = A[kM + i], iN = i * N;
                for (let j = 0; j < N; j++) C[iN + j] += a * B[kN + j];
            }
        }
    } else if (!transA && transB) {
        for (let i = 0; i < M; i++) {
            const iK = i * K, iN = i * N;
            for (let j = 0; j < N; j++) {
                let sum = 0;
                const jK = j * K;
                for (let k = 0; k < K; k++) sum += A[iK + k] * B[jK + k];
                C[iN + j] = sum;
            }
        }
    } else {
        for (let i = 0; i < M; i++) {
            const iN = i * N;
            for (let j = 0; j < N; j++) {
                let sum = 0;
                const jK = j * K;
                for (let k = 0; k < K; k++) sum += A[k * M + i] * B[jK + k];
                C[iN + j] = sum;
            }
        }
    }
    return C;
}

// Pluggable matmul: GPU implementation can be swapped in via setMatmulImpl()
var _mmImpl = cpuMatmul;
function matmul(A, B, M, K, N, transA, transB) {
    return _mmImpl(A, B, M, K, N, transA, transB);
}
function setMatmulImpl(fn) { _mmImpl = fn; }

function addBias(M, bias, rows, cols) {
    for (let i = 0; i < rows; i++) {
        const off = i * cols;
        for (let j = 0; j < cols; j++) M[off + j] += bias[j];
    }
}

function colSum(M, out, rows, cols) {
    for (let i = 0; i < rows; i++) {
        const off = i * cols;
        for (let j = 0; j < cols; j++) out[j] += M[off + j];
    }
}

function extractHead(X, S, D, h, dHead) {
    const out = new Float32Array(S * dHead);
    for (let i = 0; i < S; i++) {
        const srcOff = i * D + h * dHead, dstOff = i * dHead;
        for (let k = 0; k < dHead; k++) out[dstOff + k] = X[srcOff + k];
    }
    return out;
}

function scatterHead(dst, src, S, D, h, dHead) {
    for (let i = 0; i < S; i++) {
        const dstOff = i * D + h * dHead, srcOff = i * dHead;
        for (let k = 0; k < dHead; k++) dst[dstOff + k] += src[srcOff + k];
    }
}

// ===== Activation functions =====

const SQRT_2_PI = Math.sqrt(2 / Math.PI);

function geluForward(x, n) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const xi = x[i];
        const s = SQRT_2_PI * (xi + 0.044715 * xi * xi * xi);
        out[i] = 0.5 * xi * (1 + Math.tanh(s));
    }
    return out;
}

function geluBackward(dout, x, n) {
    const dx = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const xi = x[i];
        const s = SQRT_2_PI * (xi + 0.044715 * xi * xi * xi);
        const ts = Math.tanh(s);
        const ds = SQRT_2_PI * (1 + 3 * 0.044715 * xi * xi);
        dx[i] = dout[i] * (0.5 * (1 + ts) + 0.5 * xi * (1 - ts * ts) * ds);
    }
    return dx;
}

function softmax(x, rows, cols) {
    const out = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
        const off = i * cols;
        let maxVal = -Infinity;
        for (let j = 0; j < cols; j++) maxVal = Math.max(maxVal, x[off + j]);
        let sumExp = 0;
        for (let j = 0; j < cols; j++) {
            out[off + j] = Math.exp(x[off + j] - maxVal);
            sumExp += out[off + j];
        }
        for (let j = 0; j < cols; j++) out[off + j] /= sumExp;
    }
    return out;
}

// ===== Layer norm =====

function layerNormForward(x, gamma, beta, S, D) {
    const out = new Float32Array(S * D);
    const xNorm = new Float32Array(S * D);
    const rstd = new Float32Array(S);

    for (let i = 0; i < S; i++) {
        const off = i * D;
        let m = 0;
        for (let j = 0; j < D; j++) m += x[off + j];
        m /= D;

        let v = 0;
        for (let j = 0; j < D; j++) { const d = x[off + j] - m; v += d * d; }
        v /= D;
        const rs = 1 / Math.sqrt(v + 1e-5);
        rstd[i] = rs;

        for (let j = 0; j < D; j++) {
            const xn = (x[off + j] - m) * rs;
            xNorm[off + j] = xn;
            out[off + j] = gamma[j] * xn + beta[j];
        }
    }
    return { out, cache: { xNorm, rstd, S, D } };
}

function layerNormBackward(dout, cache, gamma, dgamma, dbeta) {
    const { xNorm, rstd, S, D } = cache;
    const dx = new Float32Array(S * D);

    for (let i = 0; i < S; i++) {
        const off = i * D;
        for (let j = 0; j < D; j++) {
            dgamma[j] += dout[off + j] * xNorm[off + j];
            dbeta[j] += dout[off + j];
        }
    }

    for (let i = 0; i < S; i++) {
        const off = i * D;
        let sumDxn = 0, sumDxnXn = 0;
        for (let j = 0; j < D; j++) {
            const dxn = dout[off + j] * gamma[j];
            sumDxn += dxn;
            sumDxnXn += dxn * xNorm[off + j];
        }
        sumDxn /= D;
        sumDxnXn /= D;
        for (let j = 0; j < D; j++) {
            const dxn = dout[off + j] * gamma[j];
            dx[off + j] = rstd[i] * (dxn - sumDxn - xNorm[off + j] * sumDxnXn);
        }
    }
    return dx;
}

// ===== Transformer =====

const DEFAULT_CONFIG = {
    vocabSize: 128,
    dModel: 128,
    nHeads: 4,
    nLayers: 4,
    dFF: 512,
    maxSeqLen: 256,
};

class Transformer {
    constructor(config) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.paramDefs = [];
        this._buildParamDefs();
        this.params = new Float32Array(this.totalParams);
        this.grads = new Float32Array(this.totalParams);
        this._buildIndex();
        this._initParams();
    }

    _buildParamDefs() {
        const { vocabSize, dModel, nHeads, nLayers, dFF, maxSeqLen } = this.config;
        const add = (name, ...shape) => {
            const size = shape.reduce((a, b) => a * b, 1);
            this.paramDefs.push({ name, shape, size });
        };
        add('tok_emb', vocabSize, dModel);
        add('pos_emb', maxSeqLen, dModel);
        for (let l = 0; l < nLayers; l++) {
            add(`b${l}.ln1.g`, dModel);
            add(`b${l}.ln1.b`, dModel);
            add(`b${l}.attn.Wq`, dModel, dModel);
            add(`b${l}.attn.bq`, dModel);
            add(`b${l}.attn.Wk`, dModel, dModel);
            add(`b${l}.attn.bk`, dModel);
            add(`b${l}.attn.Wv`, dModel, dModel);
            add(`b${l}.attn.bv`, dModel);
            add(`b${l}.attn.Wo`, dModel, dModel);
            add(`b${l}.attn.bo`, dModel);
            add(`b${l}.ln2.g`, dModel);
            add(`b${l}.ln2.b`, dModel);
            add(`b${l}.ffn.W1`, dModel, dFF);
            add(`b${l}.ffn.b1`, dFF);
            add(`b${l}.ffn.W2`, dFF, dModel);
            add(`b${l}.ffn.b2`, dModel);
        }
        add('ln_f.g', dModel);
        add('ln_f.b', dModel);
        add('W_out', dModel, vocabSize);
        add('b_out', vocabSize);

        let offset = 0;
        for (const def of this.paramDefs) { def.offset = offset; offset += def.size; }
        this.totalParams = offset;
    }

    _buildIndex() {
        this.paramIndex = {};
        for (const def of this.paramDefs) this.paramIndex[def.name] = def;
    }

    p(name) {
        const d = this.paramIndex[name];
        return this.params.subarray(d.offset, d.offset + d.size);
    }

    g(name) {
        const d = this.paramIndex[name];
        return this.grads.subarray(d.offset, d.offset + d.size);
    }

    _initParams() {
        const { nLayers } = this.config;
        const std = 0.02;
        const residStd = std / Math.sqrt(2 * nLayers);
        const fill = (arr, s) => { for (let i = 0; i < arr.length; i++) arr[i] = randn() * s; };

        fill(this.p('tok_emb'), std);
        fill(this.p('pos_emb'), std);
        for (let l = 0; l < nLayers; l++) {
            this.p(`b${l}.ln1.g`).fill(1);
            this.p(`b${l}.ln1.b`).fill(0);
            fill(this.p(`b${l}.attn.Wq`), std);
            this.p(`b${l}.attn.bq`).fill(0);
            fill(this.p(`b${l}.attn.Wk`), std);
            this.p(`b${l}.attn.bk`).fill(0);
            fill(this.p(`b${l}.attn.Wv`), std);
            this.p(`b${l}.attn.bv`).fill(0);
            fill(this.p(`b${l}.attn.Wo`), residStd);
            this.p(`b${l}.attn.bo`).fill(0);
            this.p(`b${l}.ln2.g`).fill(1);
            this.p(`b${l}.ln2.b`).fill(0);
            fill(this.p(`b${l}.ffn.W1`), std);
            this.p(`b${l}.ffn.b1`).fill(0);
            fill(this.p(`b${l}.ffn.W2`), residStd);
            this.p(`b${l}.ffn.b2`).fill(0);
        }
        this.p('ln_f.g').fill(1);
        this.p('ln_f.b').fill(0);
        fill(this.p('W_out'), std);
        this.p('b_out').fill(0);
    }

    forward(tokenIds) {
        const S = tokenIds.length;
        const { dModel, nHeads, nLayers, dFF, vocabSize } = this.config;
        const dHead = dModel / nHeads;
        const scale = 1 / Math.sqrt(dHead);
        const cache = { layers: [], S, tokenIds: new Uint16Array(tokenIds) };

        // Embedding
        const tokEmb = this.p('tok_emb'), posEmb = this.p('pos_emb');
        let x = new Float32Array(S * dModel);
        for (let i = 0; i < S; i++) {
            const tOff = tokenIds[i] * dModel, pOff = i * dModel, xOff = i * dModel;
            for (let j = 0; j < dModel; j++) x[xOff + j] = tokEmb[tOff + j] + posEmb[pOff + j];
        }

        for (let l = 0; l < nLayers; l++) {
            const lc = { xIn: x.slice() };

            // Layer norm 1
            const ln1 = layerNormForward(x, this.p(`b${l}.ln1.g`), this.p(`b${l}.ln1.b`), S, dModel);
            lc.ln1 = ln1.cache;
            lc.lnOut = ln1.out;

            // QKV projections
            let Q = matmul(ln1.out, this.p(`b${l}.attn.Wq`), S, dModel, dModel);
            addBias(Q, this.p(`b${l}.attn.bq`), S, dModel);
            let K = matmul(ln1.out, this.p(`b${l}.attn.Wk`), S, dModel, dModel);
            addBias(K, this.p(`b${l}.attn.bk`), S, dModel);
            let V = matmul(ln1.out, this.p(`b${l}.attn.Wv`), S, dModel, dModel);
            addBias(V, this.p(`b${l}.attn.bv`), S, dModel);
            lc.Q = Q; lc.K = K; lc.V = V;

            // Multi-head attention
            const attnWeights = [];
            const attnConcat = new Float32Array(S * dModel);
            for (let h = 0; h < nHeads; h++) {
                const Qh = extractHead(Q, S, dModel, h, dHead);
                const Kh = extractHead(K, S, dModel, h, dHead);
                const Vh = extractHead(V, S, dModel, h, dHead);

                const scores = matmul(Qh, Kh, S, dHead, S, false, true);
                for (let i = 0; i < S * S; i++) scores[i] *= scale;
                for (let i = 0; i < S; i++)
                    for (let j = i + 1; j < S; j++) scores[i * S + j] = -Infinity;

                const attn = softmax(scores, S, S);
                attnWeights.push(attn);

                const outH = matmul(attn, Vh, S, S, dHead);
                scatterHead(attnConcat, outH, S, dModel, h, dHead);
            }
            lc.attnWeights = attnWeights;
            lc.attnConcat = attnConcat;

            // Output projection + residual
            let attnOut = matmul(attnConcat, this.p(`b${l}.attn.Wo`), S, dModel, dModel);
            addBias(attnOut, this.p(`b${l}.attn.bo`), S, dModel);
            x = new Float32Array(S * dModel);
            for (let i = 0; i < S * dModel; i++) x[i] = lc.xIn[i] + attnOut[i];

            lc.xMid = x.slice();

            // Layer norm 2
            const ln2 = layerNormForward(x, this.p(`b${l}.ln2.g`), this.p(`b${l}.ln2.b`), S, dModel);
            lc.ln2 = ln2.cache;
            lc.ln2Out = ln2.out;

            // FFN
            let h1 = matmul(ln2.out, this.p(`b${l}.ffn.W1`), S, dModel, dFF);
            addBias(h1, this.p(`b${l}.ffn.b1`), S, dFF);
            lc.ffnPre = h1.slice();
            const h1Act = geluForward(h1, S * dFF);
            lc.ffnPost = h1Act;

            let h2 = matmul(h1Act, this.p(`b${l}.ffn.W2`), S, dFF, dModel);
            addBias(h2, this.p(`b${l}.ffn.b2`), S, dModel);

            // Residual
            x = new Float32Array(S * dModel);
            for (let i = 0; i < S * dModel; i++) x[i] = lc.xMid[i] + h2[i];

            cache.layers.push(lc);
        }

        // Final layer norm
        cache.xPreFinal = x.slice();
        const lnF = layerNormForward(x, this.p('ln_f.g'), this.p('ln_f.b'), S, dModel);
        cache.lnF = lnF.cache;
        cache.xFinal = lnF.out;

        // Output projection
        const logits = matmul(lnF.out, this.p('W_out'), S, dModel, vocabSize);
        addBias(logits, this.p('b_out'), S, vocabSize);

        return { logits, cache };
    }

    lossAndGrad(logits, targets, S) {
        const V = this.config.vocabSize;
        let totalLoss = 0;
        const dLogits = new Float32Array(S * V);

        for (let i = 0; i < S; i++) {
            const off = i * V;
            let maxVal = -Infinity;
            for (let j = 0; j < V; j++) maxVal = Math.max(maxVal, logits[off + j]);
            let sumExp = 0;
            for (let j = 0; j < V; j++) sumExp += Math.exp(logits[off + j] - maxVal);

            for (let j = 0; j < V; j++) {
                const p = Math.exp(logits[off + j] - maxVal) / sumExp;
                const target = (j === targets[i]) ? 1 : 0;
                totalLoss -= target * Math.log(p + 1e-10);
                dLogits[off + j] = (p - target) / S;
            }
        }
        return { loss: totalLoss / S, dLogits };
    }

    backward(cache, dLogits) {
        const { S } = cache;
        const { dModel, nHeads, nLayers, dFF, vocabSize } = this.config;
        const dHead = dModel / nHeads;
        const scale = 1 / Math.sqrt(dHead);

        this.grads.fill(0);

        // Output projection gradients
        this.g('W_out').set(matmul(cache.xFinal, dLogits, dModel, S, vocabSize, true));
        colSum(dLogits, this.g('b_out'), S, vocabSize);
        let dx = matmul(dLogits, this.p('W_out'), S, vocabSize, dModel, false, true);

        // Final layer norm
        dx = layerNormBackward(dx, cache.lnF, this.p('ln_f.g'), this.g('ln_f.g'), this.g('ln_f.b'));

        // Blocks in reverse
        for (let l = nLayers - 1; l >= 0; l--) {
            const lc = cache.layers[l];

            // --- FFN residual ---
            const dxMid = dx.slice();

            // FFN backward: h2 = gelu(ln2Out @ W1 + b1) @ W2 + b2
            this.g(`b${l}.ffn.W2`).set(matmul(lc.ffnPost, dx, dFF, S, dModel, true));
            colSum(dx, this.g(`b${l}.ffn.b2`), S, dModel);
            let dh1 = matmul(dx, this.p(`b${l}.ffn.W2`), S, dModel, dFF, false, true);

            dh1 = geluBackward(dh1, lc.ffnPre, S * dFF);

            this.g(`b${l}.ffn.W1`).set(matmul(lc.ln2Out, dh1, dModel, S, dFF, true));
            colSum(dh1, this.g(`b${l}.ffn.b1`), S, dFF);
            let dln2 = matmul(dh1, this.p(`b${l}.ffn.W1`), S, dFF, dModel, false, true);

            dln2 = layerNormBackward(dln2, lc.ln2, this.p(`b${l}.ln2.g`), this.g(`b${l}.ln2.g`), this.g(`b${l}.ln2.b`));

            dx = new Float32Array(S * dModel);
            for (let i = 0; i < S * dModel; i++) dx[i] = dxMid[i] + dln2[i];

            // --- Attention residual ---
            const dxIn = dx.slice();

            // Output projection
            this.g(`b${l}.attn.Wo`).set(matmul(lc.attnConcat, dx, dModel, S, dModel, true));
            colSum(dx, this.g(`b${l}.attn.bo`), S, dModel);
            let dConcat = matmul(dx, this.p(`b${l}.attn.Wo`), S, dModel, dModel, false, true);

            // Per-head attention backward
            const dQ = new Float32Array(S * dModel);
            const dK = new Float32Array(S * dModel);
            const dV = new Float32Array(S * dModel);

            for (let h = 0; h < nHeads; h++) {
                const dOutH = extractHead(dConcat, S, dModel, h, dHead);
                const Vh = extractHead(lc.V, S, dModel, h, dHead);
                const Qh = extractHead(lc.Q, S, dModel, h, dHead);
                const Kh = extractHead(lc.K, S, dModel, h, dHead);
                const attn = lc.attnWeights[h];

                // d(attn @ V)
                const dAttn = matmul(dOutH, Vh, S, dHead, S, false, true);
                const dVh = matmul(attn, dOutH, S, S, dHead, true);

                // Softmax backward
                const dScores = new Float32Array(S * S);
                for (let i = 0; i < S; i++) {
                    const rowOff = i * S;
                    let dot = 0;
                    for (let j = 0; j < S; j++) dot += dAttn[rowOff + j] * attn[rowOff + j];
                    for (let j = 0; j < S; j++)
                        dScores[rowOff + j] = attn[rowOff + j] * (dAttn[rowOff + j] - dot) * scale;
                }

                // d(Q @ K^T)
                const dQh = matmul(dScores, Kh, S, S, dHead);
                const dKh = matmul(dScores, Qh, S, S, dHead, true);

                scatterHead(dQ, dQh, S, dModel, h, dHead);
                scatterHead(dK, dKh, S, dModel, h, dHead);
                scatterHead(dV, dVh, S, dModel, h, dHead);
            }

            // QKV projection gradients
            this.g(`b${l}.attn.Wq`).set(matmul(lc.lnOut, dQ, dModel, S, dModel, true));
            colSum(dQ, this.g(`b${l}.attn.bq`), S, dModel);
            this.g(`b${l}.attn.Wk`).set(matmul(lc.lnOut, dK, dModel, S, dModel, true));
            colSum(dK, this.g(`b${l}.attn.bk`), S, dModel);
            this.g(`b${l}.attn.Wv`).set(matmul(lc.lnOut, dV, dModel, S, dModel, true));
            colSum(dV, this.g(`b${l}.attn.bv`), S, dModel);

            let dln1 = matmul(dQ, this.p(`b${l}.attn.Wq`), S, dModel, dModel, false, true);
            const dln1K = matmul(dK, this.p(`b${l}.attn.Wk`), S, dModel, dModel, false, true);
            const dln1V = matmul(dV, this.p(`b${l}.attn.Wv`), S, dModel, dModel, false, true);
            for (let i = 0; i < S * dModel; i++) dln1[i] += dln1K[i] + dln1V[i];

            dln1 = layerNormBackward(dln1, lc.ln1, this.p(`b${l}.ln1.g`), this.g(`b${l}.ln1.g`), this.g(`b${l}.ln1.b`));

            dx = new Float32Array(S * dModel);
            for (let i = 0; i < S * dModel; i++) dx[i] = dxIn[i] + dln1[i];
        }

        // Embedding gradients
        const dTokEmb = this.g('tok_emb'), dPosEmb = this.g('pos_emb');
        const { dModel: D } = this.config;
        for (let i = 0; i < S; i++) {
            const tok = cache.tokenIds[i], xOff = i * D;
            const tOff = tok * D, pOff = i * D;
            for (let j = 0; j < D; j++) {
                dTokEmb[tOff + j] += dx[xOff + j];
                dPosEmb[pOff + j] += dx[xOff + j];
            }
        }
    }

    generate(tokenIds, maxNew, temperature) {
        temperature = temperature || 0.8;
        const { maxSeqLen, vocabSize } = this.config;
        const tokens = Array.from(tokenIds);

        for (let t = 0; t < maxNew && tokens.length < maxSeqLen; t++) {
            const { logits } = this.forward(new Uint16Array(tokens));
            const S = tokens.length;
            const lastOff = (S - 1) * vocabSize;
            const scaled = new Float32Array(vocabSize);
            for (let i = 0; i < vocabSize; i++) scaled[i] = logits[lastOff + i] / temperature;

            const probs = softmax(scaled, 1, vocabSize);
            let r = Math.random(), cumsum = 0;
            for (let i = 0; i < vocabSize; i++) {
                cumsum += probs[i];
                if (r < cumsum) { tokens.push(i); break; }
            }
            if (tokens.length === S) tokens.push(vocabSize - 1);
        }
        return new Uint16Array(tokens);
    }
}

// ===== AdamW optimizer =====

class AdamW {
    constructor(transformer, opts) {
        opts = opts || {};
        this.transformer = transformer;
        this.lr = opts.lr || 3e-4;
        this.beta1 = opts.beta1 || 0.9;
        this.beta2 = opts.beta2 || 0.999;
        this.eps = opts.eps || 1e-8;
        this.weightDecay = opts.weightDecay || 0.01;
        this.m = new Float32Array(transformer.totalParams);
        this.v = new Float32Array(transformer.totalParams);
        this.t = 0;
    }

    step() {
        this.t++;
        const { lr, beta1, beta2, eps, weightDecay } = this;
        const bc1 = 1 - Math.pow(beta1, this.t);
        const bc2 = 1 - Math.pow(beta2, this.t);
        const params = this.transformer.params;
        const grads = this.transformer.grads;

        for (const def of this.transformer.paramDefs) {
            const { offset, size, shape } = def;
            const wd = shape.length > 1 ? weightDecay : 0;
            for (let i = offset; i < offset + size; i++) {
                const g = grads[i];
                this.m[i] = beta1 * this.m[i] + (1 - beta1) * g;
                this.v[i] = beta2 * this.v[i] + (1 - beta2) * g * g;
                const mHat = this.m[i] / bc1;
                const vHat = this.v[i] / bc2;
                params[i] -= lr * (mHat / (Math.sqrt(vHat) + eps) + wd * params[i]);
            }
        }
    }
}

if (typeof module !== 'undefined') module.exports = { Transformer, AdamW, matmul, cpuMatmul, setMatmulImpl, layerNormForward, layerNormBackward, softmax, geluForward, geluBackward, randn };
