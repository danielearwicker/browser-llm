'use strict';
// Offline test of the GPU matmul shader logic.
// Since we can't run WebGL in Node, we simulate the fragment shader in JS
// to verify the texelFetch coordinate math matches the CPU matmul.

const { cpuMatmul } = require('./transformer.js');

function simulateGPUMatmul(A, B, M, K, N, transA, transB) {
    // Simulate the fragment shader:
    // A stored as texture: width = aCols, height = aRows
    // texelFetch(uA, ivec2(x, y)) => A[y * aCols + x]
    const aRows = transA ? K : M;
    const aCols = transA ? M : K;
    const bRows = transB ? N : K;
    const bCols = transB ? K : N;

    function texA(x, y) { return A[y * aCols + x]; }
    function texB(x, y) { return B[y * bCols + x]; }

    const C = new Float32Array(M * N);
    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            let s = 0;
            if (!transA && !transB) {
                for (let k = 0; k < K; k++) s += texA(k, i) * texB(j, k);
            } else if (transA && !transB) {
                for (let k = 0; k < K; k++) s += texA(i, k) * texB(j, k);
            } else if (!transA && transB) {
                for (let k = 0; k < K; k++) s += texA(k, i) * texB(k, j);
            } else {
                for (let k = 0; k < K; k++) s += texA(i, k) * texB(k, j);
            }
            C[i * N + j] = s;
        }
    }
    return C;
}

function randArr(n) {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = Math.random() * 2 - 1;
    return a;
}

function maxErr(a, b) {
    let m = 0;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
    return m;
}

console.log('Testing GPU matmul shader logic against CPU...');
const cases = [
    { M: 4, K: 3, N: 5, tA: false, tB: false, label: 'NN' },
    { M: 4, K: 3, N: 5, tA: true,  tB: false, label: 'TN' },
    { M: 4, K: 3, N: 5, tA: false, tB: true,  label: 'NT' },
    { M: 4, K: 3, N: 5, tA: true,  tB: true,  label: 'TT' },
    { M: 32, K: 16, N: 64, tA: false, tB: false, label: 'NN large' },
    { M: 32, K: 16, N: 64, tA: true,  tB: false, label: 'TN large' },
    { M: 32, K: 16, N: 64, tA: false, tB: true,  label: 'NT large' },
];

let pass = true;
for (const { M, K, N, tA, tB, label } of cases) {
    const aRows = tA ? K : M, aCols = tA ? M : K;
    const bRows = tB ? N : K, bCols = tB ? K : N;
    const A = randArr(aRows * aCols);
    const B = randArr(bRows * bCols);

    const cpu = cpuMatmul(A, B, M, K, N, tA, tB);
    const gpu = simulateGPUMatmul(A, B, M, K, N, tA, tB);
    const err = maxErr(cpu, gpu);

    if (err > 1e-5) {
        console.log(`  FAIL ${label}: maxErr=${err}`);
        pass = false;
    } else {
        console.log(`  OK ${label}: maxErr=${err.toExponential(2)}`);
    }
}

console.log(pass ? '\nAll shader logic tests passed!' : '\nSome tests FAILED!');
if (!pass) process.exit(1);
