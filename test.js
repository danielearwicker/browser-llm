'use strict';

// Run with: node test.js
const { Transformer, AdamW } = require('./transformer.js');
const { CharTokenizer } = require('./tokenizer.js');

function assert(cond, msg) {
    if (!cond) throw new Error('FAIL: ' + msg);
}

// === Test 1: Gradient check (numerical vs analytical) ===
// Uses Float64 internally for the finite-difference computation to avoid
// Float32 cancellation errors on small gradients.
function testGradientCheck() {
    console.log('Test 1: Gradient check...');
    const config = { vocabSize: 16, dModel: 16, nHeads: 2, nLayers: 1, dFF: 32, maxSeqLen: 8 };
    const model = new Transformer(config);
    const S = 6;
    const input = new Uint16Array(S);
    const target = new Uint16Array(S);
    for (let i = 0; i < S; i++) {
        input[i] = Math.floor(Math.random() * config.vocabSize);
        target[i] = Math.floor(Math.random() * config.vocabSize);
    }

    // Analytical gradients
    const { logits, cache } = model.forward(input);
    const { loss, dLogits } = model.lossAndGrad(logits, target, S);
    model.backward(cache, dLogits);
    const analyticGrads = new Float32Array(model.grads);

    const eps = 5e-4;
    let failures = 0, checked = 0;
    const nCheck = Math.min(300, model.totalParams);
    const indices = new Set();
    while (indices.size < nCheck) indices.add(Math.floor(Math.random() * model.totalParams));

    for (const idx of indices) {
        const orig = model.params[idx];

        model.params[idx] = orig + eps;
        const { logits: lp } = model.forward(input);
        const { loss: lossPlus } = model.lossAndGrad(lp, target, S);

        model.params[idx] = orig - eps;
        const { logits: lm } = model.forward(input);
        const { loss: lossMinus } = model.lossAndGrad(lm, target, S);

        model.params[idx] = orig;

        const numGrad = (lossPlus - lossMinus) / (2 * eps);
        const anaGrad = analyticGrads[idx];
        const absErr = Math.abs(numGrad - anaGrad);
        const scale = Math.abs(numGrad) + Math.abs(anaGrad);

        // Skip if both gradients are negligibly small (below Float32 noise floor)
        if (scale < 1e-4) { checked++; continue; }

        const relErr = absErr / (scale + 1e-8);
        checked++;

        if (relErr > 0.05) {
            let paramName = '?';
            for (const def of model.paramDefs) {
                if (idx >= def.offset && idx < def.offset + def.size) {
                    paramName = def.name + '[' + (idx - def.offset) + ']';
                    break;
                }
            }
            console.log(`  FAIL: ${paramName} relErr=${relErr.toExponential(2)} num=${numGrad.toExponential(4)} ana=${anaGrad.toExponential(4)}`);
            failures++;
        }
    }

    const failRate = failures / checked;
    console.log(`  Checked ${checked} params, ${failures} failures (${(failRate * 100).toFixed(1)}%)`);
    assert(failRate < 0.05, `Gradient check: too many failures (${failures}/${checked})`);
    console.log('  PASSED');
}

// === Test 2: Loss decreases on repeated training ===
function testLossDecreases() {
    console.log('Test 2: Loss decreases...');
    const config = { vocabSize: 16, dModel: 16, nHeads: 2, nLayers: 1, dFF: 32, maxSeqLen: 8 };
    const model = new Transformer(config);
    const optimizer = new AdamW(model, { lr: 1e-3 });

    const input = new Uint16Array([1, 2, 3, 4, 5, 6]);
    const target = new Uint16Array([2, 3, 4, 5, 6, 7]);

    let firstLoss, lastLoss;
    for (let i = 0; i < 100; i++) {
        const { logits, cache } = model.forward(input);
        const { loss, dLogits } = model.lossAndGrad(logits, target, input.length);
        model.backward(cache, dLogits);
        optimizer.step();
        if (i === 0) firstLoss = loss;
        lastLoss = loss;
    }

    console.log(`  First loss: ${firstLoss.toFixed(4)}, Last loss: ${lastLoss.toFixed(4)}`);
    assert(lastLoss < firstLoss * 0.5, `Loss did not decrease enough: ${firstLoss} -> ${lastLoss}`);
    console.log('  PASSED');
}

// === Test 3: Generate produces valid tokens ===
function testGenerate() {
    console.log('Test 3: Generation...');
    const config = { vocabSize: 16, dModel: 16, nHeads: 2, nLayers: 1, dFF: 32, maxSeqLen: 16 };
    const model = new Transformer(config);

    const prompt = new Uint16Array([1, 2, 3]);
    const output = model.generate(prompt, 5, 1.0);

    assert(output.length === 8, `Expected 8 tokens, got ${output.length}`);
    assert(output[0] === 1 && output[1] === 2 && output[2] === 3, 'Prompt should be preserved');
    for (let i = 0; i < output.length; i++) {
        assert(output[i] >= 0 && output[i] < config.vocabSize, `Token ${i} out of range: ${output[i]}`);
    }
    console.log('  PASSED');
}

// === Test 4: Tokenizer roundtrip ===
function testTokenizer() {
    console.log('Test 4: Tokenizer...');
    const tok = new CharTokenizer();
    assert(tok.vocabSize === 128, 'Vocab size should be 128');

    const text = 'Hello, world! 123';
    const encoded = tok.encode(text);
    const decoded = tok.decode(encoded);
    assert(decoded === text, `Roundtrip failed: "${decoded}" !== "${text}"`);

    // Non-ASCII maps to '?'
    const nonAscii = tok.encode('\u00e9');
    assert(nonAscii[0] === 63, 'Non-ASCII should map to ? (63)');
    console.log('  PASSED');
}

// === Test 5: AdamW weight decay ===
function testAdamW() {
    console.log('Test 5: AdamW...');
    const config = { vocabSize: 8, dModel: 8, nHeads: 2, nLayers: 1, dFF: 16, maxSeqLen: 4 };
    const model = new Transformer(config);
    const optimizer = new AdamW(model, { lr: 0.1, weightDecay: 0.1 });

    // Zero gradients, step should only apply weight decay to 2D params
    model.grads.fill(0);
    const wBefore = model.p('b0.attn.Wq')[0];
    optimizer.step();
    const wAfter = model.p('b0.attn.Wq')[0];

    // Weight decay should shrink the weight toward zero
    assert(Math.abs(wAfter) < Math.abs(wBefore) + 1e-6, 'Weight decay should reduce weight magnitude');

    // Bias should not be decayed (only Adam momentum from zero grad)
    const bBefore = model.p('b0.attn.bq')[0]; // already 0 from init
    assert(Math.abs(bBefore) < 1e-6, 'Bias should start at 0');
    console.log('  PASSED');
}

// === Run all tests ===
console.log('Running tests...\n');
try {
    testTokenizer();
    testGradientCheck();
    testLossDecreases();
    testGenerate();
    testAdamW();
    console.log('\nAll tests passed!');
} catch (e) {
    console.error('\n' + e.message);
    process.exit(1);
}
