'use strict';
// Verify GPUTrainer paramDefs match the CPU Transformer paramDefs exactly.
const { Transformer } = require('./transformer.js');

// Simulate GPUTrainer paramDefs construction
function buildGPUParamDefs(config) {
    const { vocabSize, dModel, nHeads, nLayers, dFF, maxSeqLen } = config;
    const defs = [];
    const add = (name, rows, cols, wd) => defs.push({ name, rows, cols, size: rows * cols, wd: !!wd });
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
    let off = 0;
    for (const d of defs) { d.offset = off; off += d.size; }
    return { defs, total: off };
}

const config = { vocabSize: 128, dModel: 128, nHeads: 4, nLayers: 4, dFF: 512, maxSeqLen: 256 };
const cpu = new Transformer(config);
const gpu = buildGPUParamDefs(config);

console.log(`CPU total params: ${cpu.totalParams}`);
console.log(`GPU total params: ${gpu.total}`);
console.log(`Match: ${cpu.totalParams === gpu.total}`);

let ok = true;
for (let i = 0; i < cpu.paramDefs.length; i++) {
    const c = cpu.paramDefs[i], g = gpu.defs[i];
    if (c.name !== g.name || c.offset !== g.offset || c.size !== g.size) {
        console.log(`MISMATCH at ${i}: CPU=${c.name}@${c.offset}(${c.size}) GPU=${g.name}@${g.offset}(${g.size})`);
        ok = false;
    }
}
if (ok) console.log(`All ${cpu.paramDefs.length} parameter definitions match!`);
else process.exit(1);
