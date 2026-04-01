'use strict';

// === Character-level tokenizer (legacy / fallback) ===

class CharTokenizer {
    constructor() {
        this.vocabSize = 128;
        this.merges = [];
    }
    encode(str) {
        var ids = new Uint16Array(str.length);
        for (var i = 0; i < str.length; i++) {
            var c = str.charCodeAt(i);
            ids[i] = c < 128 ? c : 63;
        }
        return ids;
    }
    decode(ids) {
        var s = '';
        for (var i = 0; i < ids.length; i++) s += String.fromCharCode(ids[i]);
        return s;
    }
}

// === BPE tokenizer ===
// Starts from 128 ASCII base tokens, learns merge pairs from a corpus.
// merges: ordered array of [tokenA, tokenB] pairs.
// Token IDs 0-127 are ASCII. Token 128+i is the result of merges[i].

class BPETokenizer {
    constructor(merges) {
        this.merges = merges || [];
        this.vocabSize = 128 + this.merges.length;
        this._buildDecodeTable();
        this._buildMergeIndex();
    }

    _buildDecodeTable() {
        this._decode = new Array(this.vocabSize);
        for (var i = 0; i < 128; i++) this._decode[i] = String.fromCharCode(i);
        for (var i = 0; i < this.merges.length; i++) {
            this._decode[128 + i] = this._decode[this.merges[i][0]] + this._decode[this.merges[i][1]];
        }
    }

    _buildMergeIndex() {
        // Integer key: a * 65536 + b (both < 8000, so fits in safe integer)
        this._mergeMap = new Map();
        for (var i = 0; i < this.merges.length; i++) {
            var key = this.merges[i][0] * 65536 + this.merges[i][1];
            if (!this._mergeMap.has(key)) // keep first = highest priority
                this._mergeMap.set(key, 128 + i);
        }
    }

    // Train BPE from a Uint16Array of base (ASCII) token IDs.
    // Samples up to 2M tokens (plenty for learning good merges).
    static train(baseTokens, targetVocabSize, onProgress) {
        targetVocabSize = targetVocabSize || 2000;
        var numMerges = Math.max(0, targetVocabSize - 128);
        var merges = [];

        var maxSample = 2000000;
        var sampleLen = Math.min(baseTokens.length, maxSample);
        var tokens = new Uint16Array(sampleLen);
        for (var i = 0; i < sampleLen; i++) tokens[i] = baseTokens[i];
        var len = sampleLen;

        for (var m = 0; m < numMerges; m++) {
            if (len < 2) break;

            // Count adjacent pairs using integer keys
            var counts = new Map();
            var bestKey = -1, bestCount = 0;
            for (var i = 0; i < len - 1; i++) {
                var key = tokens[i] * 65536 + tokens[i + 1];
                var c = (counts.get(key) || 0) + 1;
                counts.set(key, c);
                if (c > bestCount) { bestCount = c; bestKey = key; }
            }

            if (bestCount < 2) break;

            var a = (bestKey >>> 16), b = (bestKey & 0xFFFF);
            var newId = 128 + m;
            merges.push([a, b]);

            var write = 0;
            for (var i = 0; i < len; i++) {
                if (i < len - 1 && tokens[i] === a && tokens[i + 1] === b) {
                    tokens[write++] = newId; i++;
                } else {
                    tokens[write++] = tokens[i];
                }
            }
            len = write;

            if (onProgress && m % 100 === 0) onProgress(m, numMerges);
        }

        return new BPETokenizer(merges);
    }

    // Greedy multi-merge encoding: applies all available merges per pass,
    // converging in ~10-15 rounds instead of one pass per merge.
    encode(str) {
        var n = str.length;
        if (n === 0) return new Uint16Array(0);

        var tokens = new Uint16Array(n);
        for (var i = 0; i < n; i++) {
            var c = str.charCodeAt(i);
            tokens[i] = c < 128 ? c : 63;
        }
        var len = n;
        var map = this._mergeMap;

        var changed = true;
        while (changed) {
            changed = false;
            var write = 0;
            for (var i = 0; i < len; i++) {
                if (i < len - 1) {
                    var newId = map.get(tokens[i] * 65536 + tokens[i + 1]);
                    if (newId !== undefined) {
                        tokens[write++] = newId; i++; changed = true; continue;
                    }
                }
                tokens[write++] = tokens[i];
            }
            len = write;
        }

        return new Uint16Array(tokens.buffer, 0, len);
    }

    // In-place greedy encoding for pre-tokenized data.
    // Mutates the input array. Returns a subarray view.
    encodePre(baseTokens) {
        var len = baseTokens.length;
        var map = this._mergeMap;

        var changed = true;
        while (changed) {
            changed = false;
            var write = 0;
            for (var i = 0; i < len; i++) {
                if (i < len - 1) {
                    var newId = map.get(baseTokens[i] * 65536 + baseTokens[i + 1]);
                    if (newId !== undefined) {
                        baseTokens[write++] = newId; i++; changed = true; continue;
                    }
                }
                baseTokens[write++] = baseTokens[i];
            }
            len = write;
        }
        return baseTokens.subarray(0, len);
    }

    decode(ids) {
        var s = '';
        for (var i = 0; i < ids.length; i++) {
            var id = ids[i];
            s += id < this.vocabSize ? this._decode[id] : '?';
        }
        return s;
    }
}

if (typeof module !== 'undefined') module.exports = { CharTokenizer, BPETokenizer };
