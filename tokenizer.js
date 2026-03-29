'use strict';

class CharTokenizer {
    constructor() {
        this.vocabSize = 128; // ASCII
    }
    encode(str) {
        const ids = new Uint16Array(str.length);
        for (let i = 0; i < str.length; i++) {
            const c = str.charCodeAt(i);
            ids[i] = c < 128 ? c : 63; // '?' for non-ASCII
        }
        return ids;
    }
    decode(ids) {
        let s = '';
        for (let i = 0; i < ids.length; i++) s += String.fromCharCode(ids[i]);
        return s;
    }
}

if (typeof module !== 'undefined') module.exports = { CharTokenizer };
