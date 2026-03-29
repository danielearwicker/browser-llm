'use strict';

// GPU tensor operations via WebGL2 fragment shaders.
// All data stays as R32F textures on the GPU between operations.

const G = `#version 300 es
precision highp float;
precision highp int;
`;

const SRC_MATMUL = G + `uniform sampler2D uA,uB;uniform int uK,uTA,uTB;out vec4 fc;
void main(){int i=int(gl_FragCoord.y),j=int(gl_FragCoord.x);float s=0.0;
if(uTA==0&&uTB==0){for(int k=0;k<uK;k++)s+=texelFetch(uA,ivec2(k,i),0).r*texelFetch(uB,ivec2(j,k),0).r;}
else if(uTA!=0&&uTB==0){for(int k=0;k<uK;k++)s+=texelFetch(uA,ivec2(i,k),0).r*texelFetch(uB,ivec2(j,k),0).r;}
else if(uTA==0){for(int k=0;k<uK;k++)s+=texelFetch(uA,ivec2(k,i),0).r*texelFetch(uB,ivec2(k,j),0).r;}
else{for(int k=0;k<uK;k++)s+=texelFetch(uA,ivec2(i,k),0).r*texelFetch(uB,ivec2(k,j),0).r;}
fc=vec4(s,0,0,1);}`;

const SRC_ADD = G + `uniform sampler2D uA,uB;out vec4 fc;
void main(){ivec2 p=ivec2(gl_FragCoord.xy);fc=vec4(texelFetch(uA,p,0).r+texelFetch(uB,p,0).r,0,0,1);}`;

const SRC_ADD_BIAS = G + `uniform sampler2D uA,uBias;out vec4 fc;
void main(){ivec2 p=ivec2(gl_FragCoord.xy);fc=vec4(texelFetch(uA,p,0).r+texelFetch(uBias,ivec2(p.x,0),0).r,0,0,1);}`;

const SRC_GELU = G + `uniform sampler2D uX;out vec4 fc;
void main(){float x=texelFetch(uX,ivec2(gl_FragCoord.xy),0).r;
float s=0.7978845608*(x+0.044715*x*x*x);fc=vec4(0.5*x*(1.0+tanh(s)),0,0,1);}`;

const SRC_GELU_BACK = G + `uniform sampler2D uD,uX;out vec4 fc;
void main(){ivec2 p=ivec2(gl_FragCoord.xy);float x=texelFetch(uX,p,0).r,d=texelFetch(uD,p,0).r;
float s=0.7978845608*(x+0.044715*x*x*x),ts=tanh(s),ds=0.7978845608*(1.0+3.0*0.044715*x*x);
fc=vec4(d*(0.5*(1.0+ts)+0.5*x*(1.0-ts*ts)*ds),0,0,1);}`;

const SRC_SOFTMAX_CAUSAL = G + `uniform sampler2D uX;uniform float uScale;out vec4 fc;
void main(){int i=int(gl_FragCoord.y),j=int(gl_FragCoord.x);
if(j>i){fc=vec4(0,0,0,1);return;}
float mx=-1.0e30;for(int k=0;k<=i;k++)mx=max(mx,texelFetch(uX,ivec2(k,i),0).r*uScale);
float se=0.0;for(int k=0;k<=i;k++)se+=exp(texelFetch(uX,ivec2(k,i),0).r*uScale-mx);
fc=vec4(exp(texelFetch(uX,ivec2(j,i),0).r*uScale-mx)/se,0,0,1);}`;

const SRC_SOFTMAX_BACK = G + `uniform sampler2D uDA,uA;uniform int uS;uniform float uScale;out vec4 fc;
void main(){int i=int(gl_FragCoord.y),j=int(gl_FragCoord.x);
float dot=0.0;for(int k=0;k<uS;k++)dot+=texelFetch(uDA,ivec2(k,i),0).r*texelFetch(uA,ivec2(k,i),0).r;
float a=texelFetch(uA,ivec2(j,i),0).r,da=texelFetch(uDA,ivec2(j,i),0).r;
fc=vec4(a*(da-dot)*uScale,0,0,1);}`;

const SRC_LN_STATS = G + `uniform sampler2D uX;uniform int uD;out vec4 fc;
void main(){int i=int(gl_FragCoord.y),c=int(gl_FragCoord.x);
float m=0.0;for(int k=0;k<uD;k++)m+=texelFetch(uX,ivec2(k,i),0).r;m/=float(uD);
if(c==0){fc=vec4(m,0,0,1);return;}
float v=0.0;for(int k=0;k<uD;k++){float d=texelFetch(uX,ivec2(k,i),0).r-m;v+=d*d;}
fc=vec4(inversesqrt(v/float(uD)+1e-5),0,0,1);}`;

const SRC_LN_APPLY = G + `uniform sampler2D uX,uSt,uG,uB;
layout(location=0) out vec4 o0;layout(location=1) out vec4 o1;
void main(){int i=int(gl_FragCoord.y),j=int(gl_FragCoord.x);
float mn=texelFetch(uSt,ivec2(0,i),0).r,rs=texelFetch(uSt,ivec2(1,i),0).r;
float xn=(texelFetch(uX,ivec2(j,i),0).r-mn)*rs;
o0=vec4(texelFetch(uG,ivec2(j,0),0).r*xn+texelFetch(uB,ivec2(j,0),0).r,0,0,1);
o1=vec4(xn,0,0,1);}`;

const SRC_LN_BACK_DX = G + `uniform sampler2D uDO,uXN,uSt,uG;uniform int uD;out vec4 fc;
void main(){int i=int(gl_FragCoord.y),j=int(gl_FragCoord.x);
float rs=texelFetch(uSt,ivec2(1,i),0).r,s1=0.0,s2=0.0;
for(int k=0;k<uD;k++){float dn=texelFetch(uDO,ivec2(k,i),0).r*texelFetch(uG,ivec2(k,0),0).r;
float xn=texelFetch(uXN,ivec2(k,i),0).r;s1+=dn;s2+=dn*xn;}
s1/=float(uD);s2/=float(uD);
float dn=texelFetch(uDO,ivec2(j,i),0).r*texelFetch(uG,ivec2(j,0),0).r;
float xn=texelFetch(uXN,ivec2(j,i),0).r;
fc=vec4(rs*(dn-s1-xn*s2),0,0,1);}`;

const SRC_LN_BACK_P = G + `uniform sampler2D uDO,uXN;uniform int uS;
layout(location=0) out vec4 o0;layout(location=1) out vec4 o1;
void main(){int j=int(gl_FragCoord.x);float dg=0.0,db=0.0;
for(int i=0;i<uS;i++){float d=texelFetch(uDO,ivec2(j,i),0).r;
dg+=d*texelFetch(uXN,ivec2(j,i),0).r;db+=d;}
o0=vec4(dg,0,0,1);o1=vec4(db,0,0,1);}`;

const SRC_COL_SUM = G + `uniform sampler2D uX;uniform int uS;out vec4 fc;
void main(){int j=int(gl_FragCoord.x);float s=0.0;
for(int i=0;i<uS;i++)s+=texelFetch(uX,ivec2(j,i),0).r;fc=vec4(s,0,0,1);}`;

const SRC_EMBED = G + `uniform sampler2D uTok,uTE,uPE;out vec4 fc;
void main(){int i=int(gl_FragCoord.y),j=int(gl_FragCoord.x);
int t=int(texelFetch(uTok,ivec2(i,0),0).r);
fc=vec4(texelFetch(uTE,ivec2(j,t),0).r+texelFetch(uPE,ivec2(j,i),0).r,0,0,1);}`;

const SRC_EMBED_BACK_TOK = G + `uniform sampler2D uDx,uTok;uniform int uS;out vec4 fc;
void main(){int tok=int(gl_FragCoord.y),j=int(gl_FragCoord.x);float s=0.0;
for(int i=0;i<uS;i++){if(int(texelFetch(uTok,ivec2(i,0),0).r)==tok)
s+=texelFetch(uDx,ivec2(j,i),0).r;}fc=vec4(s,0,0,1);}`;

const SRC_EMBED_BACK_POS = G + `uniform sampler2D uDx;uniform int uS;out vec4 fc;
void main(){int i=int(gl_FragCoord.y),j=int(gl_FragCoord.x);
fc=vec4(i<uS?texelFetch(uDx,ivec2(j,i),0).r:0.0,0,0,1);}`;

const SRC_CE_GRAD = G + `uniform sampler2D uL,uT;uniform int uV,uS;out vec4 fc;
void main(){int i=int(gl_FragCoord.y),j=int(gl_FragCoord.x);
int tgt=int(texelFetch(uT,ivec2(i,0),0).r);
float mx=-1.0e30;for(int k=0;k<uV;k++)mx=max(mx,texelFetch(uL,ivec2(k,i),0).r);
float se=0.0;for(int k=0;k<uV;k++)se+=exp(texelFetch(uL,ivec2(k,i),0).r-mx);
float p=exp(texelFetch(uL,ivec2(j,i),0).r-mx)/se;
fc=vec4((p-(j==tgt?1.0:0.0))/float(uS),0,0,1);}`;

const SRC_CE_LOSS = G + `uniform sampler2D uL,uT;uniform int uV;out vec4 fc;
void main(){int i=int(gl_FragCoord.y);
int tgt=int(texelFetch(uT,ivec2(i,0),0).r);
float mx=-1.0e30;for(int k=0;k<uV;k++)mx=max(mx,texelFetch(uL,ivec2(k,i),0).r);
float se=0.0;for(int k=0;k<uV;k++)se+=exp(texelFetch(uL,ivec2(k,i),0).r-mx);
float p=exp(texelFetch(uL,ivec2(tgt,i),0).r-mx)/se;
fc=vec4(-log(max(p,1.0e-10)),0,0,1);}`;

const SRC_ADAM = G + `uniform sampler2D uP,uGr,uM,uV;
uniform float uLR,uB1,uB2,uBC1,uBC2,uEPS,uWD;
layout(location=0) out vec4 oP;layout(location=1) out vec4 oM;layout(location=2) out vec4 oV;
void main(){ivec2 p=ivec2(gl_FragCoord.xy);
float pr=texelFetch(uP,p,0).r,g=texelFetch(uGr,p,0).r;
float m=uB1*texelFetch(uM,p,0).r+(1.0-uB1)*g;
float v=uB2*texelFetch(uV,p,0).r+(1.0-uB2)*g*g;
pr-=uLR*(m/uBC1/(sqrt(v/uBC2)+uEPS)+uWD*pr);
oP=vec4(pr,0,0,1);oM=vec4(m,0,0,1);oV=vec4(v,0,0,1);}`;

const SRC_EXTRACT = G + `uniform sampler2D uX;uniform int uOff;out vec4 fc;
void main(){int i=int(gl_FragCoord.y),j=int(gl_FragCoord.x);
fc=vec4(texelFetch(uX,ivec2(uOff+j,i),0).r,0,0,1);}`;

const SRC_SCATTER = G + `uniform sampler2D uAcc,uSrc;uniform int uOff,uDH;out vec4 fc;
void main(){int i=int(gl_FragCoord.y),j=int(gl_FragCoord.x);
float v=texelFetch(uAcc,ivec2(j,i),0).r;
if(j>=uOff&&j<uOff+uDH)v+=texelFetch(uSrc,ivec2(j-uOff,i),0).r;
fc=vec4(v,0,0,1);}`;

class WebGLOps {
    constructor(gl) {
        this.gl = gl;
        if (!gl.getExtension('EXT_color_buffer_float'))
            throw new Error('EXT_color_buffer_float required');
        this._pool = {};
        this._cache = new Map();
        this._mrtFB = gl.createFramebuffer();
        this.vao = gl.createVertexArray();

        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, `#version 300 es
            void main(){gl_Position=vec4(gl_VertexID==1?3.:-1.,gl_VertexID==2?3.:-1.,0,1);}`);
        gl.compileShader(vs);
        this._vs = vs;
    }

    // --- Tensor management ---

    alloc(rows, cols) {
        const key = rows + ',' + cols;
        const list = this._pool[key];
        if (list && list.length) return list.pop();
        const gl = this.gl, tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, cols, rows, 0, gl.RED, gl.FLOAT, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        const fb = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return { tex, fb, rows, cols };
    }

    fromData(data, rows, cols) {
        const t = this.alloc(rows, cols);
        this.gl.bindTexture(this.gl.TEXTURE_2D, t.tex);
        this.gl.texSubImage2D(this.gl.TEXTURE_2D, 0, 0, 0, cols, rows, this.gl.RED, this.gl.FLOAT, data);
        return t;
    }

    read(t) {
        const gl = this.gl, d = new Float32Array(t.rows * t.cols);
        gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
        gl.readPixels(0, 0, t.cols, t.rows, gl.RED, gl.FLOAT, d);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return d;
    }

    free(t) {
        if (!t) return;
        const key = t.rows + ',' + t.cols;
        (this._pool[key] || (this._pool[key] = [])).push(t);
    }

    zeros(rows, cols) {
        const t = this.alloc(rows, cols), gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
        gl.viewport(0, 0, cols, rows);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return t;
    }

    // --- Shader helpers ---

    _prog(src) {
        if (this._cache.has(src)) return this._cache.get(src);
        const gl = this.gl, fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, src); gl.compileShader(fs);
        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS))
            throw new Error('FS: ' + gl.getShaderInfoLog(fs));
        const prog = gl.createProgram();
        gl.attachShader(prog, this._vs); gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
            throw new Error('Link: ' + gl.getProgramInfoLog(prog));
        const locs = {};
        const info = { prog, ul: n => (n in locs) ? locs[n] : (locs[n] = gl.getUniformLocation(prog, n)) };
        this._cache.set(src, info);
        return info;
    }

    _run(src, oR, oC, tex, ints, floats) {
        const p = this._prog(src), gl = this.gl, out = this.alloc(oR, oC);
        gl.useProgram(p.prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, out.fb);
        gl.viewport(0, 0, oC, oR);
        let u = 0;
        for (const [n, t] of tex) {
            gl.activeTexture(gl.TEXTURE0 + u);
            gl.bindTexture(gl.TEXTURE_2D, t.tex);
            gl.uniform1i(p.ul(n), u++);
        }
        if (ints) for (const [n, v] of ints) gl.uniform1i(p.ul(n), v);
        if (floats) for (const [n, v] of floats) gl.uniform1f(p.ul(n), v);
        gl.bindVertexArray(this.vao);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return out;
    }

    _runMRT(src, outs, tex, ints, floats) {
        const p = this._prog(src), gl = this.gl;
        gl.useProgram(p.prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this._mrtFB);
        const bufs = [];
        for (let i = 0; i < outs.length; i++) {
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, outs[i].tex, 0);
            bufs.push(gl.COLOR_ATTACHMENT0 + i);
        }
        gl.drawBuffers(bufs);
        gl.viewport(0, 0, outs[0].cols, outs[0].rows);
        let u = 0;
        for (const [n, t] of tex) {
            gl.activeTexture(gl.TEXTURE0 + u);
            gl.bindTexture(gl.TEXTURE_2D, t.tex);
            gl.uniform1i(p.ul(n), u++);
        }
        if (ints) for (const [n, v] of ints) gl.uniform1i(p.ul(n), v);
        if (floats) for (const [n, v] of floats) gl.uniform1f(p.ul(n), v);
        gl.bindVertexArray(this.vao);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        for (let i = 0; i < outs.length; i++)
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, null, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    // --- Operations ---

    matmul(a, b, M, K, N, tA, tB) {
        return this._run(SRC_MATMUL, M, N,
            [['uA', a], ['uB', b]],
            [['uK', K], ['uTA', tA ? 1 : 0], ['uTB', tB ? 1 : 0]]);
    }

    add(a, b) {
        return this._run(SRC_ADD, a.rows, a.cols, [['uA', a], ['uB', b]]);
    }

    addBias(a, bias) {
        return this._run(SRC_ADD_BIAS, a.rows, a.cols, [['uA', a], ['uBias', bias]]);
    }

    matmulBias(a, w, bias, M, K, N, tA, tB) {
        const mm = this.matmul(a, w, M, K, N, tA, tB);
        const out = this.addBias(mm, bias);
        this.free(mm);
        return out;
    }

    gelu(x) {
        return this._run(SRC_GELU, x.rows, x.cols, [['uX', x]]);
    }

    geluBack(dout, x) {
        return this._run(SRC_GELU_BACK, x.rows, x.cols, [['uD', dout], ['uX', x]]);
    }

    softmaxCausal(scores, scale) {
        return this._run(SRC_SOFTMAX_CAUSAL, scores.rows, scores.cols,
            [['uX', scores]], null, [['uScale', scale]]);
    }

    softmaxBack(dAttn, attn, scale) {
        return this._run(SRC_SOFTMAX_BACK, attn.rows, attn.cols,
            [['uDA', dAttn], ['uA', attn]],
            [['uS', attn.cols]], [['uScale', scale]]);
    }

    layerNorm(x, gamma, beta) {
        const S = x.rows, D = x.cols;
        const stats = this._run(SRC_LN_STATS, S, 2, [['uX', x]], [['uD', D]]);
        const out = this.alloc(S, D), xNorm = this.alloc(S, D);
        this._runMRT(SRC_LN_APPLY, [out, xNorm],
            [['uX', x], ['uSt', stats], ['uG', gamma], ['uB', beta]]);
        return { out, xNorm, stats };
    }

    layerNormBackDx(dout, xNorm, stats, gamma) {
        return this._run(SRC_LN_BACK_DX, dout.rows, dout.cols,
            [['uDO', dout], ['uXN', xNorm], ['uSt', stats], ['uG', gamma]],
            [['uD', dout.cols]]);
    }

    layerNormBackParams(dout, xNorm) {
        const D = dout.cols, dg = this.alloc(1, D), db = this.alloc(1, D);
        this._runMRT(SRC_LN_BACK_P, [dg, db],
            [['uDO', dout], ['uXN', xNorm]], [['uS', dout.rows]]);
        return { dgamma: dg, dbeta: db };
    }

    colSum(a) {
        return this._run(SRC_COL_SUM, 1, a.cols, [['uX', a]], [['uS', a.rows]]);
    }

    embed(tokenIds, tokEmb, posEmb) {
        const S = tokenIds.cols, D = tokEmb.cols;
        return this._run(SRC_EMBED, S, D,
            [['uTok', tokenIds], ['uTE', tokEmb], ['uPE', posEmb]]);
    }

    embedBackTok(dx, tokenIds, vocabSize) {
        return this._run(SRC_EMBED_BACK_TOK, vocabSize, dx.cols,
            [['uDx', dx], ['uTok', tokenIds]], [['uS', dx.rows]]);
    }

    embedBackPos(dx, maxSeqLen) {
        return this._run(SRC_EMBED_BACK_POS, maxSeqLen, dx.cols,
            [['uDx', dx]], [['uS', dx.rows]]);
    }

    ceGrad(logits, targets) {
        return this._run(SRC_CE_GRAD, logits.rows, logits.cols,
            [['uL', logits], ['uT', targets]],
            [['uV', logits.cols], ['uS', logits.rows]]);
    }

    ceLoss(logits, targets) {
        return this._run(SRC_CE_LOSS, logits.rows, 1,
            [['uL', logits], ['uT', targets]], [['uV', logits.cols]]);
    }

    adamW(param, grad, m, v, lr, b1, b2, bc1, bc2, eps, wd) {
        const pOut = this.alloc(param.rows, param.cols);
        const mOut = this.alloc(param.rows, param.cols);
        const vOut = this.alloc(param.rows, param.cols);
        this._runMRT(SRC_ADAM, [pOut, mOut, vOut],
            [['uP', param], ['uGr', grad], ['uM', m], ['uV', v]],
            null,
            [['uLR', lr], ['uB1', b1], ['uB2', b2], ['uBC1', bc1], ['uBC2', bc2], ['uEPS', eps], ['uWD', wd]]);
        return { param: pOut, m: mOut, v: vOut };
    }

    extractHead(x, h, dHead) {
        return this._run(SRC_EXTRACT, x.rows, dHead, [['uX', x]], [['uOff', h * dHead]]);
    }

    scatterHead(accum, src, h, dHead) {
        return this._run(SRC_SCATTER, accum.rows, accum.cols,
            [['uAcc', accum], ['uSrc', src]],
            [['uOff', h * dHead], ['uDH', dHead]]);
    }
}

if (typeof module !== 'undefined') module.exports = { WebGLOps };
