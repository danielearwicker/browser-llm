'use strict';

// GPU-accelerated matrix multiply via WebGL2 fragment shaders.
// Each fragment computes one element of C = op(A) @ op(B).
// Data is uploaded as R32F textures, computed on GPU, read back.

class WebGLMatMul {
    constructor(gl) {
        this.gl = gl;
        if (!gl.getExtension('EXT_color_buffer_float')) {
            throw new Error('EXT_color_buffer_float not available');
        }

        const vs = this._shader(gl.VERTEX_SHADER, `#version 300 es
            void main() {
                gl_Position = vec4(
                    gl_VertexID == 1 ? 3.0 : -1.0,
                    gl_VertexID == 2 ? 3.0 : -1.0, 0, 1);
            }`);

        const fs = this._shader(gl.FRAGMENT_SHADER, `#version 300 es
            precision highp float;
            precision highp int;
            uniform sampler2D uA, uB;
            uniform int uK, uTransA, uTransB;
            out vec4 fc;
            void main() {
                int i = int(gl_FragCoord.y), j = int(gl_FragCoord.x);
                float s = 0.0;
                if (uTransA == 0 && uTransB == 0) {
                    for (int k = 0; k < uK; k++)
                        s += texelFetch(uA, ivec2(k, i), 0).r * texelFetch(uB, ivec2(j, k), 0).r;
                } else if (uTransA != 0 && uTransB == 0) {
                    for (int k = 0; k < uK; k++)
                        s += texelFetch(uA, ivec2(i, k), 0).r * texelFetch(uB, ivec2(j, k), 0).r;
                } else if (uTransA == 0 && uTransB != 0) {
                    for (int k = 0; k < uK; k++)
                        s += texelFetch(uA, ivec2(k, i), 0).r * texelFetch(uB, ivec2(k, j), 0).r;
                } else {
                    for (int k = 0; k < uK; k++)
                        s += texelFetch(uA, ivec2(i, k), 0).r * texelFetch(uB, ivec2(k, j), 0).r;
                }
                fc = vec4(s, 0.0, 0.0, 1.0);
            }`);

        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
            throw new Error('Link: ' + gl.getProgramInfoLog(prog));

        this.prog = prog;
        this.loc = {
            uA: gl.getUniformLocation(prog, 'uA'),
            uB: gl.getUniformLocation(prog, 'uB'),
            uK: gl.getUniformLocation(prog, 'uK'),
            uTransA: gl.getUniformLocation(prog, 'uTransA'),
            uTransB: gl.getUniformLocation(prog, 'uTransB'),
        };

        this.texA = this._makeTex();
        this.texB = this._makeTex();
        this.texOut = this._makeTex();
        this.fb = gl.createFramebuffer();
        this.vao = gl.createVertexArray();
        this._szA = [0, 0];
        this._szB = [0, 0];
        this._szOut = [0, 0];
    }

    compute(A, B, M, K, N, transA, transB) {
        const gl = this.gl;

        // Upload inputs
        this._upload(this.texA, A, transA ? K : M, transA ? M : K, this._szA);
        this._upload(this.texB, B, transB ? N : K, transB ? K : N, this._szB);

        // Prepare output
        this._resize(this.texOut, M, N, this._szOut);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
            gl.TEXTURE_2D, this.texOut, 0);

        // Draw
        gl.useProgram(this.prog);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.texA);
        gl.uniform1i(this.loc.uA, 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.texB);
        gl.uniform1i(this.loc.uB, 1);
        gl.uniform1i(this.loc.uK, K);
        gl.uniform1i(this.loc.uTransA, transA ? 1 : 0);
        gl.uniform1i(this.loc.uTransB, transB ? 1 : 0);
        gl.viewport(0, 0, N, M);
        gl.bindVertexArray(this.vao);
        gl.drawArrays(gl.TRIANGLES, 0, 3);

        // Read back
        const C = new Float32Array(M * N);
        gl.readPixels(0, 0, N, M, gl.RED, gl.FLOAT, C);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return C;
    }

    _shader(type, src) {
        const gl = this.gl, sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS))
            throw new Error('Compile: ' + gl.getShaderInfoLog(sh));
        return sh;
    }

    _makeTex() {
        const gl = this.gl, t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return t;
    }

    _upload(tex, data, rows, cols, sz) {
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_2D, tex);
        if (sz[0] !== rows || sz[1] !== cols) {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, cols, rows, 0,
                gl.RED, gl.FLOAT, data);
            sz[0] = rows; sz[1] = cols;
        } else {
            gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, cols, rows,
                gl.RED, gl.FLOAT, data);
        }
    }

    _resize(tex, rows, cols, sz) {
        if (sz[0] === rows && sz[1] === cols) return;
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, cols, rows, 0,
            gl.RED, gl.FLOAT, null);
        sz[0] = rows; sz[1] = cols;
    }
}

if (typeof module !== 'undefined') module.exports = { WebGLMatMul };
