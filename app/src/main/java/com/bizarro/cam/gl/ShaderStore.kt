package com.bizarro.cam.gl

object ShaderStore {

    const val QUAD_VERT = """
attribute vec2 aPos;
attribute vec2 aUV;
uniform mat4 uTexMatrix;
uniform vec4 uCrop;
varying vec2 vUV;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vec2 uv = vec2(mix(uCrop.x, uCrop.z, aUV.x), mix(uCrop.y, uCrop.w, aUV.y));
    vUV = (uTexMatrix * vec4(uv, 0.0, 1.0)).xy;
}
"""

    const val GRID_VERT = """
attribute vec2 aPos;
attribute vec2 aUV;
uniform mat4 uTexMatrix;
uniform float uMeshAmp;
uniform float uTime;
uniform float uMorph;
uniform float uPointCount;
uniform vec2 uPoints[128];
varying vec2 vUV;
varying vec2 vWarp;
void main() {
    vec2 uv = aUV;
    vec2 disp = vec2(0.0);
    float wsum = 0.0;
    vec2 mwarp = vec2(0.0);
    for (int i = 0; i < 128; i++) {
        float en = step(float(i), uPointCount - 0.5);
        vec2 dv = uv - uPoints[i];
        float l = length(dv) + 1e-4;
        float infl = 0.012 / (l * l * 18.0 + 0.02);
        disp += (dv / l) * infl * en;
        wsum += infl * en;
        float l2 = l * l;
        mwarp += dv * (1.0 / (l2 * 160.0 + 0.6)) * en;
    }
    disp /= max(wsum, 1e-3);
    float n1 = sin(uv.y * 21.0 + uTime * 1.9) * cos(uv.x * 17.0 - uTime * 1.3);
    float n2 = sin(uv.x * 29.0 - uTime * 2.3 + uv.y * 7.0);
    vec2 total = disp * uMeshAmp * 0.10 + vec2(n1, n2) * 0.006 * uMeshAmp;
    vWarp = mwarp * uMorph * 0.10;
    gl_Position = vec4(aPos + total * 2.0, 0.0, 1.0);
    vUV = (uTexMatrix * vec4(uv + total * 0.6, 0.0, 1.0)).xy;
}
"""

    const val DISP_FRAG = """
#extension GL_OES_EGL_image_external : require
precision mediump float;
varying vec2 vUV;
uniform samplerExternalOES uTex;
uniform sampler2D uPrev;
uniform vec2 uResolution;
uniform float uTime;
void main() {
    vec2 px = 1.0 / uResolution;
    vec3 c  = texture2D(uTex, vUV).rgb;
    vec3 cL = texture2D(uTex, vUV - vec2(px.x, 0.0)).rgb;
    vec3 cR = texture2D(uTex, vUV + vec2(px.x, 0.0)).rgb;
    vec3 cU = texture2D(uTex, vUV - vec2(0.0, px.y)).rgb;
    vec3 cD = texture2D(uTex, vUV + vec2(0.0, px.y)).rgb;
    float l  = dot(c,  vec3(0.299, 0.587, 0.114));
    float lL = dot(cL, vec3(0.299, 0.587, 0.114));
    float lR = dot(cR, vec3(0.299, 0.587, 0.114));
    float lU = dot(cU, vec3(0.299, 0.587, 0.114));
    float lD = dot(cD, vec3(0.299, 0.587, 0.114));
    float gx = (lR - lL) * 0.5;
    float gy = (lD - lU) * 0.5;
    float edge = clamp(length(vec2(gx, gy)) * 2.5, 0.0, 1.0);
    vec3 p = texture2D(uPrev, vUV).rgb;
    float motion = clamp(distance(c, p) * 3.0, 0.0, 1.0);
    vec2 disp = vec2(gx, gy) * 3.0 + vec2(sin(vUV.y * 30.0 + uTime), cos(vUV.x * 30.0 - uTime)) * motion * 0.35;
    gl_FragColor = vec4(clamp(disp * 0.5 + 0.5, 0.0, 1.0), edge, motion);
}
"""

    const val COMPOSITE_FRAG = """
#extension GL_OES_EGL_image_external : require
precision mediump float;
varying vec2 vUV;
uniform samplerExternalOES uTex;
uniform sampler2D uPrev;
uniform sampler2D uDisp;
uniform vec2 uResolution;
uniform float uTime;
uniform float uEdge;
uniform float uDisplace;
uniform float uGlitch;
uniform float uPoster;
uniform float uChroma;
uniform float uHue;
uniform float uEcho;
uniform float uNoise;
uniform float uSlit;
uniform float uKaleido;
uniform float uInvert;
uniform float uScan;
varying vec2 vWarp;

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec2 uv = vUV;

    if (uKaleido > 0.001) {
        vec2 d = uv - 0.5;
        float r = length(d);
        float a = atan(d.y, d.x);
        float seg = mix(1.0, 7.0, uKaleido);
        a = mod(a, 6.2831853 / seg);
        a = abs(a - 3.14159265 / seg);
        float dir = step(fract(uTime * 0.07), 0.5) * 2.0 - 1.0;
        uv = 0.5 + r * vec2(cos(a), sin(a)) * vec2(dir, 1.0);
    }

    if (uGlitch > 0.001) {
        float t = floor(uTime * 24.0);
        float row = floor(uv.y * 90.0);
        float h1 = hash(vec2(row, t));
        if (h1 > 1.0 - uGlitch * 0.35) {
            uv.x = fract(uv.x + (hash(vec2(t, row)) - 0.5) * uGlitch * 0.4);
        }
        vec2 blk = floor(uv * vec2(16.0, 32.0));
        float h2 = hash(blk + t);
        if (h2 > 1.0 - uGlitch * 0.3) {
            uv = mix(uv, (blk + 0.5) / vec2(16.0, 32.0), uGlitch * 0.6);
        }
    }

    vec2 puv = uv;
    if (uSlit > 0.001) {
        float band = floor(uv.x * 18.0);
        float ph = fract(uTime * 0.13 + hash(vec2(band, 7.0)));
        puv = vec2(uv.x, fract(uv.y + ph * uSlit));
    }
    vec3 prev = texture2D(uPrev, puv).rgb;

    if (uMorph > 0.001) {
        uv = clamp(uv + vWarp, 0.0, 1.0);
    }

    vec4 dm = texture2D(uDisp, uv);
    uv = clamp(uv + (dm.rg * 2.0 - 1.0) * uDisplace * 0.08, 0.0, 1.0);

    vec3 c = texture2D(uTex, uv).rgb;

    if (uChroma > 0.001) {
        float ca = uChroma * 0.015;
        c.r = texture2D(uTex, clamp(uv + vec2(ca, 0.0), 0.0, 1.0)).r;
        c.b = texture2D(uTex, clamp(uv - vec2(ca, 0.0), 0.0, 1.0)).b;
    }

    if (uPoster > 0.001) {
        float lv = mix(255.0, 3.0, uPoster);
        c = floor(c * lv + 0.5) / lv;
    }

    if (uHue > 0.001) {
        vec3 hsv = rgb2hsv(clamp(c, 0.0, 1.0));
        hsv.x = fract(hsv.x + uHue + uTime * 0.15 * uHue);
        c = hsv2rgb(hsv);
    }

    if (uInvert > 0.5) c = 1.0 - c;

    c *= 1.0 - uScan * 0.4 * (0.5 + 0.5 * sin(uv.y * uResolution.y * 3.14159));

    c += (hash(uv * uResolution + fract(uTime) * 61.7) - 0.5) * uNoise * 0.6;

    c = mix(c, prev, uEcho * 0.85);

    float e = dm.b;
    vec3 neon = mix(vec3(0.0, 1.0, 0.8), vec3(1.0, 0.1, 0.9), 0.5 + 0.5 * sin(uTime * 2.0));
    c = mix(c, neon * 1.6, clamp(e * uEdge * 1.4, 0.0, 0.9));

    gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""

    const val RAW_FRAG = """
#extension GL_OES_EGL_image_external : require
precision mediump float;
varying vec2 vUV;
uniform samplerExternalOES uTex;
void main() {
    gl_FragColor = texture2D(uTex, vUV);
}
"""

    const val BLIT_FRAG = """
precision mediump float;
varying vec2 vUV;
uniform sampler2D uTex;
void main() {
    gl_FragColor = vec4(texture2D(uTex, vUV).rgb, 1.0);
}
"""

    const val LINE_VERT = """
attribute vec2 aPos;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

    const val LINE_FRAG = """
precision mediump float;
uniform vec4 uColor;
void main() {
    gl_FragColor = uColor;
}
"""
}
