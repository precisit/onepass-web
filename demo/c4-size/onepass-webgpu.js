/* onepass-webgpu 9dbc4f3 (MIT), https://github.com/precisit/onepass-webgpu */
var V=n=>`${n?`enable f16;
`:""}alias WT = ${n?"f16":"f32"};
`,Z=`
var<workgroup> red: array<f32, 256>;

fn wsum(v: f32, t: u32) -> f32 {
  red[t] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) { red[t] += red[t + s]; }
    workgroupBarrier();
  }
  let out = red[0];
  workgroupBarrier();
  return out;
}
`,ae=n=>`${V(n)}
struct P { width: u32, seqLen: u32, idsOff: u32, embOff: u32, posOff: u32, lnW: u32, lnB: u32, eps: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> ids: array<i32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${Z}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let m = wg.x + wg.y * 65535u;
  let id = u32(ids[p.idsOff + m]);
  let pos = m % p.seqLen;
  var x = 0.0;
  if (t < p.width) {
    x = f32(W[p.embOff + id * p.width + t]) + f32(W[p.posOff + pos * p.width + t]);
    X[m * p.width + t] = x;
  }
  let mean = wsum(select(0.0, x, t < p.width), t) / f32(p.width);
  let d = select(0.0, x - mean, t < p.width);
  let inv = 1.0 / sqrt(wsum(d * d, t) / f32(p.width) + p.eps);
  if (t < p.width) { Y[m * p.width + t] = d * inv * f32(W[p.lnW + t]) + f32(W[p.lnB + t]); }
}
`,ye=n=>n==="int8"?`
fn unpack8(q: u32) -> vec4<f32> {
  let s = bitcast<i32>(q);
  return vec4<f32>(vec4<i32>((s << 24u) >> 24u, (s << 16u) >> 24u, (s << 8u) >> 24u, s >> 24u));
}`:`
fn unpack8(q: u32) -> vec4<f32> {
  return vec4<f32>(f32(q & 255u), f32((q >> 8u) & 255u), f32((q >> 16u) & 255u), f32(q >> 24u));
}`,ee=(n,e,s="")=>`${V(n)}
const RM = ${e.RM}u;
const KS = ${e.KS}u;
${e.plugin?`struct P { M: u32, N: u32, K: u32, qOff: u32, aBias: u32, fOff: u32, x0: u32, x1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<u32>;
@group(0) @binding(4) var<storage, read> W: array<vec4<WT>>;
fn qbyte(i: u32) -> u32 { let b = p.qOff + i; return (Q[b >> 2u] >> ((b & 3u) * 8u)) & 255u; }
fn fval(i: u32) -> f32 { let j = p.fOff + i; return f32(W[j / 4u][j % 4u]); }
${s}`:`struct P { M: u32, N: u32, K: u32, wOff: u32, aBias: u32, zp: f32, scale: f32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
${e.w8?`@group(0) @binding(1) var<storage, read> Q: array<u32>;
${e.aSplits?"@group(0) @binding(4) var<storage, read> W: array<vec4<WT>>;":""}
${ye(e.w8)}`:"@group(0) @binding(1) var<storage, read> W: array<vec4<WT>>;"}`}
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<vec4<f32>>;
var<workgroup> at: array<f32, ${e.RM*e.KS}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let row0 = wg.y * RM;
  let k0 = wg.z * KS;
  for (var e = t; e < RM * KS; e += 64u) {
    let m = row0 + e / KS;
    let k = k0 + e % KS;
    var a = 0.0;
    if (m < p.M) {
${e.aSplits===0?"      a = A[m * p.K + k];":`      for (var j = 0u; j < ${e.aSplits}u; j += 1u) { a += A[(j * p.M + m) * p.K + k]; }
      a += f32(W[(p.aBias + k) / 4u][k % 4u]);${e.aRelu?`
      a = max(a, 0.0);`:""}`}
    }
    at[e] = a;
  }
  workgroupBarrier();
  let n4 = wg.x * 64u + t;
  if (n4 * 4u >= p.N) { return; }
  let stride = p.N / 4u;
  var acc: array<vec4<f32>, RM>;
  ${e.pluginInner?"inner(k0, n4, &acc);":""}
  ${e.plugin?"":`var wi = ${e.w8?"p.wOff":"p.wOff / 4u"} + k0 * stride + n4;`}
  for (var kk = 0u; kk < ${e.pluginInner?"0u":"KS"}; kk += 1u) {
    ${e.plugin?"let w = w4(k0 + kk, n4);":`let w = ${e.w8?"unpack8(Q[wi]) - vec4<f32>(p.zp)":"vec4<f32>(W[wi])"};
    wi += stride;`}
    for (var r = 0u; r < RM; r += 1u) { acc[r] = fma(vec4<f32>(at[r * KS + kk]), w, acc[r]); }
  }
  let rows = min(RM, p.M - row0);
  for (var r = 0u; r < rows; r += 1u) { Out[(wg.z * p.M + row0 + r) * stride + n4] = acc[r]${e.w8?" * p.scale":""}; }
}
`,ue=(n,e)=>`${V(n)}
struct P { M: u32, width: u32, bOff: u32, lnW: u32, lnB: u32, eps: f32, p6: u32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> Part: array<f32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${Z}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let m = wg.x + wg.y * 65535u;
  var x = 0.0;
  if (t < p.width) {
    var y = 0.0;
    for (var j = 0u; j < ${e}u; j += 1u) { y += Part[(j * p.M + m) * p.width + t]; }
    x = X[m * p.width + t] + (y + f32(W[p.bOff + t]));
    X[m * p.width + t] = x;
  }
  let mean = wsum(select(0.0, x, t < p.width), t) / f32(p.width);
  let d = select(0.0, x - mean, t < p.width);
  let inv = 1.0 / sqrt(wsum(d * d, t) / f32(p.width) + p.eps);
  if (t < p.width) { Y[m * p.width + t] = d * inv * f32(W[p.lnW + t]) + f32(W[p.lnB + t]); }
}
`,fe=(n,e,s,t)=>`${V(n)}
const D = ${e}u;
const LM = ${t}u;
const QB = 8u;
struct P { L: u32, idsOff: u32, width: u32, scale: f32, M: u32, bOff: u32, p6: u32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> Part: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<i32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
var<workgroup> ks: array<f32, ${t*e}>;
var<workgroup> vs: array<f32, ${t*e}>;
var<workgroup> qs: array<f32, ${8*e}>;
var<workgroup> sc: array<f32, ${8*t}>;

fn proj(row: u32, col: u32) -> f32 {
  var a = 0.0;
  for (var j = 0u; j < ${s}u; j += 1u) { a += Part[(j * p.M + row) * 3u * p.width + col]; }
  return a + f32(W[p.bOff + col]);
}

fn valid(base: u32, l: u32) -> bool { return l == 0u || ids[p.idsOff + base + l] != 0; }

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let h = wg.x;
  let base = wg.y * p.L;
  let q0 = wg.z * QB;
  for (var e = t; e < p.L * D; e += 256u) {
    let col = h * D + e % D;
    ks[e] = proj(base + e / D, p.width + col);
    vs[e] = proj(base + e / D, 2u * p.width + col);
  }
  for (var e = t; e < QB * D; e += 256u) {
    let qi = q0 + e / D;
    qs[e] = select(0.0, proj(base + qi, h * D + e % D), qi < p.L);
  }
  workgroupBarrier();
  let qi = t / 32u;
  let lane = t % 32u;
  for (var l = lane; l < p.L; l += 32u) {
    var dot = 0.0;
    for (var d = 0u; d < D; d += 1u) { dot += qs[qi * D + d] * ks[l * D + d]; }
    sc[qi * LM + l] = select(-3.0e38, dot * p.scale, valid(base, l));
  }
  workgroupBarrier();
  var mx = -3.0e38;
  for (var l = 0u; l < p.L; l += 1u) { mx = max(mx, sc[qi * LM + l]); }
  workgroupBarrier();
  for (var l = lane; l < p.L; l += 32u) {
    sc[qi * LM + l] = select(0.0, exp(sc[qi * LM + l] - mx), valid(base, l));
  }
  workgroupBarrier();
  let query = q0 + qi;
  if (query >= p.L) { return; }
  var sum = 0.0;
  for (var l = 0u; l < p.L; l += 1u) { sum += sc[qi * LM + l]; }
  for (var d = lane; d < D; d += 32u) {
    var o = 0.0;
    for (var l = 0u; l < p.L; l += 1u) { o += sc[qi * LM + l] * vs[l * D + d]; }
    O[(base + query) * p.width + h * D + d] = o / sum;
  }
}
`,pe=(n,e,s)=>`${V(n)}
const D = ${e}u;
const QB = 8u;
const CH = 32u;
const DL = ${Math.ceil(e/32)}u;
struct P { L: u32, idsOff: u32, width: u32, scale: f32, M: u32, bOff: u32, p6: u32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> Part: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<i32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
var<workgroup> ks: array<f32, ${32*e}>;
var<workgroup> vs: array<f32, ${32*e}>;
var<workgroup> qs: array<f32, ${8*e}>;
var<workgroup> sc: array<f32, 256>;

fn proj(row: u32, col: u32) -> f32 {
  var a = 0.0;
  for (var j = 0u; j < ${s}u; j += 1u) { a += Part[(j * p.M + row) * 3u * p.width + col]; }
  return a + f32(W[p.bOff + col]);
}

fn valid(base: u32, l: u32) -> bool { return l == 0u || ids[p.idsOff + base + l] != 0; }

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let h = wg.x;
  let base = wg.y * p.L;
  let q0 = wg.z * QB;
  for (var e = t; e < QB * D; e += 256u) {
    let qi = q0 + e / D;
    qs[e] = select(0.0, proj(base + qi, h * D + e % D), qi < p.L);
  }
  let qi = t / 32u;
  let lane = t % 32u;
  var m = -3.0e38;
  var sum = 0.0;
  var o: array<f32, DL>;
  for (var c0 = 0u; c0 < p.L; c0 += CH) {
    workgroupBarrier();
    for (var e = t; e < CH * D; e += 256u) {
      let l = c0 + e / D;
      let col = h * D + e % D;
      ks[e] = select(0.0, proj(base + l, p.width + col), l < p.L);
      vs[e] = select(0.0, proj(base + l, 2u * p.width + col), l < p.L);
    }
    workgroupBarrier();
    let l = c0 + lane;
    var s = -3.0e38;
    if (l < p.L && valid(base, l)) {
      var dot = 0.0;
      for (var d = 0u; d < D; d += 1u) { dot += qs[qi * D + d] * ks[lane * D + d]; }
      s = dot * p.scale;
    }
    sc[qi * CH + lane] = s;
    workgroupBarrier();
    var cm = m;
    for (var i = 0u; i < CH; i += 1u) { cm = max(cm, sc[qi * CH + i]); }
    let corr = exp(m - cm);
    sum = sum * corr;
    for (var j = 0u; j < DL; j += 1u) { o[j] = o[j] * corr; }
    for (var i = 0u; i < CH; i += 1u) {
      let li = c0 + i;
      if (li >= p.L || !valid(base, li)) { continue; }
      let e = exp(sc[qi * CH + i] - cm);
      sum += e;
      for (var j = 0u; j < DL; j += 1u) {
        let d = lane + 32u * j;
        if (d < D) { o[j] += e * vs[i * D + d]; }
      }
    }
    m = cm;
  }
  let query = q0 + qi;
  if (query >= p.L) { return; }
  for (var j = 0u; j < DL; j += 1u) {
    let d = lane + 32u * j;
    if (d < D) { O[(base + query) * p.width + h * D + d] = o[j] / sum; }
  }
}
`,le=n=>`${V(n)}
struct P { L: u32, width: u32, idsOff: u32, lnW: u32, lnB: u32, eps: f32, p6: u32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<i32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${Z}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let s = wg.x + wg.y * 65535u;
  var x = 0.0;
  if (t < p.width) {
    var count = 0.0;
    for (var l = 0u; l < p.L; l += 1u) {
      if (ids[p.idsOff + s * p.L + l] != 0) {
        x += X[(s * p.L + l) * p.width + t];
        count += 1.0;
      }
    }
    x = x / max(count, 1.0);
  }
  let mean = wsum(select(0.0, x, t < p.width), t) / f32(p.width);
  let d = select(0.0, x - mean, t < p.width);
  let inv = 1.0 / sqrt(wsum(d * d, t) / f32(p.width) + p.eps);
  if (t < p.width) { Y[s * p.width + t] = d * inv * f32(W[p.lnW + t]) + f32(W[p.lnB + t]); }
}
`,ce=(n,e,s,t)=>`
struct P { Lc: u32, slots: u32, rank: u32, ctxOff: u32, maskOff: u32, scale: f32, Mq: u32, Mc: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> ids: array<i32>;
@group(0) @binding(5) var<storage, read_write> logits: array<f32>;
var<workgroup> qv: array<f32, 1024>;
var<workgroup> sc: array<f32, ${Math.max(64,Math.ceil(t/64)*64)}>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let n = wg.x;
  let b = wg.y;
  let qrow = b * p.slots + n;
  let krow = b * p.Lc;
  for (var r = t; r < p.rank; r += 256u) {
    var a = 0.0;
    for (var j = 0u; j < ${n}u; j += 1u) { a += q[(j * p.Mq + qrow) * p.rank + r]; }
    qv[r] = a;
  }
  workgroupBarrier();
  // scores: 4 threads per context position, 64 positions per pass
  let part = t % 4u;
  for (var l0 = 0u; l0 < p.Lc; l0 += 64u) {
    let l = l0 + t / 4u;
    var dot = 0.0;
    if (l < p.Lc) {
      for (var r = part; r < p.rank; r += 4u) {
        var kk = 0.0;
        for (var j = 0u; j < ${e}u; j += 1u) { kk += k[(j * p.Mc + krow + l) * p.rank + r]; }
        dot += qv[r] * kk;
      }
    }
    red[t] = dot;
    workgroupBarrier();
    if (part == 0u && l < p.Lc) {
      sc[l] = (red[t] + red[t + 1u] + red[t + 2u] + red[t + 3u]) * p.scale;
    }
    workgroupBarrier();
  }
  var mx = -3.0e38;
  var sum = 0.0;
  for (var i = 0u; i < p.Lc; i += 1u) { if (ids[p.ctxOff + krow + i] != 0) { mx = max(mx, sc[i]); } }
  for (var i = 0u; i < p.Lc; i += 1u) { if (ids[p.ctxOff + krow + i] != 0) { sum += exp(sc[i] - mx); } }
  var acc = 0.0;
  for (var r = t; r < p.rank; r += 256u) {
    var att = 0.0;
    for (var i = 0u; i < p.Lc; i += 1u) {
      if (ids[p.ctxOff + krow + i] == 0) { continue; }
      var vv = 0.0;
      for (var j = 0u; j < ${s}u; j += 1u) { vv += v[(j * p.Mc + krow + i) * p.rank + r]; }
      att += (exp(sc[i] - mx) / sum) * vv;
    }
    acc += qv[r] * att;
  }
  workgroupBarrier();
  red[t] = acc;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) { red[t] += red[t + s]; }
    workgroupBarrier();
  }
  if (t == 0u) {
    logits[qrow] = select(-3.4028234663852886e38, red[0] * p.scale, ids[p.maskOff + qrow] != 0);
  }
}
`;var te=class{constructor(e,s=0,t=e.length){this.buf=e;this.end=t;this.pos=s}pos;varint(){let e=0,s=1;for(;;){let t=this.buf[this.pos++];if(e+=(t&127)*s,t<128)return e;s*=128}}skip(e){if(e===0)this.varint();else if(e===1)this.pos+=8;else if(e===2){let s=this.varint();this.pos+=s}else if(e===5)this.pos+=4;else throw new Error(`onnx: unsupported wire type ${e}`)}};function re(n,e,s,t){let r=new te(n,e,s);for(;r.pos<s;){let o=r.varint(),a=Math.floor(o/8),f=o&7,h=r.pos;t(a,f,r),r.pos===h&&r.skip(f)}}function xe(n,e,s){let t={name:"",dims:[],dataType:0,bytes:new Uint8Array(0)},r=null,o=null;re(n,e,s,(f,h,u)=>{if(f===1&&h===0)t.dims.push(u.varint());else if(f===1&&h===2){let p=u.varint()+u.pos;for(;u.pos<p;)t.dims.push(u.varint())}else if(f===2)t.dataType=u.varint();else if(f===8){let p=u.varint();t.name=new TextDecoder().decode(n.subarray(u.pos,u.pos+p)),u.pos+=p}else if(f===9){let p=u.varint();t.bytes=n.subarray(u.pos,u.pos+p),u.pos+=p}else if(f===4&&h===2){let p=u.varint();r=Array.from(new Float32Array(n.slice(u.pos,u.pos+p).buffer)),u.pos+=p}else if(f===5&&h===2){let p=u.varint(),k=u.pos+p;for(o??=[];u.pos<k;)o.push(u.varint())}else if(f===5&&h===0)(o??=[]).push(u.varint());else if(f===14&&u.varint()===1)throw new Error("onnx: external data is not supported yet")}),r&&(t.bytes=new Uint8Array(new Float32Array(r).buffer));let a=o;if(a&&!t.bytes.byteLength&&(t.dataType===2||t.dataType===3?t.bytes=new Uint8Array(a.map(f=>f&255)):t.dataType===6&&(t.bytes=new Uint8Array(new Int32Array(a).buffer))),!t.bytes.byteLength&&t.dims.length===0){let f={1:4,2:1,3:1,6:4,7:8,10:2}[t.dataType]??0;t.bytes=new Uint8Array(f)}return t}function H(n){let e=n instanceof Uint8Array?n:new Uint8Array(n),s=new Map;return re(e,0,e.length,(t,r,o)=>{if(t!==7||r!==2)return;let a=o.varint(),f=o.pos+a;re(e,o.pos,f,(h,u,p)=>{if(h!==5||u!==2)return;let k=p.varint(),d=xe(e,p.pos,p.pos+k);s.set(d.name,d),p.pos+=k}),o.pos=f}),s}function de(n){let e=n&32768?-1:1,s=n>>10&31,t=n&1023;return s===0?e*t*2**-24:s===31?t?NaN:e*(1/0):e*(1+t/1024)*2**(s-15)}function ne(n,e=!1){let s;if(n.dataType===1)s=new Float32Array(n.bytes.byteLength/4),new Uint8Array(s.buffer).set(n.bytes);else if(n.dataType===10){let a=new Uint16Array(n.bytes.byteLength/2);new Uint8Array(a.buffer).set(n.bytes),s=Float32Array.from(a,de)}else throw new Error(`onnx: ${n.name} is data type ${n.dataType}, expected float32 or float16`);if(!e)return s;let[t,r]=n.dims,o=new Float32Array(s.length);for(let a=0;a<t;a+=1)for(let f=0;f<r;f+=1)o[f*t+a]=s[a*r+f];return o}function ge(n,e,s){if(n.dataType!==2&&n.dataType!==3)throw new Error(`onnx: ${n.name} is not int8 or uint8`);let t=n.dataType===3?new Int8Array(n.bytes.buffer,n.bytes.byteOffset,n.bytes.byteLength):n.bytes,r=new Float32Array(t.length);for(let o=0;o<t.length;o+=1)r[o]=Math.fround(Math.fround(t[o]-s)*e);return r}var oe=class{constructor(e,s,t){this.b=e;this.end=t;this.pos=s}pos;varint(){let e=0,s=0,t;do{if(t=this.b[this.pos++],s<49)e+=(t&127)*2**s;else return this.bigTail(e,s,t);s+=7}while(t&128);return e}bigTail(e,s,t){let r=BigInt(e)+(BigInt(t&127)<<BigInt(s)),o=t;for(;o&128;)s+=7,o=this.b[this.pos++],r+=BigInt(o&127)<<BigInt(s);return Number(BigInt.asIntN(64,r))}str(e){let s=new TextDecoder().decode(this.b.subarray(this.pos,this.pos+e));return this.pos+=e,s}skip(e){if(e===0)this.varint();else if(e===1)this.pos+=8;else if(e===2){let s=this.varint();this.pos+=s}else if(e===5)this.pos+=4;else throw new Error(`onnx: unsupported wire type ${e}`)}};function se(n,e,s,t){let r=new oe(n,e,s);for(;r.pos<s;){let o=r.varint(),a=Math.floor(o/8),f=o&7,h=r.pos;t(a,f,r),r.pos===h&&r.skip(f)}}function Y(n,e){let s=n.varint(),t=n.pos;se(n.b,t,t+s,e),n.pos=t+s}function $e(n,e,s){let t={name:"",dims:[],dataType:0,bytes:new Uint8Array(0)};return se(n,e,s,(r,o,a)=>{if(r===1&&o===0)t.dims.push(a.varint());else if(r===1&&o===2){let f=a.varint(),h=a.pos+f;for(;a.pos<h;)t.dims.push(a.varint())}else if(r===2)t.dataType=a.varint();else if(r===8)t.name=a.str(a.varint());else if(r===9){let f=a.varint();t.bytes=n.subarray(a.pos,a.pos+f),a.pos+=f}else if(r===7&&o===2){let f=a.varint(),h=a.pos+f,u=[];for(;a.pos<h;)u.push(a.varint());let p=new BigInt64Array(u.map(k=>BigInt(k)));t.bytes=new Uint8Array(p.buffer)}}),t}function ke(n){let e=[],s=new Map;return se(n,0,n.length,(t,r,o)=>{t!==7||r!==2||Y(o,(a,f,h)=>{if(a===1&&f===2){let u={opType:"",name:"",inputs:[],outputs:[],attrs:new Map};Y(h,(p,k,d)=>{if(p===1)u.inputs.push(d.str(d.varint()));else if(p===2)u.outputs.push(d.str(d.varint()));else if(p===3)u.name=d.str(d.varint());else if(p===4)u.opType=d.str(d.varint());else if(p===5){let m={name:""};Y(d,(w,q,b)=>{if(w===1)m.name=b.str(b.varint());else if(w===2&&q===5)m.f=new DataView(b.b.buffer,b.b.byteOffset+b.pos,4).getFloat32(0,!0),b.pos+=4;else if(w===3)m.i=b.varint();else if(w===5&&q===2){let g=b.varint();m.t=$e(b.b,b.pos,b.pos+g),b.pos+=g}else if(w===8)if(m.ints??=[],q===0)m.ints.push(b.varint());else{let g=b.varint(),M=b.pos+g;for(;b.pos<M;)m.ints.push(b.varint())}}),u.attrs.set(m.name,m)}}),e.push(u)}else if(a===11&&f===2){let u="",p=[];Y(h,(k,d,m)=>{k===1?u=m.str(m.varint()):k===2&&Y(m,(w,q,b)=>{w===1&&Y(b,(g,M,B)=>{g===2&&Y(B,(L,W,F)=>{if(L!==1)return;let j=-1;Y(F,(v,x,U)=>{v===1&&(j=U.varint())}),p.push(j)})})})}),s.set(u,p)}})}),{nodes:e,inputs:s,inits:H(n)}}var O=class extends Error{};function he(n){let e=new DataView(n.bytes.buffer,n.bytes.byteOffset,n.bytes.byteLength);if(n.dataType===1)return e.getFloat32(0,!0);if(n.dataType===3)return e.getInt8(0);if(n.dataType===2)return e.getUint8(0);throw new O(`${n.name}: unexpected scalar type ${n.dataType}`)}function me(n){if(n.dataType!==7)return[];let e=new BigInt64Array(n.bytes.slice().buffer);return Array.from(e,Number)}function ie(n,e="model.onnx"){let s=n instanceof Uint8Array?n:new Uint8Array(n),{nodes:t,inputs:r,inits:o}=ke(s),a=new Map;for(let i of t)for(let l of i.outputs)a.set(l,i);let f=i=>o.get(i)?.dims??[],h=r.get("context_ids"),u=r.get("option_ids");if(!h||!u||!r.has("option_mask"))throw new O("expected inputs context_ids / option_ids / option_mask");let[p,k,d]=[h[1],u[1],u[2]],m=i=>{if(!i.endsWith("_quantized"))return;let l=i.slice(0,-10),y=o.get(`${l}_scale`),$=o.get(`${l}_zero_point`),G=o.get(i);if(!y||!$||y.bytes.byteLength>4||$.bytes.byteLength>1)throw new O(`${i}: only per-tensor quantization`);if(G.dataType!==2&&G.dataType!==3)throw new O(`${i}: unsupported quantized type`);return{dtype:G.dataType===3?"int8":"uint8",scale:he(y),zero_point:he($)}},w=[];for(let i of t)if(o.has(i.inputs[1]??"")){if(i.opType==="MatMul"||i.opType==="MatMulInteger")i.opType==="MatMulInteger"&&m(i.inputs[1]),w.push({ref:[i.inputs[1],!1],bias:null,node:i});else if(i.opType==="Gemm"){let l=(y,$)=>i.attrs.get(y)?.i??i.attrs.get(y)?.f??$;if(l("transA",0)||l("alpha",1)!==1||l("beta",1)!==1)throw new O("unsupported Gemm attributes");w.push({ref:[i.inputs[1],!!l("transB",0)],bias:o.has(i.inputs[2]??"")?i.inputs[2]:null,node:i})}}let q=["model.embedding.weight","model.embedding.weight_quantized"].find(i=>o.has(i));if(!q)throw new O("no model.embedding.weight");let[b,g]=f(q),M=i=>{let l=new Set;for(let y of o.keys()){let $=y.match(new RegExp(`^model\\.${i}\\.layers\\.(\\d+)\\.`));$&&l.add(Number($[1]))}return[...l].sort((y,$)=>y-$).map(y=>`model.${i}.layers.${y}`)},B=M("encoder"),L=M("option_encoder");if(L.length!==1)throw new O(`expected one option-encoder layer, found ${L.length}`);let W=4;if(w.length!==W*(B.length+1)+3)throw new O(`expected ${W*(B.length+1)+3} weight matmuls, found ${w.length}`);let F={},j=([i,l])=>l?[...f(i)].reverse():f(i),v=(i,l)=>{let y={initializer:l[0],transpose:l[1],shape:j(l)},$=m(l[0]);$&&(y.quant=$),F[i]=y};v("embedding",[q,!1]);let x=new Map;for(let[i,l]of o)i.startsWith("onnx::Add")&&l.dims.length===2&&l.dims[1]===g&&x.set(l.dims[0],i);if(!x.has(p)||!x.has(d))throw new O(`positional rows for lengths ${p}/${d} not found`);v("pos_context",[x.get(p),!1]),v("pos_option",[x.get(d),!1]);let U=w.every(i=>i.node.name.startsWith("/model/")),E=new Set,I=(i,l)=>{if(!U)return w.slice(l*W,(l+1)*W);let[y,$,G,A]=i.split("."),N=w.filter(Q=>Q.node.name.startsWith(`/${y}/${$}/${G}.${A}/`));if(N.length!==W)throw new O(`${i}: expected ${W} weight matmuls, found ${N.length}`);return N.forEach(Q=>E.add(Q)),N},z=0;[...B,...L].forEach((i,l)=>{let y=l<B.length?`layer${l}`:"option_layer",[$,G,A,N]=I(i,l),Q=[[j($.ref),[g,3*g]],[j(G.ref),[g,g]]];for(let[K,X]of Q)if(K.join()!==X.join())throw new O(`${i}: weight shape ${K}, wanted ${X}`);if(j(A.ref)[0]!==g||j(N.ref)[1]!==g)throw new O(`${i}: MLP shapes`);z=j(A.ref)[1],v(`${y}.qkv.w`,$.ref),v(`${y}.qkv.b`,[`${i}.self_attn.in_proj_bias`,!1]),v(`${y}.out.w`,G.ref),v(`${y}.out.b`,[G.bias??`${i}.self_attn.out_proj.bias`,!1]),v(`${y}.ff1.w`,A.ref),v(`${y}.ff1.b`,[`${i}.linear1.bias`,!1]),v(`${y}.ff2.w`,N.ref),v(`${y}.ff2.b`,[`${i}.linear2.bias`,!1]);for(let K of["norm1","norm2"])v(`${y}.${K}.w`,[`${i}.${K}.weight`,!1]),v(`${y}.${K}.b`,[`${i}.${K}.bias`,!1])});let D=i=>{let l=a.get(i.inputs[0]);for(;l&&l.opType!=="LayerNormalization";)l=a.get(l.inputs[0]);if(!l)throw new O("head matmul not fed by a LayerNormalization");return l.inputs[1]},c=U?w.filter(i=>!E.has(i)):w.slice(-3);if(c.length!==3)throw new O(`expected three head matmuls, found ${c.length}`);let _=c.map(i=>D(i.node)),P=c.filter((i,l)=>_[l]==="model.head.option_norm.weight"),T=c.filter((i,l)=>_[l]==="model.head.context_norm.weight");if(P.length!==1||T.length!==2)throw new O(`head roles not recognised: ${_}`);v("head.q.w",P[0].ref),v("head.k.w",T[0].ref),v("head.v.w",T[1].ref);for(let i of["context_norm","option_norm"])v(`head.${i}.w`,[`model.head.${i}.weight`,!1]),v(`head.${i}.b`,[`model.head.${i}.bias`,!1]);let R=F["head.q.w"].shape[1],S=0;for(let i of t){if(i.opType!=="Reshape")continue;let l=o.get(i.inputs[1]),y=a.get(i.inputs[1]),$=l?me(l):y?.opType==="Constant"&&y.attrs.get("value")?.t?me(y.attrs.get("value").t):[];if($.length===3&&$[1]>0&&$[2]>0&&$[1]*$[2]===g&&$[2]<g){S=$[1];break}}if(!S)throw new O("could not find the number of attention heads");for(let[i,l]of Object.entries(F)){let y=o.get(l.initializer);if(!y)throw new O(`${i}: missing initializer ${l.initializer}`);if(y.dataType!==1&&!l.quant)throw new O(`${i}: ${l.initializer} is not float32`)}return{format:"onepass-plan/1",architecture:"onepass-scorer",config:{vocab:b,width:g,heads:S,layers:B.length,ff:z,rank:R,context_len:p,option_slots:k,option_len:d,eps:1e-5},model:{file:e,bytes:s.byteLength,sha256:""},tensors:F}}var qe={splitTarget:4096,qkvSplitTarget:4096},J=256;function we(n,e,s,t){let o=Math.ceil(s/256)*64*Math.ceil(n/4),a=1;for(;a*2<=e/32&&o*a<t;)a*=2;return{S:a,RM:4}}function Pe(n,e,s){let t=(a,f,h,u=s.splitTarget)=>we(a,f,h,u).S*a*h,r={p1:0,p2:0,pq:t(e*n.option_slots,n.width,n.rank)};for(let a of[e*n.context_len,e*n.option_slots*n.option_len])r.p1=Math.max(r.p1,t(a,n.width,3*n.width,s.qkvSplitTarget),t(a,n.width,n.ff)),r.p2=Math.max(r.p2,t(a,n.width,n.width),t(a,n.ff,n.width));let o=e*n.context_len;return r.p1=Math.max(r.p1,t(o,n.width,n.rank)),r.p2=Math.max(r.p2,t(o,n.width,n.rank)),r}function _e(n){let e=globalThis.Float16Array;if(e){let o=new e(n);return new Uint16Array(o.buffer,o.byteOffset,n.length)}let s=new Uint16Array(n.length),t=new Float32Array(1),r=new Uint32Array(t.buffer);for(let o=0;o<n.length;o+=1){t[0]=n[o];let a=r[0],f=a>>>16&32768,h=a>>>23&255,u=a&8388607;if(h===255){s[o]=f|31744|(u?512:0);continue}let p=h-127+15;if(p>=31){s[o]=f|31744;continue}if(p<=0){if(p<-10){s[o]=f;continue}u|=8388608;let m=14-p,w=u>>>m,q=u&(1<<m)-1,b=1<<m-1;(q>b||q===b&&w&1)&&(w+=1),s[o]=f|w;continue}let k=p<<10|u>>>13,d=u&8191;(d>4096||d===4096&&k&1)&&(k+=1),s[o]=f|k}return s}var be=class n{constructor(e,s,t,r,o,a,f,h,u,p=null,k=new Map){this.device=e;this.adapterInfo=s;this.precision=t;this.maxBatch=r;this.tuning=u;this.packed=k;this.config=o.config,this.offsets=f;let d=o.config,m=r,w=m*d.context_len,q=m*d.option_slots*d.option_len,b=Math.max(w,q),g=(B,L=0)=>e.createBuffer({size:Math.max(16,B*4),usage:GPUBufferUsage.STORAGE|L}),M={p1:0,p2:0,pq:0};for(let B=1;B<=m;B+=1){let L=Pe(d,B,u);M.p1=Math.max(M.p1,L.p1),M.p2=Math.max(M.p2,L.p2),M.pq=Math.max(M.pq,L.pq)}this.buffers={weights:a,weights8:p??g(4),ids:g(m*(d.context_len+d.option_slots*d.option_len+d.option_slots),GPUBufferUsage.COPY_DST),xc:g(w*d.width),yc:g(w*d.width),xo:g(q*d.width),yo:g(q*d.width),att:g(b*d.width),p1:g(M.p1),p2:g(M.p2),pq:g(M.pq),pooled:g(m*d.option_slots*d.width),logits:g(m*d.option_slots,GPUBufferUsage.COPY_SRC)},this.staging=e.createBuffer({size:272,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),m*d.option_slots*4>256&&(this.staging.destroy(),this.staging=e.createBuffer({size:Math.ceil(m*d.option_slots*4/256)*256+16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})),h&&(this.querySet=e.createQuerySet({type:"timestamp",count:2}),this.queryBuffer=e.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}))}config;get weightFormat(){return this.packed.size?`${[...new Set([...this.packed.values()].map(s=>s.format?.kind??s.kind))].join(" + ")} (${this.precision} for the rest)`:this.precision}get weightBytes(){return this.buffers.weights.size+(this.packed.size?this.buffers.weights8.size:0)}lastGpuMs=null;programs=new Map;pipelines=new Map;offsets=new Map;buffers={};staging;querySet=null;queryBuffer=null;busy=Promise.resolve();static async fromOnnx(e,s={}){return n.load(ie(e),e,s)}static async load(e,s,t={}){if(e.format!=="onepass-plan/1"||e.architecture!=="onepass-scorer")throw new Error(`unsupported plan ${e.format} / ${e.architecture}`);if(!navigator.gpu)throw new Error("WebGPU is not available in this browser");let r=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!r)throw new Error("no WebGPU adapter");let o=r.info;if((/swiftshader|llvmpipe|software|basic render/i.test(`${o.vendor} ${o.architecture} ${o.description}`)||r.isFallbackAdapter===!0)&&!t.allowSoftware)throw new Error(`software WebGPU adapter (${o.vendor} ${o.architecture})`);let f=t.precision??"f32";if(f==="f16"&&!r.features.has("shader-f16"))throw new Error("this GPU has no shader-f16");let h=!!t.gpuTiming&&r.features.has("timestamp-query"),u=[];f==="f16"&&u.push("shader-f16"),h&&u.push("timestamp-query");let p=await r.requestDevice({requiredFeatures:u,requiredLimits:{maxStorageBufferBindingSize:r.limits.maxStorageBufferBindingSize,maxBufferSize:r.limits.maxBufferSize}}),k=H(s),d=[],m=new Map,w=new Map,q=[],b=0,g=0;for(let[v,x]of Object.entries(e.tensors)){let U=k.get(x.initializer);if(!U)throw new Error(`the ONNX file has no initializer ${x.initializer} (for ${v})`);let E=x.shape.reduce((z,D)=>z*D,1);if(x.format){let z=(t.formats??[]).find(T=>T.kind===x.format.kind);if(!z)throw new Error(`${v}: no plugin for weight format ${x.format.kind}`);let{bytes:D,floats:c,params:_}=z.pack(x,k),P={offsetWords:g/4,kind:"int8",scale:1,zeroPoint:0,format:z,offsetBytes:g,floatOffset:0,params:_??[0,0]};w.set(v,P),q.push(D),g+=Math.ceil(D.byteLength/256)*256,c&&(P.floatOffset=b,d.push({name:`${v}#side`,data:c}),m.set(`${v}#side`,b),b+=Math.ceil(c.length/64)*64);continue}if(x.quant&&v!=="embedding"){if(x.shape.length!==2||x.shape[1]%4!==0)throw new Error(`${v}: unsupported 8-bit layout`);if(U.bytes.byteLength!==E)throw new Error(`${v}: ${U.bytes.byteLength} bytes, plan says ${E}`);let z=U.bytes;if(x.transpose){let[D,c]=U.dims;z=new Uint8Array(z.length);for(let _=0;_<D;_+=1)for(let P=0;P<c;P+=1)z[P*D+_]=U.bytes[_*c+P]}w.set(v,{offsetWords:g/4,kind:x.quant.dtype,scale:x.quant.scale,zeroPoint:x.quant.zero_point}),q.push(z),g+=Math.ceil(U.bytes.byteLength/256)*256;continue}let I=x.quant?ge(U,x.quant.scale,x.quant.zero_point):ne(U,x.transpose);if(x.rows){let z=I.length/(x.transpose?U.dims[1]:U.dims[0]);I=I.slice(x.rows[0]*z,x.rows[1]*z)}if(I.length!==E)throw new Error(`${v}: ${I.length} values, plan says ${E}`);m.set(v,b),d.push({name:v,data:I}),b+=Math.ceil(I.length/64)*64}let M=new Float32Array(b);for(let{name:v,data:x}of d)M.set(x,m.get(v));let B=f==="f16"?_e(M):M,L=p.createBuffer({size:B.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});p.queue.writeBuffer(L,0,B.buffer,B.byteOffset,B.byteLength);let W=null;if(g>0){W=p.createBuffer({size:g,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});let v=0;for(let x of q){let U=new Uint8Array(Math.ceil(x.byteLength/4)*4);U.set(x),p.queue.writeBuffer(W,v,U),v+=Math.ceil(x.byteLength/256)*256}}let F={...qe,...t.tuning},j=new n(p,o,f,t.maxBatch??1,e,L,m,h,F,W,w);return j.program(1),await p.queue.onSubmittedWorkDone(),j}pipeline(e,s){let t=this.pipelines.get(e);return t||(t=this.device.createComputePipeline({layout:"auto",compute:{module:this.device.createShaderModule({code:s()}),entryPoint:"main"}}),this.pipelines.set(e,t)),t}off(e){let s=this.offsets.get(e);if(s===void 0)throw new Error(`plan has no tensor ${e}`);return s}program(e){let s=this.programs.get(e);if(s)return s;if(e<1||e>this.maxBatch)throw new Error(`batch ${e} outside 1..${this.maxBatch}`);let t=this.config,r=this.buffers,o=this.precision==="f16",a=[],f=[],h="",u=(c,_,P,T,R)=>{let S=new ArrayBuffer(32),C=new Uint32Array(S),i=new Float32Array(S);_.forEach((l,y)=>{C[y]=l});for(let[l,y]of Object.entries(P))i[Number(l)]=y;a.push(S),f.push({label:h,pipeline:c,buffers:T,groups:R})},p=0,k=e*t.context_len,d=k+e*t.option_slots*t.option_len,m=e*t.context_len,w=e*t.option_slots,q=w*t.option_len,b=this.pipeline(`embed${o}`,()=>ae(o));h="embed",u(b,[t.width,t.context_len,p,this.off("embedding"),this.off("pos_context"),this.off("layer0.norm1.w"),this.off("layer0.norm1.b")],{7:t.eps},[r.weights,r.ids,r.xc,r.yc],[m,1,1]),u(b,[t.width,t.option_len,k,this.off("embedding"),this.off("pos_option"),this.off("option_layer.norm1.w"),this.off("option_layer.norm1.b")],{7:t.eps},[r.weights,r.ids,r.xo,r.yo],[q,1,1]);let g=(c,_,P,T,R,S,C=0,i=null,l=!1,y=this.tuning.splitTarget)=>{let{S:$,RM:G}=we(c,_,P,y),A=this.packed.get(S);if(A?.format){let K={RM:G,KS:_/$,aSplits:C,aRelu:l,plugin:A.format.kind,pluginInner:!!A.format.inner},X=this.pipeline(`mm${o}${JSON.stringify(K)}`,()=>ee(o,K,A.format.wgsl));return u(X,[c,P,_,A.offsetBytes,i?this.off(i):0,A.floatOffset,...A.params],{},[r.weights8,T,R,r.weights],[Math.ceil(P/256),Math.ceil(c/G),$]),$}let N={RM:G,KS:_/$,aSplits:C,aRelu:l,...A?{w8:A.kind}:{}},Q=this.pipeline(`mm${o}${JSON.stringify(N)}`,()=>ee(o,N));return A?u(Q,[c,P,_,A.offsetWords,i?this.off(i):0],{5:A.zeroPoint,6:A.scale},[r.weights8,T,R,...C?[r.weights]:[]],[Math.ceil(P/256),Math.ceil(c/G),$]):u(Q,[c,P,_,this.off(S),i?this.off(i):0],{},[r.weights,T,R],[Math.ceil(P/256),Math.ceil(c/G),$]),$},M=(c,_,P,T,R,S,C)=>{let i=this.pipeline(`rn${o}${P}`,()=>ue(o,P));u(i,[c,t.width,this.off(T),this.off(`${C}.w`),this.off(`${C}.b`)],{5:t.eps},[r.weights,_,R,S],[c,1,1])},B=t.width/t.heads,L=c=>c<=64&&Math.ceil(c/8)*8*B*2*4<=13e3,W=[t.context_len,t.option_len].filter(L),F=W.length?Math.ceil(Math.max(...W)/8)*8:8;if(B>128)throw new Error("head dimension above 128 is not supported");let j=(c,_,P,T,R,S,C)=>{let i=R*S;h=`${c}.qkv`;let l=g(i,t.width,3*t.width,T,r.p1,`${c}.qkv.w`,0,null,!1,this.tuning.qkvSplitTarget);h=`${c}.attention`;let y=L(S)?this.pipeline(`att${o}${l}`,()=>fe(o,B,l,F)):this.pipeline(`attlong${o}${l}`,()=>pe(o,B,l));u(y,[S,C,t.width,0,i,this.off(`${c}.qkv.b`)],{3:1/Math.sqrt(B)},[r.weights,r.p1,r.ids,r.att],[t.heads,R,Math.ceil(S/8)]),h=`${c}.out`;let $=g(i,t.width,t.width,r.att,r.p2,`${c}.out.w`);M(i,r.p2,$,`${c}.out.b`,P,T,`${c}.norm2`),h=`${c}.ff1`;let G=g(i,t.width,t.ff,T,r.p1,`${c}.ff1.w`);h=`${c}.ff2`;let A=g(i,t.ff,t.width,r.p1,r.p2,`${c}.ff2.w`,G,`${c}.ff1.b`,!0);M(i,r.p2,A,`${c}.ff2.b`,P,T,_)};for(let c=0;c<t.layers;c+=1)j(`layer${c}`,c+1<t.layers?`layer${c+1}.norm1`:"head.context_norm",r.xc,r.yc,e,t.context_len,p);j("option_layer","option_layer.norm1",r.xo,r.yo,w,t.option_len,k),h="pool",u(this.pipeline(`pool${o}`,()=>le(o)),[t.option_len,t.width,k,this.off("head.option_norm.w"),this.off("head.option_norm.b")],{5:t.eps},[r.weights,r.xo,r.ids,r.pooled],[w,1,1]),h="head.qkv";let v=g(w,t.width,t.rank,r.pooled,r.pq,"head.q.w"),x=g(m,t.width,t.rank,r.yc,r.p1,"head.k.w"),U=g(m,t.width,t.rank,r.yc,r.p2,"head.v.w");h="head",u(this.pipeline(`head${v}${x}${U}`,()=>ce(v,x,U,t.context_len)),[t.context_len,t.option_slots,t.rank,p,d,0,w,m],{5:1/Math.sqrt(t.rank)},[r.pq,r.p1,r.p2,r.ids,r.logits],[t.option_slots,e,1]);let E=this.device.createBuffer({size:a.length*J,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),I=new Uint8Array(a.length*J);a.forEach((c,_)=>I.set(new Uint8Array(c),_*J)),this.device.queue.writeBuffer(E,0,I);let z=f.map((c,_)=>({label:c.label,pipeline:c.pipeline,groups:c.groups,bindGroup:this.device.createBindGroup({layout:c.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:E,offset:_*J,size:32}},...c.buffers.map((P,T)=>({binding:T+1,resource:{buffer:P}}))]})})),D={batch:e,dispatches:z,params:E};return this.programs.set(e,D),D}wake(e=1){let s=this.busy.then(async()=>{let t=this.program(e),r=this.device.createCommandEncoder(),o=r.beginComputePass();for(let a of t.dispatches)o.setPipeline(a.pipeline),o.setBindGroup(0,a.bindGroup),o.dispatchWorkgroups(...a.groups);o.end(),this.device.queue.submit([r.finish()]),await this.device.queue.onSubmittedWorkDone()});return this.busy=s.catch(()=>{}),s}dispatchCount(e=1){return this.program(e).dispatches.length}score(e,s,t,r=1){let o=this.busy.then(()=>this.run(e,s,t,r));return this.busy=o.catch(()=>{}),o}async run(e,s,t,r){let o=this.config,a=this.program(r),f=r*o.context_len,h=r*o.option_slots*o.option_len,u=r*o.option_slots;if(e.length!==f||s.length!==h||t.length!==u)throw new Error(`input sizes ${e.length}/${s.length}/${t.length}, expected ${f}/${h}/${u}`);let p=new Int32Array(f+h+u);p.set(e,0),p.set(s,f),p.set(t,f+h);let{device:k}=this;k.queue.writeBuffer(this.buffers.ids,0,p);let d=k.createCommandEncoder(),m=d.beginComputePass(this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:void 0);for(let g of a.dispatches)m.setPipeline(g.pipeline),m.setBindGroup(0,g.bindGroup),m.dispatchWorkgroups(...g.groups);m.end();let w=u*4,q=Math.ceil(w/256)*256;d.copyBufferToBuffer(this.buffers.logits,0,this.staging,0,w),this.querySet&&this.queryBuffer&&(d.resolveQuerySet(this.querySet,0,2,this.queryBuffer,0),d.copyBufferToBuffer(this.queryBuffer,0,this.staging,q,16)),k.queue.submit([d.finish()]),await this.staging.mapAsync(GPUMapMode.READ);let b=new Float32Array(this.staging.getMappedRange(0,w).slice(0));if(this.querySet){let g=new BigUint64Array(this.staging.getMappedRange(q,16));this.lastGpuMs=Number(g[1]-g[0])/1e6}return this.staging.unmap(),b}async profile(e,s,t,r=1){if(!this.querySet)throw new Error("profile() needs gpuTiming and an adapter with timestamp-query");await this.score(e,s,t,r);let{device:o}=this,a=this.program(r),f=a.dispatches.length,h=o.createQuerySet({type:"timestamp",count:2*f}),u=o.createBuffer({size:16*f,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),p=o.createBuffer({size:16*f,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),k=o.createCommandEncoder();a.dispatches.forEach((w,q)=>{let b=k.beginComputePass({timestampWrites:{querySet:h,beginningOfPassWriteIndex:2*q,endOfPassWriteIndex:2*q+1}});b.setPipeline(w.pipeline),b.setBindGroup(0,w.bindGroup),b.dispatchWorkgroups(...w.groups),b.end()}),k.resolveQuerySet(h,0,2*f,u,0),k.copyBufferToBuffer(u,0,p,0,16*f),o.queue.submit([k.finish()]),await p.mapAsync(GPUMapMode.READ);let d=new BigUint64Array(p.getMappedRange()),m={};return a.dispatches.forEach((w,q)=>{let b=w.label.replace(/^layer\d+\./,"layer.");m[b]=(m[b]??0)+Number(d[2*q+1]-d[2*q])/1e6}),p.unmap(),h.destroy(),u.destroy(),p.destroy(),m}destroy(){for(let e of Object.values(this.buffers))e.destroy();for(let e of this.programs.values())e.params.destroy();this.staging.destroy(),this.querySet?.destroy(),this.queryBuffer?.destroy(),this.device.destroy()}};export{be as Engine,ne as floatTensor,de as halfToFloat,ie as planFromOnnx,H as readInitializers};
