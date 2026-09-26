/* onepass-webgpu 29a830c (MIT), https://github.com/precisit/onepass-webgpu */
var j=o=>`${o?`enable f16;
`:""}alias WT = ${o?"f16":"f32"};
`,F=`
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
`,V=o=>`${j(o)}
struct P { width: u32, seqLen: u32, idsOff: u32, embOff: u32, posOff: u32, lnW: u32, lnB: u32, eps: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> ids: array<i32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${F}
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
`,se=o=>o==="int8"?`
fn unpack8(q: u32) -> vec4<f32> {
  let s = bitcast<i32>(q);
  return vec4<f32>(vec4<i32>((s << 24u) >> 24u, (s << 16u) >> 24u, (s << 8u) >> 24u, s >> 24u));
}`:`
fn unpack8(q: u32) -> vec4<f32> {
  return vec4<f32>(f32(q & 255u), f32((q >> 8u) & 255u), f32((q >> 16u) & 255u), f32(q >> 24u));
}`,H=(o,r)=>`${j(o)}
const RM = ${r.RM}u;
const KS = ${r.KS}u;
struct P { M: u32, N: u32, K: u32, wOff: u32, aBias: u32, zp: f32, scale: f32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
${r.w8?`@group(0) @binding(1) var<storage, read> Q: array<u32>;
${r.aSplits?"@group(0) @binding(4) var<storage, read> W: array<vec4<WT>>;":""}
${se(r.w8)}`:"@group(0) @binding(1) var<storage, read> W: array<vec4<WT>>;"}
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<vec4<f32>>;
var<workgroup> at: array<f32, ${r.RM*r.KS}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let row0 = wg.y * RM;
  let k0 = wg.z * KS;
  for (var e = t; e < RM * KS; e += 64u) {
    let m = row0 + e / KS;
    let k = k0 + e % KS;
    var a = 0.0;
    if (m < p.M) {
${r.aSplits===0?"      a = A[m * p.K + k];":`      for (var j = 0u; j < ${r.aSplits}u; j += 1u) { a += A[(j * p.M + m) * p.K + k]; }
      a += f32(W[(p.aBias + k) / 4u][k % 4u]);${r.aRelu?`
      a = max(a, 0.0);`:""}`}
    }
    at[e] = a;
  }
  workgroupBarrier();
  let n4 = wg.x * 64u + t;
  if (n4 * 4u >= p.N) { return; }
  let stride = p.N / 4u;
  var acc: array<vec4<f32>, RM>;
  var wi = ${r.w8?"p.wOff":"p.wOff / 4u"} + k0 * stride + n4;
  for (var kk = 0u; kk < KS; kk += 1u) {
    let w = ${r.w8?"unpack8(Q[wi]) - vec4<f32>(p.zp)":"vec4<f32>(W[wi])"};
    wi += stride;
    for (var r = 0u; r < RM; r += 1u) { acc[r] = fma(vec4<f32>(at[r * KS + kk]), w, acc[r]); }
  }
  let rows = min(RM, p.M - row0);
  for (var r = 0u; r < rows; r += 1u) { Out[(wg.z * p.M + row0 + r) * stride + n4] = acc[r]${r.w8?" * p.scale":""}; }
}
`,J=(o,r)=>`${j(o)}
struct P { M: u32, width: u32, bOff: u32, lnW: u32, lnB: u32, eps: f32, p6: u32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> Part: array<f32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${F}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let m = wg.x + wg.y * 65535u;
  var x = 0.0;
  if (t < p.width) {
    var y = 0.0;
    for (var j = 0u; j < ${r}u; j += 1u) { y += Part[(j * p.M + m) * p.width + t]; }
    x = X[m * p.width + t] + (y + f32(W[p.bOff + t]));
    X[m * p.width + t] = x;
  }
  let mean = wsum(select(0.0, x, t < p.width), t) / f32(p.width);
  let d = select(0.0, x - mean, t < p.width);
  let inv = 1.0 / sqrt(wsum(d * d, t) / f32(p.width) + p.eps);
  if (t < p.width) { Y[m * p.width + t] = d * inv * f32(W[p.lnW + t]) + f32(W[p.lnB + t]); }
}
`,Z=(o,r,s,e)=>`${j(o)}
const D = ${r}u;
const LM = ${e}u;
const QB = 8u;
struct P { L: u32, idsOff: u32, width: u32, scale: f32, M: u32, bOff: u32, p6: u32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> Part: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<i32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
var<workgroup> ks: array<f32, ${e*r}>;
var<workgroup> vs: array<f32, ${e*r}>;
var<workgroup> qs: array<f32, ${8*r}>;
var<workgroup> sc: array<f32, ${8*e}>;

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
`,ee=o=>`${j(o)}
struct P { L: u32, width: u32, idsOff: u32, lnW: u32, lnB: u32, eps: f32, p6: u32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<i32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${F}
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
`,re=(o,r,s)=>`
struct P { Lc: u32, slots: u32, rank: u32, ctxOff: u32, maskOff: u32, scale: f32, Mq: u32, Mc: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> ids: array<i32>;
@group(0) @binding(5) var<storage, read_write> logits: array<f32>;
var<workgroup> qv: array<f32, 1024>;
var<workgroup> sc: array<f32, 64>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let n = wg.x;
  let b = wg.y;
  let qrow = b * p.slots + n;
  let krow = b * p.Lc;
  for (var r = t; r < p.rank; r += 256u) {
    var a = 0.0;
    for (var j = 0u; j < ${o}u; j += 1u) { a += q[(j * p.Mq + qrow) * p.rank + r]; }
    qv[r] = a;
  }
  workgroupBarrier();
  // scores: 4 threads per context position
  let l = t / 4u;
  let part = t % 4u;
  var dot = 0.0;
  if (l < p.Lc) {
    for (var r = part; r < p.rank; r += 4u) {
      var kk = 0.0;
      for (var j = 0u; j < ${r}u; j += 1u) { kk += k[(j * p.Mc + krow + l) * p.rank + r]; }
      dot += qv[r] * kk;
    }
  }
  red[t] = dot;
  workgroupBarrier();
  if (part == 0u && l < p.Lc) {
    sc[l] = (red[t] + red[t + 1u] + red[t + 2u] + red[t + 3u]) * p.scale;
  }
  workgroupBarrier();
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
`;var K=class{constructor(r,s=0,e=r.length){this.buf=r;this.end=e;this.pos=s}pos;varint(){let r=0,s=1;for(;;){let e=this.buf[this.pos++];if(r+=(e&127)*s,e<128)return r;s*=128}}skip(r){if(r===0)this.varint();else if(r===1)this.pos+=8;else if(r===2){let s=this.varint();this.pos+=s}else if(r===5)this.pos+=4;else throw new Error(`onnx: unsupported wire type ${r}`)}};function Y(o,r,s,e){let t=new K(o,r,s);for(;t.pos<s;){let n=t.varint(),a=Math.floor(n/8),i=n&7,u=t.pos;e(a,i,t),t.pos===u&&t.skip(i)}}function ue(o,r,s){let e={name:"",dims:[],dataType:0,bytes:new Uint8Array(0)},t=null;return Y(o,r,s,(n,a,i)=>{if(n===1&&a===0)e.dims.push(i.varint());else if(n===1&&a===2){let u=i.varint()+i.pos;for(;i.pos<u;)e.dims.push(i.varint())}else if(n===2)e.dataType=i.varint();else if(n===8){let u=i.varint();e.name=new TextDecoder().decode(o.subarray(i.pos,i.pos+u)),i.pos+=u}else if(n===9){let u=i.varint();e.bytes=o.subarray(i.pos,i.pos+u),i.pos+=u}else if(n===4&&a===2){let u=i.varint();t=Array.from(new Float32Array(o.slice(i.pos,i.pos+u).buffer)),i.pos+=u}else if(n===14&&i.varint()===1)throw new Error("onnx: external data is not supported yet")}),t&&(e.bytes=new Uint8Array(new Float32Array(t).buffer)),e}function N(o){let r=o instanceof Uint8Array?o:new Uint8Array(o),s=new Map;return Y(r,0,r.length,(e,t,n)=>{if(e!==7||t!==2)return;let a=n.varint(),i=n.pos+a;Y(r,n.pos,i,(u,l,c)=>{if(u!==5||l!==2)return;let w=c.varint(),p=ue(r,c.pos,c.pos+w);s.set(p.name,p),c.pos+=w}),n.pos=i}),s}function te(o,r=!1){if(o.dataType!==1)throw new Error(`onnx: ${o.name} is data type ${o.dataType}, expected float32`);let s=new Float32Array(o.bytes.byteLength/4);if(new Uint8Array(s.buffer).set(o.bytes),!r)return s;let[e,t]=o.dims,n=new Float32Array(s.length);for(let a=0;a<e;a+=1)for(let i=0;i<t;i+=1)n[i*e+a]=s[a*t+i];return n}function ne(o,r,s){if(o.dataType!==2&&o.dataType!==3)throw new Error(`onnx: ${o.name} is not int8 or uint8`);let e=o.dataType===3?new Int8Array(o.bytes.buffer,o.bytes.byteOffset,o.bytes.byteLength):o.bytes,t=new Float32Array(e.length);for(let n=0;n<e.length;n+=1)t[n]=Math.fround(Math.fround(e[n]-s)*r);return t}var fe={splitTarget:4096,qkvSplitTarget:4096},I=256;function ie(o,r,s,e){let n=Math.ceil(s/256)*64*Math.ceil(o/4),a=1;for(;a*2<=r/32&&n*a<e;)a*=2;return{S:a,RM:4}}function pe(o,r,s){let e=(a,i,u,l=s.splitTarget)=>ie(a,i,u,l).S*a*u,t={p1:0,p2:0,pq:e(r*o.option_slots,o.width,o.rank)};for(let a of[r*o.context_len,r*o.option_slots*o.option_len])t.p1=Math.max(t.p1,e(a,o.width,3*o.width,s.qkvSplitTarget),e(a,o.width,o.ff)),t.p2=Math.max(t.p2,e(a,o.width,o.width),e(a,o.ff,o.width));let n=r*o.context_len;return t.p1=Math.max(t.p1,e(n,o.width,o.rank)),t.p2=Math.max(t.p2,e(n,o.width,o.rank)),t}function le(o){let r=globalThis.Float16Array;if(r){let n=new r(o);return new Uint16Array(n.buffer,n.byteOffset,o.length)}let s=new Uint16Array(o.length),e=new Float32Array(1),t=new Uint32Array(e.buffer);for(let n=0;n<o.length;n+=1){e[0]=o[n];let a=t[0],i=a>>>16&32768,u=a>>>23&255,l=a&8388607;if(u===255){s[n]=i|31744|(l?512:0);continue}let c=u-127+15;if(c>=31){s[n]=i|31744;continue}if(c<=0){if(c<-10){s[n]=i;continue}l|=8388608;let g=14-c,h=l>>>g,y=l&(1<<g)-1,b=1<<g-1;(y>b||y===b&&h&1)&&(h+=1),s[n]=i|h;continue}let w=c<<10|l>>>13,p=l&8191;(p>4096||p===4096&&w&1)&&(w+=1),s[n]=i|w}return s}var oe=class o{constructor(r,s,e,t,n,a,i,u,l,c=null,w=new Map){this.device=r;this.adapterInfo=s;this.precision=e;this.maxBatch=t;this.tuning=l;this.packed=w;this.config=n.config,this.offsets=i;let p=n.config,g=t,h=g*p.context_len,y=g*p.option_slots*p.option_len,b=Math.max(h,y),d=(B,M=0)=>r.createBuffer({size:Math.max(16,B*4),usage:GPUBufferUsage.STORAGE|M}),v={p1:0,p2:0,pq:0};for(let B=1;B<=g;B+=1){let M=pe(p,B,l);v.p1=Math.max(v.p1,M.p1),v.p2=Math.max(v.p2,M.p2),v.pq=Math.max(v.pq,M.pq)}this.buffers={weights:a,weights8:c??d(4),ids:d(g*(p.context_len+p.option_slots*p.option_len+p.option_slots),GPUBufferUsage.COPY_DST),xc:d(h*p.width),yc:d(h*p.width),xo:d(y*p.width),yo:d(y*p.width),att:d(b*p.width),p1:d(v.p1),p2:d(v.p2),pq:d(v.pq),pooled:d(g*p.option_slots*p.width),logits:d(g*p.option_slots,GPUBufferUsage.COPY_SRC)},this.staging=r.createBuffer({size:272,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),g*p.option_slots*4>256&&(this.staging.destroy(),this.staging=r.createBuffer({size:Math.ceil(g*p.option_slots*4/256)*256+16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})),u&&(this.querySet=r.createQuerySet({type:"timestamp",count:2}),this.queryBuffer=r.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}))}config;get weightFormat(){return this.packed.size?`int8 (${this.precision} for the rest)`:this.precision}get weightBytes(){return this.buffers.weights.size+(this.packed.size?this.buffers.weights8.size:0)}lastGpuMs=null;programs=new Map;pipelines=new Map;offsets=new Map;buffers={};staging;querySet=null;queryBuffer=null;busy=Promise.resolve();static async load(r,s,e={}){if(r.format!=="onepass-plan/1"||r.architecture!=="onepass-scorer")throw new Error(`unsupported plan ${r.format} / ${r.architecture}`);if(!navigator.gpu)throw new Error("WebGPU is not available in this browser");let t=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!t)throw new Error("no WebGPU adapter");let n=t.info;if((/swiftshader|llvmpipe|software|basic render/i.test(`${n.vendor} ${n.architecture} ${n.description}`)||t.isFallbackAdapter===!0)&&!e.allowSoftware)throw new Error(`software WebGPU adapter (${n.vendor} ${n.architecture})`);let i=e.precision??"f32";if(i==="f16"&&!t.features.has("shader-f16"))throw new Error("this GPU has no shader-f16");let u=!!e.gpuTiming&&t.features.has("timestamp-query"),l=[];i==="f16"&&l.push("shader-f16"),u&&l.push("timestamp-query");let c=await t.requestDevice({requiredFeatures:l,requiredLimits:{maxStorageBufferBindingSize:t.limits.maxStorageBufferBindingSize,maxBufferSize:t.limits.maxBufferSize}}),w=N(s),p=[],g=new Map,h=new Map,y=[],b=0,d=0;for(let[x,m]of Object.entries(r.tensors)){let _=w.get(m.initializer);if(!_)throw new Error(`the ONNX file has no initializer ${m.initializer} (for ${x})`);let W=m.shape.reduce((f,P)=>f*P,1);if(m.quant&&x!=="embedding"){if(m.transpose||m.shape.length!==2||m.shape[1]%4!==0)throw new Error(`${x}: unsupported 8-bit layout`);if(_.bytes.byteLength!==W)throw new Error(`${x}: ${_.bytes.byteLength} bytes, plan says ${W}`);h.set(x,{offsetWords:d/4,kind:m.quant.dtype,scale:m.quant.scale,zeroPoint:m.quant.zero_point}),y.push(_.bytes),d+=Math.ceil(_.bytes.byteLength/256)*256;continue}let O=m.quant?ne(_,m.quant.scale,m.quant.zero_point):te(_,m.transpose);if(O.length!==W)throw new Error(`${x}: ${O.length} values, plan says ${W}`);g.set(x,b),p.push({name:x,data:O}),b+=Math.ceil(O.length/64)*64}let v=new Float32Array(b);for(let{name:x,data:m}of p)v.set(m,g.get(x));let B=i==="f16"?le(v):v,M=c.createBuffer({size:B.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});c.queue.writeBuffer(M,0,B.buffer,B.byteOffset,B.byteLength);let z=null;if(d>0){z=c.createBuffer({size:d,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});let x=0;for(let m of y){let _=new Uint8Array(Math.ceil(m.byteLength/4)*4);_.set(m),c.queue.writeBuffer(z,x,_),x+=Math.ceil(m.byteLength/256)*256}}let C={...fe,...e.tuning},E=new o(c,n,i,e.maxBatch??1,r,M,g,u,C,z,h);return E.program(1),await c.queue.onSubmittedWorkDone(),E}pipeline(r,s){let e=this.pipelines.get(r);return e||(e=this.device.createComputePipeline({layout:"auto",compute:{module:this.device.createShaderModule({code:s()}),entryPoint:"main"}}),this.pipelines.set(r,e)),e}off(r){let s=this.offsets.get(r);if(s===void 0)throw new Error(`plan has no tensor ${r}`);return s}program(r){let s=this.programs.get(r);if(s)return s;if(r<1||r>this.maxBatch)throw new Error(`batch ${r} outside 1..${this.maxBatch}`);let e=this.config,t=this.buffers,n=this.precision==="f16",a=[],i=[],u="",l=(f,P,q,U,A)=>{let $=new ArrayBuffer(32),G=new Uint32Array($),k=new Float32Array($);P.forEach((S,T)=>{G[T]=S});for(let[S,T]of Object.entries(q))k[Number(S)]=T;a.push($),i.push({label:u,pipeline:f,buffers:U,groups:A})},c=0,w=r*e.context_len,p=w+r*e.option_slots*e.option_len,g=r*e.context_len,h=r*e.option_slots,y=h*e.option_len,b=this.pipeline(`embed${n}`,()=>V(n));u="embed",l(b,[e.width,e.context_len,c,this.off("embedding"),this.off("pos_context"),this.off("layer0.norm1.w"),this.off("layer0.norm1.b")],{7:e.eps},[t.weights,t.ids,t.xc,t.yc],[g,1,1]),l(b,[e.width,e.option_len,w,this.off("embedding"),this.off("pos_option"),this.off("option_layer.norm1.w"),this.off("option_layer.norm1.b")],{7:e.eps},[t.weights,t.ids,t.xo,t.yo],[y,1,1]);let d=(f,P,q,U,A,$,G=0,k=null,S=!1,T=this.tuning.splitTarget)=>{let{S:R,RM:D}=ie(f,P,q,T),L=this.packed.get($),Q={RM:D,KS:P/R,aSplits:G,aRelu:S,...L?{w8:L.kind}:{}},X=this.pipeline(`mm${n}${JSON.stringify(Q)}`,()=>H(n,Q));return L?l(X,[f,q,P,L.offsetWords,k?this.off(k):0],{5:L.zeroPoint,6:L.scale},[t.weights8,U,A,...G?[t.weights]:[]],[Math.ceil(q/256),Math.ceil(f/D),R]):l(X,[f,q,P,this.off($),k?this.off(k):0],{},[t.weights,U,A],[Math.ceil(q/256),Math.ceil(f/D),R]),R},v=(f,P,q,U,A,$,G)=>{let k=this.pipeline(`rn${n}${q}`,()=>J(n,q));l(k,[f,e.width,this.off(U),this.off(`${G}.w`),this.off(`${G}.b`)],{5:e.eps},[t.weights,P,A,$],[f,1,1])},B=e.width/e.heads,M=Math.ceil(Math.max(e.context_len,e.option_len)/8)*8;if(M*B*2*4>13e3)throw new Error("sequence too long for the attention kernel");let z=(f,P,q,U,A,$,G)=>{let k=A*$;u=`${f}.qkv`;let S=d(k,e.width,3*e.width,U,t.p1,`${f}.qkv.w`,0,null,!1,this.tuning.qkvSplitTarget);u=`${f}.attention`,l(this.pipeline(`att${n}${S}`,()=>Z(n,B,S,M)),[$,G,e.width,0,k,this.off(`${f}.qkv.b`)],{3:1/Math.sqrt(B)},[t.weights,t.p1,t.ids,t.att],[e.heads,A,Math.ceil($/8)]),u=`${f}.out`;let T=d(k,e.width,e.width,t.att,t.p2,`${f}.out.w`);v(k,t.p2,T,`${f}.out.b`,q,U,`${f}.norm2`),u=`${f}.ff1`;let R=d(k,e.width,e.ff,U,t.p1,`${f}.ff1.w`);u=`${f}.ff2`;let D=d(k,e.ff,e.width,t.p1,t.p2,`${f}.ff2.w`,R,`${f}.ff1.b`,!0);v(k,t.p2,D,`${f}.ff2.b`,q,U,P)};for(let f=0;f<e.layers;f+=1)z(`layer${f}`,f+1<e.layers?`layer${f+1}.norm1`:"head.context_norm",t.xc,t.yc,r,e.context_len,c);z("option_layer","option_layer.norm1",t.xo,t.yo,h,e.option_len,w),u="pool",l(this.pipeline(`pool${n}`,()=>ee(n)),[e.option_len,e.width,w,this.off("head.option_norm.w"),this.off("head.option_norm.b")],{5:e.eps},[t.weights,t.xo,t.ids,t.pooled],[h,1,1]),u="head.qkv";let C=d(h,e.width,e.rank,t.pooled,t.pq,"head.q.w"),E=d(g,e.width,e.rank,t.yc,t.p1,"head.k.w"),x=d(g,e.width,e.rank,t.yc,t.p2,"head.v.w");u="head",l(this.pipeline(`head${C}${E}${x}`,()=>re(C,E,x)),[e.context_len,e.option_slots,e.rank,c,p,0,h,g],{5:1/Math.sqrt(e.rank)},[t.pq,t.p1,t.p2,t.ids,t.logits],[e.option_slots,r,1]);let m=this.device.createBuffer({size:a.length*I,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),_=new Uint8Array(a.length*I);a.forEach((f,P)=>_.set(new Uint8Array(f),P*I)),this.device.queue.writeBuffer(m,0,_);let W=i.map((f,P)=>({label:f.label,pipeline:f.pipeline,groups:f.groups,bindGroup:this.device.createBindGroup({layout:f.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:m,offset:P*I,size:32}},...f.buffers.map((q,U)=>({binding:U+1,resource:{buffer:q}}))]})})),O={batch:r,dispatches:W,params:m};return this.programs.set(r,O),O}wake(r=1){let s=this.busy.then(async()=>{let e=this.program(r),t=this.device.createCommandEncoder(),n=t.beginComputePass();for(let a of e.dispatches)n.setPipeline(a.pipeline),n.setBindGroup(0,a.bindGroup),n.dispatchWorkgroups(...a.groups);n.end(),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone()});return this.busy=s.catch(()=>{}),s}dispatchCount(r=1){return this.program(r).dispatches.length}score(r,s,e,t=1){let n=this.busy.then(()=>this.run(r,s,e,t));return this.busy=n.catch(()=>{}),n}async run(r,s,e,t){let n=this.config,a=this.program(t),i=t*n.context_len,u=t*n.option_slots*n.option_len,l=t*n.option_slots;if(r.length!==i||s.length!==u||e.length!==l)throw new Error(`input sizes ${r.length}/${s.length}/${e.length}, expected ${i}/${u}/${l}`);let c=new Int32Array(i+u+l);c.set(r,0),c.set(s,i),c.set(e,i+u);let{device:w}=this;w.queue.writeBuffer(this.buffers.ids,0,c);let p=w.createCommandEncoder(),g=p.beginComputePass(this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:void 0);for(let d of a.dispatches)g.setPipeline(d.pipeline),g.setBindGroup(0,d.bindGroup),g.dispatchWorkgroups(...d.groups);g.end();let h=l*4,y=Math.ceil(h/256)*256;p.copyBufferToBuffer(this.buffers.logits,0,this.staging,0,h),this.querySet&&this.queryBuffer&&(p.resolveQuerySet(this.querySet,0,2,this.queryBuffer,0),p.copyBufferToBuffer(this.queryBuffer,0,this.staging,y,16)),w.queue.submit([p.finish()]),await this.staging.mapAsync(GPUMapMode.READ);let b=new Float32Array(this.staging.getMappedRange(0,h).slice(0));if(this.querySet){let d=new BigUint64Array(this.staging.getMappedRange(y,16));this.lastGpuMs=Number(d[1]-d[0])/1e6}return this.staging.unmap(),b}async profile(r,s,e,t=1){if(!this.querySet)throw new Error("profile() needs gpuTiming and an adapter with timestamp-query");await this.score(r,s,e,t);let{device:n}=this,a=this.program(t),i=a.dispatches.length,u=n.createQuerySet({type:"timestamp",count:2*i}),l=n.createBuffer({size:16*i,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),c=n.createBuffer({size:16*i,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),w=n.createCommandEncoder();a.dispatches.forEach((h,y)=>{let b=w.beginComputePass({timestampWrites:{querySet:u,beginningOfPassWriteIndex:2*y,endOfPassWriteIndex:2*y+1}});b.setPipeline(h.pipeline),b.setBindGroup(0,h.bindGroup),b.dispatchWorkgroups(...h.groups),b.end()}),w.resolveQuerySet(u,0,2*i,l,0),w.copyBufferToBuffer(l,0,c,0,16*i),n.queue.submit([w.finish()]),await c.mapAsync(GPUMapMode.READ);let p=new BigUint64Array(c.getMappedRange()),g={};return a.dispatches.forEach((h,y)=>{let b=h.label.replace(/^layer\d+\./,"layer.");g[b]=(g[b]??0)+Number(p[2*y+1]-p[2*y])/1e6}),c.unmap(),u.destroy(),l.destroy(),c.destroy(),g}destroy(){for(let r of Object.values(this.buffers))r.destroy();for(let r of this.programs.values())r.params.destroy();this.staging.destroy(),this.querySet?.destroy(),this.queryBuffer?.destroy(),this.device.destroy()}};export{oe as Engine,N as readInitializers};
