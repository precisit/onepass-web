/* onepass-webgpu 385727d (MIT), https://github.com/precisit/onepass-webgpu */
var z=o=>`${o?`enable f16;
`:""}alias WT = ${o?"f16":"f32"};
`,j=`
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
`,N=o=>`${z(o)}
struct P { width: u32, seqLen: u32, idsOff: u32, embOff: u32, posOff: u32, lnW: u32, lnB: u32, eps: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> ids: array<i32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${j}
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
`,Q=(o,e)=>`${z(o)}
const RM = ${e.RM}u;
const KS = ${e.KS}u;
struct P { M: u32, N: u32, K: u32, wOff: u32, aBias: u32, p5: u32, p6: u32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<vec4<WT>>;
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
  var wi = p.wOff / 4u + k0 * stride + n4;
  for (var kk = 0u; kk < KS; kk += 1u) {
    let w = vec4<f32>(W[wi]);
    wi += stride;
    for (var r = 0u; r < RM; r += 1u) { acc[r] = fma(vec4<f32>(at[r * KS + kk]), w, acc[r]); }
  }
  let rows = min(RM, p.M - row0);
  for (var r = 0u; r < rows; r += 1u) { Out[(wg.z * p.M + row0 + r) * stride + n4] = acc[r]; }
}
`,X=(o,e)=>`${z(o)}
struct P { M: u32, width: u32, bOff: u32, lnW: u32, lnB: u32, eps: f32, p6: u32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> Part: array<f32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${j}
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
`,V=(o,e,s,r)=>`${z(o)}
const D = ${e}u;
const LM = ${r}u;
const QB = 8u;
struct P { L: u32, idsOff: u32, width: u32, scale: f32, M: u32, bOff: u32, p6: u32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> Part: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<i32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
var<workgroup> ks: array<f32, ${r*e}>;
var<workgroup> vs: array<f32, ${r*e}>;
var<workgroup> qs: array<f32, ${8*e}>;
var<workgroup> sc: array<f32, ${8*r}>;

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
`,H=o=>`${z(o)}
struct P { L: u32, width: u32, idsOff: u32, lnW: u32, lnB: u32, eps: f32, p6: u32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<i32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${j}
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
`,J=(o,e,s)=>`
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
`;var C=class{constructor(e,s=0,r=e.length){this.buf=e;this.end=r;this.pos=s}pos;varint(){let e=0,s=1;for(;;){let r=this.buf[this.pos++];if(e+=(r&127)*s,r<128)return e;s*=128}}skip(e){if(e===0)this.varint();else if(e===1)this.pos+=8;else if(e===2){let s=this.varint();this.pos+=s}else if(e===5)this.pos+=4;else throw new Error(`onnx: unsupported wire type ${e}`)}};function I(o,e,s,r){let t=new C(o,e,s);for(;t.pos<s;){let n=t.varint(),u=Math.floor(n/8),i=n&7,p=t.pos;r(u,i,t),t.pos===p&&t.skip(i)}}function ir(o,e,s){let r={name:"",dims:[],dataType:0,bytes:new Uint8Array(0)},t=null;return I(o,e,s,(n,u,i)=>{if(n===1&&u===0)r.dims.push(i.varint());else if(n===1&&u===2){let p=i.varint()+i.pos;for(;i.pos<p;)r.dims.push(i.varint())}else if(n===2)r.dataType=i.varint();else if(n===8){let p=i.varint();r.name=new TextDecoder().decode(o.subarray(i.pos,i.pos+p)),i.pos+=p}else if(n===9){let p=i.varint();r.bytes=o.subarray(i.pos,i.pos+p),i.pos+=p}else if(n===4&&u===2){let p=i.varint();t=Array.from(new Float32Array(o.slice(i.pos,i.pos+p).buffer)),i.pos+=p}else if(n===14&&i.varint()===1)throw new Error("onnx: external data is not supported yet")}),t&&(r.bytes=new Uint8Array(new Float32Array(t).buffer)),r}function F(o){let e=o instanceof Uint8Array?o:new Uint8Array(o),s=new Map;return I(e,0,e.length,(r,t,n)=>{if(r!==7||t!==2)return;let u=n.varint(),i=n.pos+u;I(e,n.pos,i,(p,l,a)=>{if(p!==5||l!==2)return;let c=a.varint(),w=ir(e,a.pos,a.pos+c);s.set(w.name,w),a.pos+=c}),n.pos=i}),s}function Z(o,e=!1){if(o.dataType!==1)throw new Error(`onnx: ${o.name} is data type ${o.dataType}, expected float32`);let s=new Float32Array(o.bytes.byteLength/4);if(new Uint8Array(s.buffer).set(o.bytes),!e)return s;let[r,t]=o.dims,n=new Float32Array(s.length);for(let u=0;u<r;u+=1)for(let i=0;i<t;i+=1)n[i*r+u]=s[u*t+i];return n}var sr={splitTarget:4096,qkvSplitTarget:4096},D=256;function er(o,e,s,r){let n=Math.ceil(s/256)*64*Math.ceil(o/4),u=1;for(;u*2<=e/32&&n*u<r;)u*=2;return{S:u,RM:4}}function ar(o,e,s){let r=(u,i,p,l=s.splitTarget)=>er(u,i,p,l).S*u*p,t={p1:0,p2:0,pq:r(e*o.option_slots,o.width,o.rank)};for(let u of[e*o.context_len,e*o.option_slots*o.option_len])t.p1=Math.max(t.p1,r(u,o.width,3*o.width,s.qkvSplitTarget),r(u,o.width,o.ff)),t.p2=Math.max(t.p2,r(u,o.width,o.width),r(u,o.ff,o.width));let n=e*o.context_len;return t.p1=Math.max(t.p1,r(n,o.width,o.rank)),t.p2=Math.max(t.p2,r(n,o.width,o.rank)),t}function ur(o){let e=globalThis.Float16Array;if(e){let n=new e(o);return new Uint16Array(n.buffer,n.byteOffset,o.length)}let s=new Uint16Array(o.length),r=new Float32Array(1),t=new Uint32Array(r.buffer);for(let n=0;n<o.length;n+=1){r[0]=o[n];let u=t[0],i=u>>>16&32768,p=u>>>23&255,l=u&8388607;if(p===255){s[n]=i|31744|(l?512:0);continue}let a=p-127+15;if(a>=31){s[n]=i|31744;continue}if(a<=0){if(a<-10){s[n]=i;continue}l|=8388608;let h=14-a,m=l>>>h,g=l&(1<<h)-1,d=1<<h-1;(g>d||g===d&&m&1)&&(m+=1),s[n]=i|m;continue}let c=a<<10|l>>>13,w=l&8191;(w>4096||w===4096&&c&1)&&(c+=1),s[n]=i|c}return s}var rr=class o{constructor(e,s,r,t,n,u,i,p,l){this.device=e;this.adapterInfo=s;this.precision=r;this.maxBatch=t;this.tuning=l;this.config=n.config,this.offsets=i;let a=n.config,c=t,w=c*a.context_len,h=c*a.option_slots*a.option_len,m=Math.max(w,h),g=(b,B=0)=>e.createBuffer({size:Math.max(16,b*4),usage:GPUBufferUsage.STORAGE|B}),d={p1:0,p2:0,pq:0};for(let b=1;b<=c;b+=1){let B=ar(a,b,l);d.p1=Math.max(d.p1,B.p1),d.p2=Math.max(d.p2,B.p2),d.pq=Math.max(d.pq,B.pq)}this.buffers={weights:u,ids:g(c*(a.context_len+a.option_slots*a.option_len+a.option_slots),GPUBufferUsage.COPY_DST),xc:g(w*a.width),yc:g(w*a.width),xo:g(h*a.width),yo:g(h*a.width),att:g(m*a.width),p1:g(d.p1),p2:g(d.p2),pq:g(d.pq),pooled:g(c*a.option_slots*a.width),logits:g(c*a.option_slots,GPUBufferUsage.COPY_SRC)},this.staging=e.createBuffer({size:272,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),c*a.option_slots*4>256&&(this.staging.destroy(),this.staging=e.createBuffer({size:Math.ceil(c*a.option_slots*4/256)*256+16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})),p&&(this.querySet=e.createQuerySet({type:"timestamp",count:2}),this.queryBuffer=e.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}))}config;lastGpuMs=null;programs=new Map;pipelines=new Map;offsets=new Map;buffers={};staging;querySet=null;queryBuffer=null;busy=Promise.resolve();static async load(e,s,r={}){if(e.format!=="onepass-plan/1"||e.architecture!=="onepass-scorer")throw new Error(`unsupported plan ${e.format} / ${e.architecture}`);if(!navigator.gpu)throw new Error("WebGPU is not available in this browser");let t=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!t)throw new Error("no WebGPU adapter");let n=t.info;if((/swiftshader|llvmpipe|software|basic render/i.test(`${n.vendor} ${n.architecture} ${n.description}`)||t.isFallbackAdapter===!0)&&!r.allowSoftware)throw new Error(`software WebGPU adapter (${n.vendor} ${n.architecture})`);let i=r.precision??"f32";if(i==="f16"&&!t.features.has("shader-f16"))throw new Error("this GPU has no shader-f16");let p=!!r.gpuTiming&&t.features.has("timestamp-query"),l=[];i==="f16"&&l.push("shader-f16"),p&&l.push("timestamp-query");let a=await t.requestDevice({requiredFeatures:l,requiredLimits:{maxStorageBufferBindingSize:t.limits.maxStorageBufferBindingSize,maxBufferSize:t.limits.maxBufferSize}}),c=F(s),w=[],h=new Map,m=0;for(let[q,_]of Object.entries(e.tensors)){let O=c.get(_.initializer);if(!O)throw new Error(`the ONNX file has no initializer ${_.initializer} (for ${q})`);let A=Z(O,_.transpose),R=_.shape.reduce((T,E)=>T*E,1);if(A.length!==R)throw new Error(`${q}: ${A.length} values, plan says ${R}`);h.set(q,m),w.push(A),m+=Math.ceil(A.length/64)*64}let g=new Float32Array(m);for(let[q,_]of w.entries())g.set(_,h.get(Object.keys(e.tensors)[q]));let d=i==="f16"?ur(g):g,b=a.createBuffer({size:d.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});a.queue.writeBuffer(b,0,d.buffer,d.byteOffset,d.byteLength);let B={...sr,...r.tuning},G=new o(a,n,i,r.maxBatch??1,e,b,h,p,B);return G.program(1),await a.queue.onSubmittedWorkDone(),G}pipeline(e,s){let r=this.pipelines.get(e);return r||(r=this.device.createComputePipeline({layout:"auto",compute:{module:this.device.createShaderModule({code:s()}),entryPoint:"main"}}),this.pipelines.set(e,r)),r}off(e){let s=this.offsets.get(e);if(s===void 0)throw new Error(`plan has no tensor ${e}`);return s}program(e){let s=this.programs.get(e);if(s)return s;if(e<1||e>this.maxBatch)throw new Error(`batch ${e} outside 1..${this.maxBatch}`);let r=this.config,t=this.buffers,n=this.precision==="f16",u=[],i=[],p="",l=(f,v,x,P,U)=>{let k=new ArrayBuffer(32),$=new Uint32Array(k),y=new Float32Array(k);v.forEach((M,S)=>{$[S]=M});for(let[M,S]of Object.entries(x))y[Number(M)]=S;u.push(k),i.push({label:p,pipeline:f,buffers:P,groups:U})},a=0,c=e*r.context_len,w=c+e*r.option_slots*r.option_len,h=e*r.context_len,m=e*r.option_slots,g=m*r.option_len,d=this.pipeline(`embed${n}`,()=>N(n));p="embed",l(d,[r.width,r.context_len,a,this.off("embedding"),this.off("pos_context"),this.off("layer0.norm1.w"),this.off("layer0.norm1.b")],{7:r.eps},[t.weights,t.ids,t.xc,t.yc],[h,1,1]),l(d,[r.width,r.option_len,c,this.off("embedding"),this.off("pos_option"),this.off("option_layer.norm1.w"),this.off("option_layer.norm1.b")],{7:r.eps},[t.weights,t.ids,t.xo,t.yo],[g,1,1]);let b=(f,v,x,P,U,k,$=0,y=null,M=!1,S=this.tuning.splitTarget)=>{let{S:W,RM:L}=er(f,v,x,S),Y={RM:L,KS:v/W,aSplits:$,aRelu:M},nr=this.pipeline(`mm${n}${JSON.stringify(Y)}`,()=>Q(n,Y));return l(nr,[f,x,v,this.off(k),y?this.off(y):0],{},[t.weights,P,U],[Math.ceil(x/256),Math.ceil(f/L),W]),W},B=(f,v,x,P,U,k,$)=>{let y=this.pipeline(`rn${n}${x}`,()=>X(n,x));l(y,[f,r.width,this.off(P),this.off(`${$}.w`),this.off(`${$}.b`)],{5:r.eps},[t.weights,v,U,k],[f,1,1])},G=r.width/r.heads,q=Math.ceil(Math.max(r.context_len,r.option_len)/8)*8;if(q*G*2*4>13e3)throw new Error("sequence too long for the attention kernel");let _=(f,v,x,P,U,k,$)=>{let y=U*k;p=`${f}.qkv`;let M=b(y,r.width,3*r.width,P,t.p1,`${f}.qkv.w`,0,null,!1,this.tuning.qkvSplitTarget);p=`${f}.attention`,l(this.pipeline(`att${n}${M}`,()=>V(n,G,M,q)),[k,$,r.width,0,y,this.off(`${f}.qkv.b`)],{3:1/Math.sqrt(G)},[t.weights,t.p1,t.ids,t.att],[r.heads,U,Math.ceil(k/8)]),p=`${f}.out`;let S=b(y,r.width,r.width,t.att,t.p2,`${f}.out.w`);B(y,t.p2,S,`${f}.out.b`,x,P,`${f}.norm2`),p=`${f}.ff1`;let W=b(y,r.width,r.ff,P,t.p1,`${f}.ff1.w`);p=`${f}.ff2`;let L=b(y,r.ff,r.width,t.p1,t.p2,`${f}.ff2.w`,W,`${f}.ff1.b`,!0);B(y,t.p2,L,`${f}.ff2.b`,x,P,v)};for(let f=0;f<r.layers;f+=1)_(`layer${f}`,f+1<r.layers?`layer${f+1}.norm1`:"head.context_norm",t.xc,t.yc,e,r.context_len,a);_("option_layer","option_layer.norm1",t.xo,t.yo,m,r.option_len,c),p="pool",l(this.pipeline(`pool${n}`,()=>H(n)),[r.option_len,r.width,c,this.off("head.option_norm.w"),this.off("head.option_norm.b")],{5:r.eps},[t.weights,t.xo,t.ids,t.pooled],[m,1,1]),p="head.qkv";let O=b(m,r.width,r.rank,t.pooled,t.pq,"head.q.w"),A=b(h,r.width,r.rank,t.yc,t.p1,"head.k.w"),R=b(h,r.width,r.rank,t.yc,t.p2,"head.v.w");p="head",l(this.pipeline(`head${O}${A}${R}`,()=>J(O,A,R)),[r.context_len,r.option_slots,r.rank,a,w,0,m,h],{5:1/Math.sqrt(r.rank)},[t.pq,t.p1,t.p2,t.ids,t.logits],[r.option_slots,e,1]);let T=this.device.createBuffer({size:u.length*D,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),E=new Uint8Array(u.length*D);u.forEach((f,v)=>E.set(new Uint8Array(f),v*D)),this.device.queue.writeBuffer(T,0,E);let tr=i.map((f,v)=>({label:f.label,pipeline:f.pipeline,groups:f.groups,bindGroup:this.device.createBindGroup({layout:f.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:T,offset:v*D,size:32}},...f.buffers.map((x,P)=>({binding:P+1,resource:{buffer:x}}))]})})),K={batch:e,dispatches:tr,params:T};return this.programs.set(e,K),K}wake(e=1){let s=this.busy.then(async()=>{let r=this.program(e),t=this.device.createCommandEncoder(),n=t.beginComputePass();for(let u of r.dispatches)n.setPipeline(u.pipeline),n.setBindGroup(0,u.bindGroup),n.dispatchWorkgroups(...u.groups);n.end(),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone()});return this.busy=s.catch(()=>{}),s}dispatchCount(e=1){return this.program(e).dispatches.length}score(e,s,r,t=1){let n=this.busy.then(()=>this.run(e,s,r,t));return this.busy=n.catch(()=>{}),n}async run(e,s,r,t){let n=this.config,u=this.program(t),i=t*n.context_len,p=t*n.option_slots*n.option_len,l=t*n.option_slots;if(e.length!==i||s.length!==p||r.length!==l)throw new Error(`input sizes ${e.length}/${s.length}/${r.length}, expected ${i}/${p}/${l}`);let a=new Int32Array(i+p+l);a.set(e,0),a.set(s,i),a.set(r,i+p);let{device:c}=this;c.queue.writeBuffer(this.buffers.ids,0,a);let w=c.createCommandEncoder(),h=w.beginComputePass(this.querySet?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:void 0);for(let b of u.dispatches)h.setPipeline(b.pipeline),h.setBindGroup(0,b.bindGroup),h.dispatchWorkgroups(...b.groups);h.end();let m=l*4,g=Math.ceil(m/256)*256;w.copyBufferToBuffer(this.buffers.logits,0,this.staging,0,m),this.querySet&&this.queryBuffer&&(w.resolveQuerySet(this.querySet,0,2,this.queryBuffer,0),w.copyBufferToBuffer(this.queryBuffer,0,this.staging,g,16)),c.queue.submit([w.finish()]),await this.staging.mapAsync(GPUMapMode.READ);let d=new Float32Array(this.staging.getMappedRange(0,m).slice(0));if(this.querySet){let b=new BigUint64Array(this.staging.getMappedRange(g,16));this.lastGpuMs=Number(b[1]-b[0])/1e6}return this.staging.unmap(),d}async profile(e,s,r,t=1){if(!this.querySet)throw new Error("profile() needs gpuTiming and an adapter with timestamp-query");await this.score(e,s,r,t);let{device:n}=this,u=this.program(t),i=u.dispatches.length,p=n.createQuerySet({type:"timestamp",count:2*i}),l=n.createBuffer({size:16*i,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),a=n.createBuffer({size:16*i,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),c=n.createCommandEncoder();u.dispatches.forEach((m,g)=>{let d=c.beginComputePass({timestampWrites:{querySet:p,beginningOfPassWriteIndex:2*g,endOfPassWriteIndex:2*g+1}});d.setPipeline(m.pipeline),d.setBindGroup(0,m.bindGroup),d.dispatchWorkgroups(...m.groups),d.end()}),c.resolveQuerySet(p,0,2*i,l,0),c.copyBufferToBuffer(l,0,a,0,16*i),n.queue.submit([c.finish()]),await a.mapAsync(GPUMapMode.READ);let w=new BigUint64Array(a.getMappedRange()),h={};return u.dispatches.forEach((m,g)=>{let d=m.label.replace(/^layer\d+\./,"layer.");h[d]=(h[d]??0)+Number(w[2*g+1]-w[2*g])/1e6}),a.unmap(),p.destroy(),l.destroy(),a.destroy(),h}destroy(){for(let e of Object.values(this.buffers))e.destroy();for(let e of this.programs.values())e.params.destroy();this.staging.destroy(),this.querySet?.destroy(),this.queryBuffer?.destroy(),this.device.destroy()}};export{rr as Engine,F as readInitializers};
