// Reference dump: runs the OFFICIAL demo's GaussianLUT class verbatim.
// The class body is spliced out of glut_editor.html lines 435..611 (class head
// through the end of `forward`), then closed with `}`.  Nothing else of the
// page is executed, and not one character of the spliced region is edited.
import fs from "node:fs";
import crypto from "node:crypto";

const HTML = "glut_editor.html";
const raw = fs.readFileSync(HTML, "utf8");
const sha = crypto.createHash("sha256").update(fs.readFileSync(HTML)).digest("hex");
const lines = raw.split("\n");

// 1-indexed inclusive [435, 611]
const CLASS_FROM = 435, CLASS_TO = 611;
const classSrc = lines.slice(CLASS_FROM - 1, CLASS_TO).join("\n") + "\n}\n";
if (!classSrc.startsWith("    class GaussianLUT {")) throw new Error("class head moved");
if (!/Math\.max\(0, Math\.min\(1, result\[2\]\)\)/.test(classSrc)) throw new Error("forward tail moved");

// EMBEDDED_MODELS is the single JSON constant on line 429.
const modelsLine = lines[429 - 1];
const jsonText = modelsLine.replace(/^\s*const EMBEDDED_MODELS = /, "").replace(/;\s*$/, "");
const EMBEDDED_MODELS = JSON.parse(jsonText);

const GaussianLUT = new Function(`${classSrc}\nreturn GaussianLUT;`)();

// ---- query colours: 8 corners + 4^3 grid + 184 seeded uniform = 256 --------
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const queries = [];
for (const r of [0, 1]) for (const g of [0, 1]) for (const b of [0, 1]) queries.push([r, g, b]);
for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++)
  queries.push([i / 3, j / 3, k / 3]);
const rnd = mulberry32(20260815);
while (queries.length < 256) queries.push([rnd(), rnd(), rnd()]);

const out = {
  source_url: "https://color.cvc.uab.cat/assets/html/glut_editor.html",
  source_sha256: sha,
  source_lines: { embedded_models: 429, class_from: CLASS_FROM, class_to: CLASS_TO },
  generated_by: "node " + process.version,
  queries,
  models: {},
};
for (const [name, md] of Object.entries(EMBEDDED_MODELS)) {
  const m = new GaussianLUT(md);
  out.models[name] = {
    num_gaussians: md.num_gaussians,
    residual: md.residual,
    parameters: {
      positions: md.parameters.positions,
      cholesky_diag: md.parameters.cholesky_diag,
      cholesky_off: md.parameters.cholesky_off,
      opacities_logit: md.parameters.opacities_logit,
      color_matrices: md.parameters.color_matrices,
      color_biases: md.parameters.color_biases,
      global_matrix: md.parameters.global_matrix,
      global_bias: md.parameters.global_bias,
    },
    outputs: queries.map((q) => m.forward(q)),
  };
}
fs.writeFileSync("glut_demo_ref.json", JSON.stringify(out));
console.log("models", Object.keys(out.models).length, "queries", queries.length, "sha", sha.slice(0, 16));
