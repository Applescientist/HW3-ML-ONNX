// ---------- Regression: diabetes ----------
async function runRegression() {

  const x = new Float32Array(10);
  for (let i = 0; i < 10; i++) {
    x[i] = parseFloat(document.getElementById('r' + i).value) || 0;
  }

  const tensorX = new ort.Tensor('float32', x, [1, 10]);

  try {
    const session = await ort.InferenceSession.create("./diabetes_regression.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output  = results.output1.data;      // Float32Array length 1

    document.getElementById('predictionsReg').innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>predicted disease progression</td> <td id="regOut">${output[0].toFixed(1)}</td></tr>
        <tr><td>range in the dataset</td> <td>25 to 346</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    document.getElementById('predictionsReg').innerHTML = "Error: " + e.message;
  }
}


// ---------- Classification: breast cancer ----------
const featureNames = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
  'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
  'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
  'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
  'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
  'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'];

// two real patients from the dataset: row 0 (malignant) and row 19 (benign)
const examples = [
  [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
   0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
  [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462,
   0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]
];

// make the 30 input boxes when the page loads
window.onload = function () {
  let html = '';
  for (let i = 0; i < 30; i++) {
    html += `<input type="text" id="c${i}" size="8" value="${examples[0][i]}"> ${featureNames[i]}<br/>`;
  }
  document.getElementById('classInputs').innerHTML = html;
};

function loadExample(k) {
  for (let i = 0; i < 30; i++) {
    document.getElementById('c' + i).value = examples[k][i];
  }
}

async function runClassification() {

  const x = new Float32Array(30);
  for (let i = 0; i < 30; i++) {
    x[i] = parseFloat(document.getElementById('c' + i).value) || 0;
  }

  const tensorX = new ort.Tensor('float32', x, [1, 30]);

  try {
    const session = await ort.InferenceSession.create("./breast_cancer_classifier.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output  = results.output1.data;      // [P(malignant), P(benign)]
    const label   = output[0] > output[1] ? 'malignant' : 'benign';

    document.getElementById('predictionsClass').innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>P(malignant)</td> <td id="pMal">${output[0].toFixed(3)}</td></tr>
        <tr><td>P(benign)</td>    <td id="pBen">${output[1].toFixed(3)}</td></tr>
        <tr><td>prediction</td>   <td id="label"><b>${label}</b></td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    document.getElementById('predictionsClass').innerHTML = "Error: " + e.message;
  }
}
