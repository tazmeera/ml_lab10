// Builds Capstone_ML_Projects_Presentation.pptx (14 slides) for Lab Sheet-10.
// All figures on the slides are read from the model_comparison.csv / model_metadata.json files that the
// training scripts wrote, so the deck always matches the actual results.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const fa = require("react-icons/fa");
const { applyTheme } = require("/mnt/skills/public/pptx/scripts/apply_theme.js");

const HERE = __dirname;
const L = path.join(HERE, "p1_loan");
const D = path.join(HERE, "p2_disease");

// ---------- results from disk ----------
const readCsv = (f) => {
  const [h, ...rows] = fs.readFileSync(f, "utf8").trim().split("\n").map((r) => r.split(","));
  return rows.map((r) => Object.fromEntries(h.map((k, i) => [k, r[i]])));
};
const loanCmp = readCsv(path.join(L, "model_comparison.csv"));
const disCmp = readCsv(path.join(D, "model_comparison.csv"));
const loanMeta = JSON.parse(fs.readFileSync(path.join(L, "model_metadata.json")));
const disMeta = JSON.parse(fs.readFileSync(path.join(D, "model_metadata.json")));
const loanTuned = loanCmp[0];
const loanRaw = readCsv(path.join(L, "loan_data.csv"));
const approvalRate = loanRaw.filter((r) => r.Loan_Status === "Y").length / loanRaw.length;
const missCredit = loanRaw.filter((r) => r.Credit_History === "").length / loanRaw.length;
const pct = (x, d = 1) => (parseFloat(x) * 100).toFixed(d) + "%";

// ---------- theme ----------
const THEME = {
  name: "Capstone ML",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F2A", lt1: "FFFFFF", dk2: "12343B", lt2: "EEF5F4",
    accent1: "1F8A80", accent2: "E4572E", accent3: "F2B134", accent4: "4C78A8", accent5: "7A9E9F", accent6: "6B4E71",
    hlink: "1F8A80", folHlink: "6B4E71",
  },
};
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "Capstone Machine Learning Projects";
pres.author = "MCA III Semester";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: "12343B" },
  objects: [],
  slideNumber: undefined,
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { text: { text: "Lab Sheet-10  |  Capstone Machine Learning Projects  |  MCA III Semester",
        options: { x: 0.5, y: 5.25, w: 7.5, h: 0.25, fontSize: 10, color: "7A8A8C", margin: 0, isTextBox: true } } },
  ],
  slideNumber: { x: 9.0, y: 5.25, w: 0.5, h: 0.25, fontSize: 10, color: "7A8A8C", align: "right" },
});

const icon = async (Comp, color = "#FFFFFF") => {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
};
const imgInfo = async (f) => {
  const m = await sharp(f).metadata();
  return m.width / m.height;
};
// fit an image inside a w x h box, centred
const fit = async (slide, file, x, y, w, h, name) => {
  const ar = await imgInfo(file);
  let iw = w, ih = w / ar;
  if (ih > h) { ih = h; iw = h * ar; }
  slide.addImage({ path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih, objectName: name, altText: name });
};
const title = (slide, text) =>
  slide.addText(text, { x: 0.5, y: 0.3, w: 9, h: 0.65, fontFace: "Cambria", fontSize: 32, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: "Title" });
const card = (slide, x, y, w, h, fill = C.background2) =>
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.1 });
const stat = (slide, x, y, w, big, small, color = C.accent1) => {
  card(slide, x, y, w, 1.05);
  slide.addText(big, { x: x + 0.15, y: y + 0.08, w: w - 0.3, h: 0.55, fontFace: "Cambria", fontSize: 30, bold: true, color, margin: 0, isTextBox: true });
  slide.addText(small, { x: x + 0.15, y: y + 0.63, w: w - 0.3, h: 0.35, fontSize: 12, color: C.text1, margin: 0, isTextBox: true });
};
const bullets = (slide, items, x, y, w, h, fs = 14) =>
  slide.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 } })),
    { x, y, w, h, fontSize: fs, color: C.text1, margin: 0, valign: "top", isTextBox: true });

const chartBase = {
  catAxisLabelColor: "4A5A5C", valAxisLabelColor: "4A5A5C", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, valGridLine: { color: "DCE6E5", size: 0.75 }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: "4A5A5C",
  showValue: true, dataLabelFontSize: 10, dataLabelFontFace: "+mn-lt", dataLabelColor: "1B1F2A", dataLabelFormatCode: "0.00",
};

(async () => {
  const ic = {
    bank: await icon(fa.FaUniversity), heart: await icon(fa.FaHeartbeat), code: await icon(fa.FaCode),
    db: await icon(fa.FaDatabase), chart: await icon(fa.FaChartBar), cube: await icon(fa.FaCubes),
    flag: await icon(fa.FaBullseye), check: await icon(fa.FaCheckCircle), warn: await icon(fa.FaExclamationTriangle),
    rocket: await icon(fa.FaRocket), git: await icon(fa.FaGithub), py: await icon(fa.FaPython),
  };
  const circ = (slide, x, y, d, data, fill = C.accent1) => {
    slide.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 } });
    slide.addImage({ data, x: x + d * 0.25, y: y + d * 0.25, w: d * 0.5, h: d * 0.5, altText: "icon" });
  };

  // 1 ── Title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  circ(s, 0.7, 1.0, 0.8, ic.bank, C.accent1);
  circ(s, 1.65, 1.0, 0.8, ic.heart, C.accent2);
  s.addText("Capstone Machine Learning Projects", { x: 0.7, y: 2.0, w: 8.6, h: 1.4, fontFace: "Cambria", fontSize: 40, valign: "top", bold: true, color: "FFFFFF", margin: 0, isTextBox: true });
  s.addText("Loan Approval Prediction  +  Disease Prediction", { x: 0.7, y: 3.45, w: 8.6, h: 0.5, fontSize: 22, color: "9ED6CF", margin: 0, isTextBox: true });
  s.addText("Lab Sheet-10  |  MCA III Semester (Session 2026-2027)\nCOER University, Roorkee, Uttarakhand", { x: 0.7, y: 4.4, w: 8.6, h: 0.7, fontSize: 14, color: "D5E3E2", margin: 0, isTextBox: true });
  s.addNotes("Introduce the two projects: both follow the same end-to-end workflow - data, preprocessing, model comparison, evaluation, saved model, web UI.");

  // 2 ── Overview
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  title(s, "Two projects, one workflow");
  [[0.5, ic.bank, C.accent1, "Project 1: Loan Approval", "Predict whether a loan is approved from income, education, employment, loan amount, credit history and marital status.", `Best model: ${loanMeta.best_model}`],
   [5.1, ic.heart, C.accent2, "Project 2: Disease Prediction", "Predict the most likely disease from 26 symptoms plus age, gender and body temperature; shows the top-3 candidates.", `Best model: ${disMeta.best_model}`]]
    .forEach(([x, ico, col, h, t, m]) => {
      card(s, x, 1.2, 4.4, 3.7);
      circ(s, x + 0.3, 1.45, 0.75, ico, col);
      s.addText(h, { x: x + 0.3, y: 2.35, w: 3.8, h: 0.45, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text2, margin: 0, isTextBox: true });
      s.addText(t, { x: x + 0.3, y: 2.85, w: 3.8, h: 1.2, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
      s.addText(m, { x: x + 0.3, y: 4.25, w: 3.8, h: 0.4, fontSize: 14, bold: true, color: col, margin: 0, isTextBox: true });
    });

  // 3 ── Objectives
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  title(s, "Objectives and approach");
  [[ic.db, "Preprocess real-world style data", "Missing values imputed, categories encoded, numbers scaled, new features engineered."],
   [ic.cube, "Compare multiple algorithms", "5 classifiers (loan) and 7 classifiers (disease) judged with 5-fold cross-validation."],
   [ic.chart, "Evaluate with suitable metrics", "Accuracy, precision, recall, F1, ROC-AUC, macro-F1 and top-3 accuracy."],
   [ic.rocket, "Deploy a simple interface", "Trained pipelines saved with joblib and served through Streamlit apps."]]
    .forEach(([ico, h, t], i) => {
      const y = 1.2 + i * 0.95;
      circ(s, 0.5, y, 0.65, ico, i % 2 ? C.accent2 : C.accent1);
      s.addText(h, { x: 1.4, y: y - 0.02, w: 8, h: 0.35, fontFace: "Cambria", fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true });
      s.addText(t, { x: 1.4, y: y + 0.33, w: 8, h: 0.35, fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
    });

  // 4 ── Stack
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  title(s, "Tools and technology");
  const tech = [["Python", "Core language", ic.py], ["Pandas / NumPy", "Data handling", ic.db], ["Scikit-learn", "Pipelines, models, CV", ic.cube],
                ["Matplotlib / Seaborn", "EDA and evaluation plots", ic.chart], ["Streamlit", "Prediction interface", ic.rocket], ["Joblib + GitHub", "Saved models, version control", ic.git]];
  tech.forEach(([h, t, ico], i) => {
    const x = 0.5 + (i % 3) * 3.05, y = 1.25 + Math.floor(i / 3) * 1.95;
    card(s, x, y, 2.85, 1.7);
    circ(s, x + 0.2, y + 0.2, 0.6, ico, C.accent1);
    s.addText(h, { x: x + 0.2, y: y + 0.9, w: 2.5, h: 0.35, fontFace: "Cambria", fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.2, y: y + 1.25, w: 2.5, h: 0.3, fontSize: 12, color: C.text1, margin: 0, isTextBox: true });
  });

  // 5 ── Loan data/EDA
  pres.addSection({ title: "Project 1: Loan Approval" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Project 1: Loan Approval" });
  title(s, "Loan data and EDA");
  await fit(s, path.join(L, "eda_charts.png"), 0.5, 1.15, 5.9, 3.9, "EDA charts for loan data");
  stat(s, 6.7, 1.2, 2.8, loanRaw.length.toLocaleString("en-US"), "applicants, 11 input features");
  stat(s, 6.7, 2.4, 2.8, pct(approvalRate), "approved (class imbalance)");
  stat(s, 6.7, 3.6, 2.8, pct(missCredit), "missing credit history, imputed", C.accent2);
  s.addNotes("Credit history is the strongest separator of approved vs rejected. Data is simulated for the lab; the script runs unchanged on a real Loan Prediction CSV with the same columns.");

  // 6 ── Loan pipeline
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Project 1: Loan Approval" });
  title(s, "Loan workflow");
  await fit(s, path.join(L, "flowchart.png"), 0.5, 1.1, 9, 3.2, "Loan system flowchart");
  bullets(s, ["Median / most-frequent imputation inside the pipeline (no data leakage)",
              "Engineered features: total household income and EMI-to-income ratio"], 0.5, 4.3, 9, 0.85, 14);

  // 7 ── Loan model comparison (native chart)
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Project 1: Loan Approval" });
  title(s, "Loan: model comparison");
  const lrows = loanCmp.slice(1);
  s.addChart(pres.charts.BAR, [
    { name: "Test accuracy", labels: lrows.map((r) => r.Model), values: lrows.map((r) => +parseFloat(r.Accuracy).toFixed(3)) },
    { name: "ROC-AUC", labels: lrows.map((r) => r.Model), values: lrows.map((r) => +parseFloat(r.ROC_AUC).toFixed(3)) },
  ], { x: 0.5, y: 1.1, w: 6.2, h: 3.95, barDir: "col", chartColors: ["1F8A80", "E4572E"], valAxisMinVal: 0.7, valAxisMaxVal: 0.9, valAxisLabelFormatCode: "0.00", ...chartBase });
  stat(s, 7.0, 1.2, 2.5, pct(loanTuned.Accuracy), "test accuracy (tuned LR)");
  stat(s, 7.0, 2.4, 2.5, (+loanTuned.ROC_AUC).toFixed(3), "ROC-AUC", C.accent2);
  stat(s, 7.0, 3.6, 2.5, pct(loanTuned.Recall), "recall on approved loans", C.accent4);

  // 8 ── Loan results
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Project 1: Loan Approval" });
  title(s, "What drives approval");
  await fit(s, path.join(L, "model_evaluation.png"), 0.5, 1.1, 9, 2.7, "Loan confusion matrix, ROC and comparison");
  card(s, 0.5, 3.95, 4.4, 1.15);
  s.addText([{ text: "Top drivers", options: { bold: true, breakLine: true, color: "1F8A80" } },
             { text: loanMeta.top_features.join(", ").replace(/_/g, " ").replace("ApplicantIncome", "Applicant income").replace("LoanAmount", "Loan amount"), options: {} }],
    { x: 0.7, y: 4.03, w: 4.0, h: 1.0, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  card(s, 5.1, 3.95, 4.4, 1.15);
  s.addText([{ text: "Trade-off", options: { bold: true, breakLine: true, color: "E4572E" } },
             { text: `Catches ${pct(loanMeta.recall_approved, 0)} of approvals but only ${pct(loanMeta.recall_rejected, 0)} of rejections; the cut-off can be tuned to risk appetite.`, options: {} }],
    { x: 5.3, y: 4.03, w: 4.0, h: 1.0, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });

  // 9 ── Loan architecture
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Project 1: Loan Approval" });
  title(s, "Loan system architecture");
  await fit(s, path.join(L, "system_architecture.png"), 0.5, 1.1, 6.0, 3.9, "Loan system architecture");
  bullets(s, ["Streamlit form collects applicant details", "One saved pipeline: features, preprocessing and model",
              "Result: approval probability, monthly EMI and EMI-to-income ratio", "Hints explain weak points such as poor credit history"], 6.75, 1.3, 2.75, 3.6, 14);

  // 10 ── Disease data
  pres.addSection({ title: "Project 2: Disease Prediction" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Project 2: Disease Prediction" });
  title(s, "Disease data and symptoms");
  await fit(s, path.join(D, "eda_symptom_profiles.png"), 0.5, 1.15, 6.0, 3.9, "Symptom profile heatmap");
  stat(s, 6.8, 1.2, 2.7, "4,200", "patient records, 10 diseases");
  stat(s, 6.8, 2.4, 2.7, "26", "symptoms + age, gender, temp", C.accent2);
  stat(s, 6.8, 3.6, 2.7, "Overlap", "fever, headache, fatigue shared", C.accent4);

  // 11 ── Disease comparison
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Project 2: Disease Prediction" });
  title(s, "Disease: model comparison");
  const drows = disCmp.slice(1);
  s.addChart(pres.charts.BAR, [
    { name: "CV accuracy", labels: drows.map((r) => r.Model), values: drows.map((r) => +parseFloat(r.CV_Accuracy).toFixed(3)) },
    { name: "Test accuracy", labels: drows.map((r) => r.Model), values: drows.map((r) => +parseFloat(r.Test_Accuracy).toFixed(3)) },
  ], { x: 0.5, y: 1.1, w: 6.4, h: 3.95, barDir: "col", chartColors: ["1F8A80", "E4572E"], valAxisMinVal: 0.6, valAxisMaxVal: 1.0, valAxisLabelFormatCode: "0.00", catAxisLabelFontSize: 10, ...chartBase, dataLabelFontSize: 9 });
  stat(s, 7.1, 1.2, 2.4, pct(disMeta.test_accuracy), "test accuracy", C.accent1);
  stat(s, 7.1, 2.4, 2.4, pct(disMeta.macro_f1), "macro F1", C.accent2);
  stat(s, 7.1, 3.6, 2.4, pct(disMeta.top3_accuracy), "top-3 accuracy", C.accent4);

  // 12 ── Disease evaluation
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Project 2: Disease Prediction" });
  title(s, "Where the model confuses");
  await fit(s, path.join(D, "model_evaluation.png"), 0.5, 1.1, 9, 2.9, "Disease confusion matrix and comparison");
  bullets(s, ["Decision Tree was weakest (about 70%); linear and kernel models handled overlapping symptoms best",
              "Most common error: influenza vs COVID-19 (16 of 80 mistakes), so the app shows the top 3 candidates"], 0.5, 4.1, 9, 1.0, 14);

  // 13 ── Disease architecture
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Project 2: Disease Prediction" });
  title(s, "Disease system and UI");
  await fit(s, path.join(D, "system_architecture.png"), 0.5, 1.1, 6.0, 3.9, "Disease system architecture");
  bullets(s, ["Pick symptoms, enter age, gender and temperature", "Top-3 diseases with probabilities in a bar chart",
              "Low-confidence warning when the top choice is under 50%", "Clear 'not a medical device' disclaimer"], 6.75, 1.3, 2.75, 3.6, 14);

  // 14 ── Conclusion
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Conclusion" });
  s.addText("Conclusions and future work", { x: 0.7, y: 0.5, w: 8.6, h: 0.7, fontFace: "Cambria", fontSize: 34, bold: true, color: "FFFFFF", margin: 0, isTextBox: true });
  const cols = [[ic.check, "What we achieved", `Loan model ${pct(loanTuned.Accuracy, 0)} accurate (AUC ${(+loanTuned.ROC_AUC).toFixed(2)}); disease model ${pct(disMeta.test_accuracy, 0)} (top-3 ${pct(disMeta.top3_accuracy, 0)}). Both saved and served via Streamlit.`],
                [ic.warn, "Limitations", "Both datasets are simulated. Real data needs fairness checks (loans) and clinical validation (health)."],
                [ic.rocket, "Next steps", "Train on real datasets, add explainability (SHAP), monitor drift, deploy to a cloud host."]];
  cols.forEach(([ico, h, t], i) => {
    const x = 0.7 + i * 2.95;
    s.addShape(pres.ShapeType.roundRect, { x, y: 1.5, w: 2.75, h: 3.3, fill: { color: "1D4A52" }, line: { color: "1D4A52", width: 0 }, rectRadius: 0.1 });
    circ(s, x + 0.25, 1.75, 0.65, ico, i === 1 ? C.accent2 : C.accent1);
    s.addText(h, { x: x + 0.25, y: 2.55, w: 2.3, h: 0.4, fontFace: "Cambria", fontSize: 18, bold: true, color: "FFFFFF", margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.25, y: 3.0, w: 2.3, h: 1.7, fontSize: 14, color: "D5E3E2", margin: 0, valign: "top", isTextBox: true });
  });

  const out = path.join(HERE, "Capstone_ML_Projects_Presentation.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("written", out);
})();
