import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const batchDir = "output/batch_30_azioni";
const outputPath = path.join(batchDir, "azioni_high_priority_review.xlsx");
const previewPath = path.join(batchDir, "azioni_high_priority_review_preview.png");

const HIGH_FLAGS = new Set(["missing_issuer", "missing_capital", "missing_nominal"]);

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (quoted) {
      if (ch === '"' && next === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  if (!rows.length) return [];
  const headers = rows[0];
  return rows.slice(1).filter((r) => r.some((v) => v !== "")).map((r) => {
    const obj = {};
    headers.forEach((h, i) => {
      obj[h] = r[i] ?? "";
    });
    return obj;
  });
}

async function readCsv(fileName) {
  const text = await fs.readFile(path.join(batchDir, fileName), "utf8");
  return parseCsv(text);
}

function splitFlags(row) {
  return (row.validation_flags || "").split(";").filter(Boolean);
}

function hasHighFlag(row) {
  return splitFlags(row).some((flag) => HIGH_FLAGS.has(flag));
}

function side(sampleId) {
  return sampleId.includes("_right") ? "right" : "left";
}

function samplePaths(sampleId) {
  const sampleDir = path.resolve(batchDir, "samples", sampleId);
  return {
    overlay_path: path.join(sampleDir, "structured_overlay.jpg"),
    table_crop_path: path.resolve("output/preprocess_1950_azioni/table_crops", `${sampleId}_table.jpg`),
    cleaned_rows_path: path.join(sampleDir, "cleaned_candidate_rows.csv"),
    identity_ocr_path: path.join(sampleDir, "identity_cell_ocr", "identity_cell_ocr_by_row.csv"),
  };
}

function priorityReason(row) {
  const flags = splitFlags(row);
  return flags.filter((flag) => HIGH_FLAGS.has(flag) || flag === "missing_shares").join("; ");
}

function matrixFromObjects(rows, headers) {
  return [headers, ...rows.map((row) => headers.map((header) => row[header] ?? null))];
}

function setHeader(range, fill = "#1F4E79") {
  range.format = {
    fill,
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#9EADCC" },
  };
}

function setTableStyle(sheet, range) {
  range.format.borders = { preset: "inside", style: "thin", color: "#E5E7EB" };
  range.getRow(0).format = {
    fill: "#1F4E79",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
  };
  sheet.showGridLines = false;
}

function countBy(rows, fn) {
  const counts = new Map();
  for (const row of rows) {
    const key = fn(row);
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

async function build() {
  const cleaned = await readCsv("combined_cleaned_candidate_rows.csv");
  const highRows = cleaned.filter(hasHighFlag);
  const flagCounts = [];
  const flagMap = new Map();
  for (const row of highRows) {
    for (const flag of splitFlags(row)) {
      flagMap.set(flag, (flagMap.get(flag) || 0) + 1);
    }
  }
  for (const [flag, count] of [...flagMap.entries()].sort((a, b) => b[1] - a[1])) {
    flagCounts.push({ flag, count });
  }

  const reviewRows = highRows.map((row) => ({
    review_status: "",
    corrected_issuer: "",
    corrected_capital: "",
    corrected_shares: "",
    corrected_nominal: "",
    correction_notes: "",
    priority_reason: priorityReason(row),
    sample_id: row.sample_id,
    row_index: Number.parseInt(row.row_index || "0", 10),
    side: side(row.sample_id || ""),
    row_role: row.row_role,
    section: row.section,
    issuer_name_clean: row.issuer_name_clean,
    capital_subscribed: row.capital_subscribed,
    shares_outstanding: row.shares_outstanding,
    nominal_value: row.nominal_value,
    raw_row_text: row.raw_row_text,
    validation_flags: row.validation_flags,
    ...samplePaths(row.sample_id),
  }));

  const workbook = Workbook.create();
  const summary = workbook.worksheets.add("Summary");
  const review = workbook.worksheets.add("High Priority Rows");
  const flags = workbook.worksheets.add("Flag Counts");
  const instructions = workbook.worksheets.add("Instructions");

  for (const sheet of [summary, review, flags, instructions]) {
    sheet.showGridLines = false;
  }

  summary.getRange("A1:F1").merge();
  summary.getRange("A1").values = [["Azioni High-Priority Manual Review"]];
  summary.getRange("A1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF", size: 16 } };

  const metrics = [
    ["Metric", "Value"],
    ["Cleaned candidate rows", cleaned.length],
    ["High-priority rows", highRows.length],
    ["Samples represented", new Set(highRows.map((row) => row.sample_id)).size],
    ["Rows missing capital", highRows.filter((row) => splitFlags(row).includes("missing_capital")).length],
    ["Rows missing shares", highRows.filter((row) => splitFlags(row).includes("missing_shares")).length],
    ["Rows missing nominal", highRows.filter((row) => splitFlags(row).includes("missing_nominal")).length],
    ["Rows missing issuer", highRows.filter((row) => splitFlags(row).includes("missing_issuer")).length],
  ];
  summary.getRange(`A3:B${2 + metrics.length}`).values = metrics;
  setTableStyle(summary, summary.getRange(`A3:B${2 + metrics.length}`));
  summary.getRange(`B4:B${2 + metrics.length}`).format.numberFormat = "#,##0";

  const bySample = countBy(highRows, (row) => row.sample_id).slice(0, 15);
  summary.getRange(`D3:E${3 + bySample.length}`).values = [["Sample", "High rows"], ...bySample];
  setTableStyle(summary, summary.getRange(`D3:E${3 + bySample.length}`));
  summary.getRange(`E4:E${3 + bySample.length}`).format.numberFormat = "#,##0";

  const bySection = countBy(highRows, (row) => row.section || "(blank)").slice(0, 15);
  summary.getRange(`A13:B${13 + bySection.length}`).values = [["Section", "High rows"], ...bySection];
  setTableStyle(summary, summary.getRange(`A13:B${13 + bySection.length}`));
  summary.getRange(`B14:B${13 + bySection.length}`).format.numberFormat = "#,##0";

  const reviewHeaders = [
    "review_status",
    "corrected_issuer",
    "corrected_capital",
    "corrected_shares",
    "corrected_nominal",
    "correction_notes",
    "priority_reason",
    "sample_id",
    "row_index",
    "side",
    "row_role",
    "section",
    "issuer_name_clean",
    "capital_subscribed",
    "shares_outstanding",
    "nominal_value",
    "raw_row_text",
    "validation_flags",
    "overlay_path",
    "table_crop_path",
    "cleaned_rows_path",
    "identity_ocr_path",
  ];
  review.getRangeByIndexes(0, 0, reviewRows.length + 1, reviewHeaders.length).values = matrixFromObjects(reviewRows, reviewHeaders);
  setTableStyle(review, review.getRangeByIndexes(0, 0, reviewRows.length + 1, reviewHeaders.length));
  review.freezePanes.freezeRows(1);
  review.freezePanes.freezeColumns(7);
  review.tables.add(`A1:V${reviewRows.length + 1}`, true, "HighPriorityRows");
  review.getRange(`A2:A${reviewRows.length + 1}`).dataValidation = {
    rule: { type: "list", values: ["Needs review", "Corrected", "Accept as parsed", "Not a security", "Skip"] },
  };
  review.getRange(`N2:P${reviewRows.length + 1}`).format.numberFormat = "#,##0";
  review.getRange(`Q2:R${reviewRows.length + 1}`).format.wrapText = true;

  const flagHeaders = ["flag", "count"];
  flags.getRangeByIndexes(0, 0, flagCounts.length + 1, flagHeaders.length).values = matrixFromObjects(flagCounts, flagHeaders);
  setTableStyle(flags, flags.getRangeByIndexes(0, 0, flagCounts.length + 1, flagHeaders.length));
  flags.getRange(`B2:B${flagCounts.length + 1}`).format.numberFormat = "#,##0";

  instructions.getRange("A1:D1").merge();
  instructions.getRange("A1").values = [["How to use this review file"]];
  instructions.getRange("A1").format = { fill: "#4F6228", font: { bold: true, color: "#FFFFFF", size: 14 } };
  instructions.getRange("A3:B9").values = [
    ["Step", "Action"],
    ["1", "Open the overlay_path or table_crop_path for the row you are checking."],
    ["2", "Compare the flagged parsed fields with the image."],
    ["3", "If needed, fill corrected_issuer, corrected_capital, corrected_shares, or corrected_nominal."],
    ["4", "Set review_status so corrected rows can be filtered later."],
    ["5", "Use correction_notes for uncertainty, repeated patterns, or non-security rows."],
    ["6", "Do not overwrite parsed fields directly; keep corrections in the correction columns."],
  ];
  setTableStyle(instructions, instructions.getRange("A3:B9"));
  instructions.getRange("B3:B9").format.wrapText = true;

  for (const sheet of [summary, review, flags, instructions]) {
    const used = sheet.getUsedRange();
    used.format.autofitColumns();
    used.format.autofitRows();
  }
  review.getRange("A:A").format.columnWidth = 18;
  review.getRange("B:F").format.columnWidth = 22;
  review.getRange("G:G").format.columnWidth = 30;
  review.getRange("M:R").format.columnWidth = 28;
  review.getRange("S:V").format.columnWidth = 50;
  instructions.getRange("B:B").format.columnWidth = 80;

  const summaryPreview = await workbook.render({ sheetName: "Summary", autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(previewPath, new Uint8Array(await summaryPreview.arrayBuffer()));

  const errorScan = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "final formula error scan",
  });
  console.log(errorScan.ndjson);

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  console.log(outputPath);
}

build().catch((error) => {
  console.error(error);
  process.exit(1);
});
