import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const batchDir = "output/batch_30_azioni";
const csvPath = path.join(batchDir, "manual_corrections.csv");
const outputPath = path.join(batchDir, "azioni_manual_corrections.xlsx");
const previewPath = path.join(batchDir, "azioni_manual_corrections_preview.png");

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
  if (!rows.length) return { headers: [], rows: [] };
  const headers = rows[0].map((header) => header.replace(/^\uFEFF/, ""));
  const objects = rows
    .slice(1)
    .filter((values) => values.some((value) => value !== ""))
    .map((values) => {
      const obj = {};
      headers.forEach((header, index) => {
        obj[header] = values[index] ?? "";
      });
      return obj;
    });
  return { headers, rows: objects };
}

function matrix(rows, headers) {
  return [headers, ...rows.map((row) => headers.map((header) => row[header] ?? ""))];
}

function setTableStyle(range, fill = "#1F4E79") {
  range.format.borders = { preset: "inside", style: "thin", color: "#E5E7EB" };
  range.getRow(0).format = {
    fill,
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
  };
}

function countBy(rows, field) {
  const counts = new Map();
  for (const row of rows) {
    const key = row[field] || "(blank)";
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({ name, count }));
}

async function build() {
  const { headers, rows } = parseCsv(await fs.readFile(csvPath, "utf8"));
  const workbook = Workbook.create();
  const instructions = workbook.worksheets.add("Instructions");
  const corrections = workbook.worksheets.add("Corrections");
  const actions = workbook.worksheets.add("Allowed Actions");
  const summary = workbook.worksheets.add("Summary");

  for (const sheet of [instructions, corrections, actions, summary]) {
    sheet.showGridLines = false;
  }

  instructions.getRange("A1:D1").merge();
  instructions.getRange("A1").values = [["Manual correction loop for high-priority azioni rows"]];
  instructions.getRange("A1").format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF", size: 15 },
  };
  instructions.getRange("A3:B10").values = [
    ["Step", "What to do"],
    ["1", "Use snippet_path first; use overlay_path or original_full_path only when the row context is unclear."],
    ["2", "Set review_action to keep, exclude, merge, or manual_review."],
    ["3", "For keep, fill only the corrected fields that need changing."],
    ["4", "For exclude, leave corrected fields blank; the applier marks row_role as manual_exclude."],
    ["5", "For merge, add merge_target_row_index when you know the destination row."],
    ["6", "Keep parsed fields unchanged; put corrections only in corrected_* columns."],
    ["7", "After saving CSV, run scripts/apply_manual_corrections.py."],
  ];
  setTableStyle(instructions.getRange("A3:B10"), "#4F6228");
  instructions.getRange("B:B").format.columnWidth = 95;
  instructions.getRange("B:B").format.wrapText = true;

  corrections.getRangeByIndexes(0, 0, rows.length + 1, headers.length).values = matrix(rows, headers);
  setTableStyle(corrections.getRangeByIndexes(0, 0, rows.length + 1, headers.length));
  corrections.tables.add(`A1:X${rows.length + 1}`, true, "ManualCorrections");
  corrections.freezePanes.freezeRows(1);
  corrections.freezePanes.freezeColumns(6);
  corrections.getRange(`R2:R${rows.length + 1}`).dataValidation = {
    rule: { type: "list", values: ["manual_review", "keep", "exclude", "merge"] },
  };
  corrections.getRange("A:X").format.wrapText = true;
  corrections.getRange("A:F").format.columnWidth = 22;
  corrections.getRange("G:M").format.columnWidth = 24;
  corrections.getRange("N:N").format.columnWidth = 58;
  corrections.getRange("O:Q").format.columnWidth = 48;
  corrections.getRange("R:X").format.columnWidth = 26;

  const allowedRows = [
    { action: "manual_review", meaning: "Default state. Keep it for rows that still need visual checking." },
    { action: "keep", meaning: "Use the parsed row, applying any nonblank corrected_* values." },
    { action: "exclude", meaning: "Mark the parsed row as a fragment/non-security row to exclude downstream." },
    { action: "merge", meaning: "Mark the row as needing merge; merge_target_row_index can point to the main row." },
  ];
  actions.getRange("A1:B5").values = [["review_action", "meaning"], ...allowedRows.map((row) => [row.action, row.meaning])];
  setTableStyle(actions.getRange("A1:B5"), "#8064A2");
  actions.getRange("B:B").format.columnWidth = 85;
  actions.getRange("B:B").format.wrapText = true;

  const summaryRows = [
    ["Metric", "Value"],
    ["Rows prepared for manual correction", rows.length],
    ["Samples represented", new Set(rows.map((row) => row.sample_id)).size],
    ["Rows currently set to manual_review", rows.filter((row) => row.review_action === "manual_review").length],
  ];
  summary.getRange("A1:B4").values = summaryRows;
  setTableStyle(summary.getRange("A1:B4"));
  const categories = countBy(rows, "problem_category");
  summary.getRangeByIndexes(6, 0, categories.length + 1, 2).values = [
    ["Problem category", "Rows"],
    ...categories.map((row) => [row.name, row.count]),
  ];
  setTableStyle(summary.getRangeByIndexes(6, 0, categories.length + 1, 2), "#9E480E");
  summary.getRange("A:B").format.columnWidth = 36;

  for (const sheet of [instructions, corrections, actions, summary]) {
    const used = sheet.getUsedRange();
    used.format.autofitRows();
  }

  const preview = await workbook.render({ sheetName: "Corrections", autoCrop: "all", scale: 0.8, format: "png" });
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

  const errorScan = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "manual corrections workbook formula error scan",
  });
  console.log(errorScan.ndjson);

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  console.log(outputPath);
  console.log(previewPath);
}

build().catch((error) => {
  console.error(error);
  process.exit(1);
});
