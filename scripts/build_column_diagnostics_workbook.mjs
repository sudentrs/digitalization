import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const batchDir = "output/batch_30_azioni";
const outputPath = path.join(batchDir, "azioni_column_diagnostics.xlsx");
const previewPath = path.join(batchDir, "azioni_column_diagnostics_preview.png");

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
  return rows
    .slice(1)
    .filter((r) => r.some((v) => v !== ""))
    .map((r) => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""])));
}

async function readCsv(fileName) {
  const text = await fs.readFile(path.join(batchDir, fileName), "utf8");
  return parseCsv(text);
}

function matrixFromObjects(rows, headers) {
  return [headers, ...rows.map((row) => headers.map((header) => safeCellValue(row[header] ?? "")))];
}

function safeCellValue(value) {
  if (typeof value !== "string") return value;
  if (/^[=+\-@]/.test(value) || /^#(NAME|VALUE|REF|DIV\/0|N\/A|NULL|NUM)!?\??$/i.test(value)) {
    return `'${value}`;
  }
  return value;
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

function writeTable(sheet, startRow, startCol, headers, rows) {
  const matrix = matrixFromObjects(rows, headers);
  const range = sheet.getRangeByIndexes(startRow, startCol, matrix.length, headers.length);
  range.values = matrix;
  setTableStyle(sheet, range);
  range.format.autofitColumns();
  range.format.autofitRows();
  return range;
}

function numberizeRows(rows, cols) {
  return rows.map((row) => {
    const out = { ...row };
    for (const col of cols) {
      const n = Number(out[col]);
      out[col] = Number.isFinite(n) ? n : out[col];
    }
    return out;
  });
}

const diagnostics = numberizeRows(await readCsv("combined_column_diagnostics.csv"), [
  "total_rows",
  "empty_clean",
  "empty_raw",
  "raw_present_clean_empty",
  "clean_present_raw_empty",
  "field_flagged_rows",
  "field_review_rows",
  "rescued_or_inherited_rows",
  "likely_printed_blank_rows",
]);
const examples = await readCsv("combined_column_issue_examples.csv");

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const detail = workbook.worksheets.add("Column Diagnostics");
const exampleSheet = workbook.worksheets.add("Issue Examples");

summary.showGridLines = false;
summary.getRange("A1:H1").merge();
summary.getRange("A1").values = [["Azioni Column Diagnostics"]];
summary.getRange("A1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
};
summary.getRange("A3:B8").values = [
  ["Rows checked", diagnostics[0]?.total_rows ?? 0],
  ["Columns checked", diagnostics.length],
  ["Highest review column", diagnostics[0]?.field ?? ""],
  ["Highest review rows", diagnostics[0]?.field_review_rows ?? 0],
  ["Largest blank-but-probably-benign column", "quantity_traded"],
  ["Use", "Prioritize parser/OCR fixes by column instead of by whole row"],
];
setTableStyle(summary, summary.getRange("A3:B8"));
summary.getRange("B3:B6").format.numberFormat = "#,##0";
summary.getRange("A10:F10").values = [["Field", "Review rows", "Empty clean", "Raw present but empty clean", "Rescued/inherited", "Likely printed blank"]];
summary.getRange("A11:F24").values = diagnostics.map((row) => [
  row.field,
  row.field_review_rows,
  row.empty_clean,
  row.raw_present_clean_empty,
  row.rescued_or_inherited_rows,
  row.likely_printed_blank_rows,
]);
setTableStyle(summary, summary.getRange("A10:F24"));
summary.getRange("B11:F24").format.numberFormat = "#,##0";
summary.getRange("A1:H24").format.autofitColumns();
summary.getRange("F10:F24").format.columnWidth = 18;
summary.freezePanes.freezeRows(10);

writeTable(
  detail,
  0,
  0,
  [
    "field",
    "total_rows",
    "empty_clean",
    "empty_raw",
    "raw_present_clean_empty",
    "clean_present_raw_empty",
    "field_flagged_rows",
    "field_review_rows",
    "rescued_or_inherited_rows",
    "likely_printed_blank_rows",
    "top_field_flags",
  ],
  diagnostics,
);
detail.freezePanes.freezeRows(1);
detail.getRange("K:K").format.columnWidth = 65;
detail.getRange("K:K").format.wrapText = true;

writeTable(
  exampleSheet,
  0,
  0,
  [
    "field",
    "issue_kind",
    "sample_id",
    "source_file",
    "row_index",
    "section",
    "issuer_name_clean",
    "clean_value",
    "raw_value",
    "validation_flags",
    "raw_row_text",
  ],
  examples,
);
exampleSheet.freezePanes.freezeRows(1);
exampleSheet.getRange("G:K").format.columnWidth = 28;
exampleSheet.getRange("G:K").format.wrapText = true;

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({ sheetName: "Summary", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
console.log(previewPath);
