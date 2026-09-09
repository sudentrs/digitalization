import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const batchDir = "output/batch_30_azioni";
const outputPath = path.join(batchDir, "azioni_30_page_review.xlsx");

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

function intValue(value) {
  if (value === undefined || value === null || value === "") return null;
  const cleaned = String(value).replace(/[^\d-]/g, "");
  return cleaned ? Number.parseInt(cleaned, 10) : null;
}

function numberValue(value) {
  if (value === undefined || value === null || value === "") return null;
  const cleaned = String(value).replace(",", ".").replace(/[^\d.-]/g, "");
  return cleaned ? Number.parseFloat(cleaned) : null;
}

function splitFlags(row) {
  return (row.validation_flags || "").split(";").filter(Boolean);
}

function priority(row) {
  const flags = new Set(splitFlags(row));
  if (flags.has("missing_issuer") || flags.has("missing_capital") || flags.has("missing_nominal")) return "High";
  if (flags.has("missing_close_price") || flags.has("blank_quantity")) return "Medium";
  if (flags.size) return "Low";
  return "OK";
}

function side(sampleId) {
  return sampleId.includes("_right") ? "right" : "left";
}

function mainFlag(row) {
  const ordered = [
    "missing_issuer",
    "missing_capital",
    "missing_nominal",
    "missing_close_price",
    "blank_quantity",
    "text_in_compensation",
    "multi_value_shares_raw",
    "low_mean_ocr_confidence",
  ];
  const flags = new Set(splitFlags(row));
  return ordered.find((flag) => flags.has(flag)) || splitFlags(row)[0] || "";
}

function countBy(rows, keyFn) {
  const counts = new Map();
  for (const row of rows) {
    const key = keyFn(row);
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

function normalizeCleanRow(row) {
  const displayRawText = String(row.raw_row_text || "")
    .replace(/#NAME\?/g, "[OCR_NAME?]")
    .replace(/#VALUE!/g, "[OCR_VALUE!]")
    .replace(/#REF!/g, "[OCR_REF!]")
    .replace(/#DIV\/0!/g, "[OCR_DIV0!]")
    .replace(/#N\/A/g, "[OCR_NA]");
  return {
    review_status: "",
    priority: priority(row),
    main_flag: mainFlag(row),
    sample_id: row.sample_id,
    side: side(row.sample_id || ""),
    row_index: intValue(row.row_index),
    section: row.section,
    issuer_name_clean: row.issuer_name_clean,
    capital_subscribed: intValue(row.capital_subscribed),
    shares_outstanding: intValue(row.shares_outstanding),
    nominal_value: numberValue(row.nominal_value),
    godimento_month_day: row.godimento_month_day,
    cedola_date: row.cedola_date,
    importo_lordo: numberValue(row.importo_lordo),
    importo_acconto: numberValue(row.importo_acconto),
    importo_saldo: numberValue(row.importo_saldo),
    compensation_price: numberValue(row.compensation_price),
    price_min: numberValue(row.price_min),
    price_max: numberValue(row.price_max),
    price_close: numberValue(row.price_close),
    quantity_traded: intValue(row.quantity_traded),
    validation_flags: row.validation_flags,
    raw_row_text: `'${displayRawText}`,
  };
}

function matrixFromObjects(rows, headers) {
  return [headers, ...rows.map((row) => headers.map((header) => row[header] ?? null))];
}

function setTableStyle(sheet, range, headerFill = "#1F4E79") {
  range.format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  const header = range.getRow(0);
  header.format = {
    fill: headerFill,
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
  };
}

function fit(sheet, range, widths = {}) {
  range.format.autofitColumns();
  range.format.autofitRows();
  for (const [col, width] of Object.entries(widths)) {
    sheet.getRange(`${col}:${col}`).format.columnWidth = width;
  }
}

async function build() {
  const cleanedRaw = await readCsv("combined_cleaned_candidate_rows.csv");
  const flagsRaw = await readCsv("combined_validation_flag_summary.csv");
  const sampleRaw = await readCsv("sample_summary.csv");
  const cleaned = cleanedRaw.map(normalizeCleanRow);
  const flagged = cleaned.filter((row) => row.validation_flags);
  const high = cleaned.filter((row) => row.priority === "High");
  const medium = cleaned.filter((row) => row.priority === "Medium");

  const workbook = Workbook.create();
  const summary = workbook.worksheets.add("Summary");
  const flags = workbook.worksheets.add("Flag Summary");
  const review = workbook.worksheets.add("Review Queue");
  const rows = workbook.worksheets.add("Cleaned Rows");
  const samples = workbook.worksheets.add("Sample Summary");
  const notes = workbook.worksheets.add("Notes");

  for (const sheet of [summary, flags, review, rows, samples, notes]) {
    sheet.showGridLines = false;
  }

  const sideCounts = countBy(cleaned, (row) => row.side);
  const priorityCounts = countBy(cleaned, (row) => row.priority);
  const sectionCounts = countBy(cleaned, (row) => row.section || "(blank)").slice(0, 15);
  const rescueCounts = [
    ["identity_columns_rescued", cleaned.filter((r) => splitFlags(r).includes("identity_columns_rescued")).length],
    ["nominal_cell_rescued", cleaned.filter((r) => splitFlags(r).includes("nominal_cell_rescued")).length],
    ["close_cell_backup_rescued", cleaned.filter((r) => splitFlags(r).includes("close_cell_backup_rescued")).length],
    ["probable_section_heading_row", cleaned.filter((r) => splitFlags(r).includes("probable_section_heading_row")).length],
    ["right_identity_cells_rescued", cleaned.filter((r) => splitFlags(r).includes("right_identity_cells_rescued")).length],
    ["right_price_cells_rescued", cleaned.filter((r) => splitFlags(r).includes("right_price_cells_rescued")).length],
    ["right_quantity_cells_rescued", cleaned.filter((r) => splitFlags(r).includes("right_quantity_cells_rescued")).length],
    ["left_numeric_shift_rescued", cleaned.filter((r) => splitFlags(r).includes("left_numeric_shift_rescued")).length],
  ];

  summary.getRange("A1:H1").merge();
  summary.getRange("A1").values = [["Azioni OCR Review Workbook"]];
  summary.getRange("A1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF", size: 16 } };
  summary.getRange("A3:B8").values = [
    ["Metric", "Value"],
    ["Cleaned candidate rows", cleaned.length],
    ["Rows with any flag", flagged.length],
    ["High priority review rows", high.length],
    ["Medium priority review rows", medium.length],
    ["Samples included", new Set(cleaned.map((r) => r.sample_id)).size],
  ];
  setTableStyle(summary, summary.getRange("A3:B8"));
  summary.getRange("B4:B8").format.numberFormat = "#,##0";

  summary.getRange("D3:E7").values = [["Priority", "Rows"], ...priorityCounts];
  setTableStyle(summary, summary.getRange(`D3:E${3 + priorityCounts.length}`), "#8064A2");
  summary.getRange(`E4:E${3 + priorityCounts.length}`).format.numberFormat = "#,##0";

  summary.getRange("G3:H5").values = [["Side", "Rows"], ...sideCounts];
  setTableStyle(summary, summary.getRange(`G3:H${3 + sideCounts.length}`), "#4F6228");
  summary.getRange(`H4:H${3 + sideCounts.length}`).format.numberFormat = "#,##0";

  summary.getRange(`A11:B${11 + rescueCounts.length}`).values = [["Rescue rule", "Rows"], ...rescueCounts];
  setTableStyle(summary, summary.getRange(`A11:B${11 + rescueCounts.length}`), "#C0504D");
  summary.getRange(`B12:B${11 + rescueCounts.length}`).format.numberFormat = "#,##0";

  summary.getRange(`D11:E${11 + sectionCounts.length}`).values = [["Section", "Rows"], ...sectionCounts];
  setTableStyle(summary, summary.getRange(`D11:E${11 + sectionCounts.length}`), "#31859B");
  summary.getRange(`E12:E${11 + sectionCounts.length}`).format.numberFormat = "#,##0";
  fit(summary, summary.getRange("A1:H28"), { A: 28, B: 18, D: 28, G: 16 });

  const flagHeaders = ["flag", "count"];
  flags.getRangeByIndexes(0, 0, flagsRaw.length + 1, flagHeaders.length).values = matrixFromObjects(
    flagsRaw.map((row) => ({ flag: row.flag, count: intValue(row.count) })),
    flagHeaders,
  );
  setTableStyle(flags, flags.getRangeByIndexes(0, 0, flagsRaw.length + 1, flagHeaders.length));
  flags.freezePanes.freezeRows(1);
  flags.getRange(`B2:B${flagsRaw.length + 1}`).format.numberFormat = "#,##0";
  fit(flags, flags.getRangeByIndexes(0, 0, flagsRaw.length + 1, flagHeaders.length), { A: 32, B: 12 });

  const reviewRows = flagged
    .sort((a, b) => {
      const order = { High: 0, Medium: 1, Low: 2, OK: 3 };
      return order[a.priority] - order[b.priority] || String(a.sample_id).localeCompare(String(b.sample_id)) || a.row_index - b.row_index;
    })
    .slice(0, 500);
  const reviewHeaders = [
    "review_status",
    "priority",
    "main_flag",
    "sample_id",
    "side",
    "row_index",
    "section",
    "issuer_name_clean",
    "capital_subscribed",
    "shares_outstanding",
    "nominal_value",
    "price_close",
    "quantity_traded",
    "validation_flags",
    "raw_row_text",
  ];
  review.getRangeByIndexes(0, 0, reviewRows.length + 1, reviewHeaders.length).values = matrixFromObjects(reviewRows, reviewHeaders);
  setTableStyle(review, review.getRangeByIndexes(0, 0, reviewRows.length + 1, reviewHeaders.length), "#7030A0");
  review.freezePanes.freezeRows(1);
  review.freezePanes.freezeColumns(4);
  review.getRange(`I2:M${reviewRows.length + 1}`).format.numberFormat = "#,##0.00";
  review.getRange(`O2:O${reviewRows.length + 1}`).format.wrapText = true;
  fit(review, review.getRangeByIndexes(0, 0, reviewRows.length + 1, reviewHeaders.length), {
    A: 18,
    B: 12,
    C: 26,
    D: 24,
    H: 34,
    N: 45,
    O: 85,
  });

  const rowHeaders = Object.keys(cleaned[0]);
  rows.getRangeByIndexes(0, 0, cleaned.length + 1, rowHeaders.length).values = matrixFromObjects(cleaned, rowHeaders);
  setTableStyle(rows, rows.getRangeByIndexes(0, 0, cleaned.length + 1, rowHeaders.length));
  rows.freezePanes.freezeRows(1);
  rows.freezePanes.freezeColumns(4);
  rows.getRange(`I2:T${cleaned.length + 1}`).format.numberFormat = "#,##0.00";
  fit(rows, rows.getRangeByIndexes(0, 0, cleaned.length + 1, rowHeaders.length), {
    D: 24,
    H: 34,
    U: 45,
    V: 80,
  });

  const sampleHeaders = Object.keys(sampleRaw[0]);
  samples.getRangeByIndexes(0, 0, sampleRaw.length + 1, sampleHeaders.length).values = matrixFromObjects(sampleRaw, sampleHeaders);
  setTableStyle(samples, samples.getRangeByIndexes(0, 0, sampleRaw.length + 1, sampleHeaders.length), "#4BACC6");
  samples.freezePanes.freezeRows(1);
  fit(samples, samples.getRangeByIndexes(0, 0, sampleRaw.length + 1, sampleHeaders.length), { A: 26 });

  notes.getRange("A1:F1").merge();
  notes.getRange("A1").values = [["How to read this workbook"]];
  notes.getRange("A1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF", size: 14 } };
  notes.getRange("A3:B12").values = [
    ["Sheet", "Purpose"],
    ["Summary", "Counts that show whether the parser is improving or getting worse."],
    ["Flag Summary", "All validation flags and how often they appear in the 30-page batch."],
    ["Review Queue", "The first 500 flagged rows, sorted by review priority. Use this for manual diagnosis."],
    ["Cleaned Rows", "All cleaned candidate security rows from the batch."],
    ["Sample Summary", "Per-page counts and flag totals."],
    ["High priority", "Rows missing issuer, capital, or nominal value."],
    ["Medium priority", "Rows missing close price or quantity."],
    ["Low priority", "Rows with weaker warnings or rescue flags."],
    ["Rescued rows", "Values filled by fallback logic. Keep the flag so the row can be audited later."],
  ];
  setTableStyle(notes, notes.getRange("A3:B12"), "#595959");
  fit(notes, notes.getRange("A1:F14"), { A: 24, B: 95 });

  const errorScan = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "formula error scan",
  });
  console.log(errorScan.ndjson);
  const preview = await workbook.render({ sheetName: "Summary", autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(batchDir, "azioni_30_page_review_summary_preview.png"), new Uint8Array(await preview.arrayBuffer()));
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  console.log(outputPath);
}

await build();
