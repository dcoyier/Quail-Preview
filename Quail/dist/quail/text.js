export function slugifyDatasetName(name) {
    const slug = name
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "");
    return slug || "dataset";
}
export function tokenize(text) {
    return text
        .toLowerCase()
        .normalize("NFKD")
        .replace(/[\u0300-\u036f]/g, "")
        .match(/[a-z0-9_']+/g)?.filter(Boolean) ?? [];
}
export function normalizeContainsText(text) {
    return text.toLowerCase();
}
export function stableEntryId(datasetSlug, index) {
    return `${datasetSlug}:${String(index + 1).padStart(6, "0")}`;
}
export function truncateText(text, max = 900) {
    const compact = text.replace(/\s+/g, " ").trim();
    if (compact.length <= max)
        return compact;
    return `${compact.slice(0, Math.max(0, max - 1)).trimEnd()}...`;
}
//# sourceMappingURL=text.js.map