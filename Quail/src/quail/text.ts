export function slugifyDatasetName(name: string): string {
	const slug = name
		.trim()
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "");
	return slug || "dataset";
}

export function tokenize(text: string): string[] {
	return text
		.toLowerCase()
		.normalize("NFKD")
		.replace(/[\u0300-\u036f]/g, "")
		.match(/[a-z0-9_']+/g)?.filter(Boolean) ?? [];
}

export function normalizeContainsText(text: string): string {
	return text.toLowerCase();
}

export function stableEntryId(datasetSlug: string, index: number): string {
	return `${datasetSlug}:${String(index + 1).padStart(6, "0")}`;
}

export function truncateText(text: string, max = 900): string {
	const compact = text.replace(/\s+/g, " ").trim();
	if (compact.length <= max) return compact;
	return `${compact.slice(0, Math.max(0, max - 1)).trimEnd()}...`;
}
