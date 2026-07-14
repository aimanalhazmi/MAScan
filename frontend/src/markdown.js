// Validation is stored separately and belongs in the Validator graph detail,
// not in the report shown in chat or under the Synthesizer node. Strip legacy
// saved reports that already had the section appended by an older backend.
export function withoutFactCheck(markdown) {
  if (typeof markdown !== "string") return markdown || "";
  const match = /(?:^|\n)##\s+Fact Check\s*(?:\n|$)/i.exec(markdown);
  return match ? markdown.slice(0, match.index).trimEnd() : markdown;
}
