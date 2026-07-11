import type { JsonPatchOperation, JsonValue } from "./types";

const escapeToken = (value: string) => value.replaceAll("~", "~0").replaceAll("/", "~1");

const same = (left: JsonValue, right: JsonValue) =>
  JSON.stringify(left) === JSON.stringify(right);

export function buildPatch(
  current: Record<string, JsonValue>,
  proposed: Record<string, JsonValue>,
): JsonPatchOperation[] {
  const operations: JsonPatchOperation[] = [];
  walk(current, proposed, "", operations);
  return operations.filter((operation) => operation.path !== "/schema_version");
}

function walk(
  current: JsonValue | undefined,
  proposed: JsonValue | undefined,
  path: string,
  operations: JsonPatchOperation[],
) {
  if (same(current ?? null, proposed ?? null)) return;
  if (current === undefined) {
    operations.push({ op: "add", path, value: proposed });
    return;
  }
  if (proposed === undefined) {
    operations.push({ op: "remove", path });
    return;
  }
  if (
    current !== null &&
    proposed !== null &&
    typeof current === "object" &&
    typeof proposed === "object" &&
    !Array.isArray(current) &&
    !Array.isArray(proposed)
  ) {
    for (const key of new Set([...Object.keys(current), ...Object.keys(proposed)])) {
      walk(
        current[key],
        proposed[key],
        `${path}/${escapeToken(key)}`,
        operations,
      );
    }
    return;
  }
  operations.push({ op: "replace", path, value: proposed });
}
