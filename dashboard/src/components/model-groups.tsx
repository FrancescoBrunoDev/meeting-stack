import { SelectGroup, SelectItem, SelectLabel } from "@/components/ui/select"

export type ModelChoice = { id: string; recipe?: string }

// A cloud recipe carries its provider; every other recipe (llamacpp, gufo,
// halogen, flm, local) is a runtime served on this box.
const PROVIDERS: Record<string, string> =
  { mistral: "Mistral", opencode: "opencode", unsloth: "Unsloth" }

/** One section per provider, so 90 models stay a dropdown instead of a wall. */
export function ModelGroups({ models }: { models: ModelChoice[] }) {
  const byProvider = new Map<string, ModelChoice[]>()
  for (const model of models) {
    const provider = PROVIDERS[model.recipe ?? ""] ?? "Local"
    byProvider.set(provider, [...(byProvider.get(provider) ?? []), model])
  }
  return [...byProvider.entries()].map(([provider, items]) => (
    <SelectGroup key={provider}>
      <SelectLabel>{provider}</SelectLabel>
      {items.map((model) => (
        <SelectItem key={model.id} value={model.id}>
          {model.id}
        </SelectItem>
      ))}
    </SelectGroup>
  ))
}
