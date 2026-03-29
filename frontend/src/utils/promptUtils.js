export const PROMPT_BUILDER_OPTIONS = {
  length: [
    { value: '', label: 'Any' },
    { value: 'short', label: 'Short' },
    { value: 'medium', label: 'Medium' },
    { value: 'long', label: 'Long' },
  ],
  texture: [
    { value: '', label: 'Any' },
    { value: 'straight', label: 'Straight' },
    { value: 'wavy', label: 'Wavy' },
    { value: 'curly', label: 'Curly' },
  ],
  volume: [
    { value: '', label: 'Any' },
    { value: 'low volume', label: 'Low' },
    { value: 'medium volume', label: 'Medium' },
    { value: 'high volume', label: 'High' },
  ],
  color: [
    { value: '', label: 'Any' },
    { value: 'black', label: 'Black' },
    { value: 'brown', label: 'Brown' },
    { value: 'dark brown', label: 'Dark Brown' },
    { value: 'ash brown', label: 'Ash Brown' },
    { value: 'blonde', label: 'Blonde' },
    { value: 'red', label: 'Red' },
  ],
  style: [
    { value: '', label: 'Generic' },
    { value: 'bob', label: 'Bob' },
    { value: 'layered', label: 'Layered' },
    { value: 'wolf cut', label: 'Wolf Cut' },
    { value: 'pixie', label: 'Pixie' },
    { value: 'lob', label: 'Lob' },
  ],
  bangs: [
    { value: '', label: 'No bangs' },
    { value: 'with see-through bangs', label: 'See-through' },
    { value: 'with full bangs', label: 'Full bangs' },
    { value: 'with side bangs', label: 'Side bangs' },
  ],
}

export const DEFAULT_PROMPT_BUILDER = {
  length: '',
  texture: '',
  volume: '',
  color: '',
  style: '',
  bangs: '',
}

export function buildStructuredPrompt(builderState) {
  const parts = [
    builderState.length,
    builderState.texture,
    builderState.color,
    builderState.volume,
  ].filter(Boolean)

  const stylePart = builderState.style
    ? `${builderState.style} hairstyle`
    : 'hairstyle'

  const prompt = [parts.join(' '), stylePart, builderState.bangs]
    .filter(Boolean)
    .join(', ')
    .replace(/\s+,/g, ',')
    .replace(/\s{2,}/g, ' ')
    .trim()

  return prompt === 'hairstyle' ? '' : prompt
}

export function mergePromptParts(structuredPrompt, freeformPrompt) {
  const freeform = freeformPrompt.trim()
  if (structuredPrompt && freeform) {
    return `${structuredPrompt}, ${freeform}`
  }
  return structuredPrompt || freeform
}

export function getPromptPriorityLabel(value) {
  if (value <= 30) return 'Bám ảnh mẫu'
  if (value >= 70) return 'Ưu tiên prompt'
  return 'Cân bằng'
}

export function getModelPromptSupport(modelId) {
  if (modelId === 'TryOnHairstyle') {
    return {
      tone: 'limited',
      label: 'Reference-only',
      description: 'Model này hiện bám ảnh tham chiếu và không dùng prompt trong inference.',
      promptEnabled: false,
    }
  }

  return {
    tone: 'active',
    label: 'Prompt supported',
    description: 'Prompt hoạt động tốt nhất khi mô tả ngắn, rõ thuộc tính như độ dài, độ xoăn, màu và mái.',
    promptEnabled: true,
  }
}
