import { getModelPromptSupport } from '../utils/promptUtils';

const MODELS = [
  {
    id: 'TryHairstyle',
    label: 'TryHairstyle',
    desc: 'Bản nâng cấp có prompt priority và prompt preview.',
  },
  {
    id: 'TryOnHairstyle',
    label: 'TryOnHairstyle',
    desc: 'Bản research gốc, ưu tiên ảnh tham chiếu.',
  },
];

export default function ModelSelector({ value, onChange }) {
  const support = getModelPromptSupport(value);

  return (
    <div className="rounded-2xl border border-[#dbe9e6] bg-gradient-to-br from-[#f8fbfb] to-[#eef6f4] p-4">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-gray-800">Mô hình AI</div>
          <p className="mt-1 text-xs text-gray-500">Chọn pipeline phù hợp với mục tiêu của bạn trước khi điều chỉnh prompt.</p>
        </div>
        <span className={`rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-wide ${support.promptEnabled
          ? 'bg-[#dff4ef] text-[#1a7a6d]'
          : 'bg-gray-200 text-gray-600'}`}>
          {support.label}
        </span>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        {MODELS.map((model) => (
          <button
            key={model.id}
            type="button"
            onClick={() => onChange(model.id)}
            className={`rounded-2xl border p-4 text-left transition ${value === model.id
              ? 'border-[#2d9b8e] bg-white shadow-[0_8px_24px_rgba(45,155,142,0.12)]'
              : 'border-transparent bg-white/75 hover:border-[#b8ddd7] hover:bg-white'}`}
          >
            <div className="mb-2 flex items-center justify-between gap-3">
              <span className="text-base font-semibold text-gray-800">{model.label}</span>
              <span className={`rounded-full px-2.5 py-1 text-[11px] font-medium ${model.id === 'TryHairstyle'
                ? 'bg-[#edf8f6] text-[#1a7a6d]'
                : 'bg-gray-100 text-gray-600'}`}>
                {model.id === 'TryHairstyle' ? 'Prompt ready' : 'Reference-first'}
              </span>
            </div>
            <p className="text-sm leading-6 text-gray-600">{model.desc}</p>
          </button>
        ))}
      </div>

      <p className="mt-3 text-xs text-gray-500">{support.description}</p>
    </div>
  );
}
