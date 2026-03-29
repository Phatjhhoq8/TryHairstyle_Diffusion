import { getPromptPriorityLabel } from '../utils/promptUtils';

const PRESETS = [
  { label: 'Bám ảnh mẫu', value: 10 },
  { label: 'Cân bằng', value: 50 },
  { label: 'Ưu tiên prompt', value: 85 },
];

export default function PromptPrioritySlider({ value, onChange, disabled }) {
  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-gray-800">Prompt Priority</h3>
          <p className="text-xs text-gray-500">Điều chỉnh mức ưu tiên giữa prompt và ảnh tham chiếu.</p>
        </div>
        <div className="text-right">
          <div className="text-lg font-semibold text-[#1a7a6d]">{value}%</div>
          <div className="text-xs text-gray-500">{getPromptPriorityLabel(value)}</div>
        </div>
      </div>

      <input
        type="range"
        min="0"
        max="100"
        step="5"
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mb-2 w-full accent-[#2d9b8e] disabled:accent-gray-300"
      />

      <div className="mb-3 flex items-center justify-between text-xs text-gray-500">
        <span>Bám ảnh mẫu</span>
        <span>Cân bằng</span>
        <span>Ưu tiên prompt</span>
      </div>

      <div className="flex flex-wrap gap-2">
        {PRESETS.map((preset) => (
          <button
            key={preset.label}
            type="button"
            disabled={disabled}
            onClick={() => onChange(preset.value)}
            className={`rounded-full px-3 py-1.5 text-xs font-medium transition ${value === preset.value
              ? 'bg-[#2d9b8e] text-white shadow-sm'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'} disabled:cursor-not-allowed disabled:bg-gray-100 disabled:text-gray-400`}
          >
            {preset.label}
          </button>
        ))}
      </div>

      <p className="mt-3 text-xs text-gray-500">
        {disabled
          ? 'TryOnHairstyle không hỗ trợ prompt priority, nên thanh này được khóa ở 0%.'
          : 'Kéo sang phải để prompt ảnh hưởng mạnh hơn, nhưng kết quả có thể bám ảnh tham chiếu ít hơn.'}
      </p>
    </div>
  );
}
