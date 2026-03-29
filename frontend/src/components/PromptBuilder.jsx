import { PROMPT_BUILDER_OPTIONS } from '../utils/promptUtils';

const FIELDS = [
  { key: 'length', label: 'Độ dài' },
  { key: 'texture', label: 'Texture' },
  { key: 'volume', label: 'Độ phồng' },
  { key: 'color', label: 'Màu tóc' },
  { key: 'style', label: 'Kiểu tóc' },
  { key: 'bangs', label: 'Mái' },
];

export default function PromptBuilder({ value, onChange, disabled }) {
  return (
    <div className="rounded-2xl border border-[#dbe9e6] bg-[#f8fcfb] p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-gray-800">Prompt Builder</h3>
          <p className="text-xs text-gray-500">Chọn các thuộc tính có cấu trúc để prompt gần với dữ liệu train hơn.</p>
        </div>
        <button
          type="button"
          onClick={() => onChange({ length: '', texture: '', volume: '', color: '', style: '', bangs: '' })}
          disabled={disabled}
          className="rounded-lg border border-gray-200 px-3 py-1.5 text-xs text-gray-500 transition hover:border-gray-300 hover:text-gray-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Đặt lại tag
        </button>
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {FIELDS.map(({ key, label }) => (
          <label key={key} className="flex flex-col gap-1.5">
            <span className="text-xs font-medium uppercase tracking-wide text-gray-500">{label}</span>
            <select
              value={value[key]}
              onChange={(event) => onChange({ ...value, [key]: event.target.value })}
              disabled={disabled}
              className="rounded-xl border border-gray-200 bg-white px-3 py-2.5 text-sm text-gray-700 focus:border-[#2d9b8e] focus:outline-none focus:ring-2 focus:ring-[#2d9b8e]/15 disabled:cursor-not-allowed disabled:bg-gray-100"
            >
              {PROMPT_BUILDER_OPTIONS[key].map((option) => (
                <option key={option.label} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
        ))}
      </div>
    </div>
  );
}
