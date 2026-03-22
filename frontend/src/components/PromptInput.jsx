export default function PromptInput({ value, onChange }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-sm font-semibold text-gray-700">Text Prompt</label>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Mô tả kiểu tóc mong muốn..."
        rows={2}
        className="w-full rounded-lg border border-gray-200 px-4 py-3 text-sm resize-none
                   focus:outline-none focus:ring-2 focus:ring-[#2d9b8e]/30 focus:border-[#2d9b8e]
                   transition placeholder:text-gray-300"
      />
    </div>
  );
}
