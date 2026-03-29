export default function PromptInput({ value, onChange, language, onLanguageChange, disabled }) {
  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-4">
      <div className="mb-3 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <label className="text-sm font-semibold text-gray-800">Prompt tự do</label>
          <p className="text-xs text-gray-500">Thêm mô tả bổ sung nếu bạn muốn override hoặc bổ sung chi tiết từ Prompt Builder.</p>
        </div>
        <div className="flex items-center gap-1 rounded-lg bg-gray-100 p-0.5">
          <button
            type="button"
            onClick={() => onLanguageChange('en')}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-all duration-200 ${language === 'en'
              ? 'bg-white text-[#2d9b8e] shadow-sm'
              : 'text-gray-500 hover:text-gray-700'}`}
          >
            English
          </button>
          <button
            type="button"
            onClick={() => onLanguageChange('vi')}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-all duration-200 ${language === 'vi'
              ? 'bg-white text-[#2d9b8e] shadow-sm'
              : 'text-gray-500 hover:text-gray-700'}`}
          >
            Tiếng Việt
          </button>
        </div>
      </div>

      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        placeholder={language === 'vi'
          ? 'Ví dụ: tóc ngắn, bob nâu, có mái mỏng...'
          : 'Example: sleek straight finish, soft face-framing strands...'}
        rows={3}
        className="w-full resize-none rounded-xl border border-gray-200 px-4 py-3 text-sm leading-6 focus:border-[#2d9b8e] focus:outline-none focus:ring-2 focus:ring-[#2d9b8e]/20 disabled:cursor-not-allowed disabled:bg-gray-100"
      />

      <p className="mt-2 text-xs text-gray-500">
        {language === 'vi'
          ? 'Prompt tiếng Việt sẽ được dịch sang tiếng Anh trước khi gửi vào TryHairstyle.'
          : 'Prompt ngắn, rõ thuộc tính thường ổn định hơn prompt quá dài hoặc mơ hồ.'}
      </p>
    </div>
  );
}
